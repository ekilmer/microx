/*
 * Copyright (c) 2018 Trail of Bits, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Hand-written CPython binding for microx_core, targeting the Stable ABI
// (Py_LIMITED_API, defined by the build via CMake's USE_SABI). The extension
// therefore builds one abi3 wheel per platform that loads on any CPython >=
// 3.10.

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <type_traits>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "microx/Executor.h"

#ifndef MICROX_HOST_ARCH_STR
#error "MICROX_HOST_ARCH_STR must be defined by the build (x86_64 or aarch64)."
#endif

#if PY_MAJOR_VERSION < 3
#error "Python 2 builds are no longer supported"
#endif

namespace microx {
namespace {

// Extends the executor to invoke Python methods to satisfy requests for
// data from the environment and respond with new values to place into the
// environment.
struct PythonExecutor : public Executor {
  PythonExecutor(PyObject* self_, Arch arch, size_t addr_size);

  virtual ~PythonExecutor(void);

  bool ReadValue(PyObject* res, size_t num_bits, Data& val,
                 const char* usage) const;

  uintptr_t ComputeAddress(const char* seg_name, uintptr_t base,
                           uintptr_t index, uintptr_t scale,
                           uintptr_t displacement, size_t size,
                           MemRequestHint hint) const override;

  bool ReadReg(const char* name, size_t size, RegRequestHint hint,
               Data& val) const override;

  bool WriteReg(const char* name, size_t size, const Data& val) const override;

  bool ReadMem(uintptr_t addr, size_t size, MemRequestHint hint,
               Data& val) const override;

  bool WriteMem(uintptr_t addr, size_t size, const Data& val) const override;

  bool ReadFPU(FPU& val) const override;

  bool WriteFPU(const FPU& val) const override;

  PyObject* const self;
  mutable bool has_error{false};
  mutable PyObject* error{nullptr};
  mutable char error_message[512];
};

// Python representation for an instance of an executor.
struct PythonExecutorObject {
  PyObject_HEAD;
  PythonExecutor* executor;
  typename std::aligned_storage<sizeof(PythonExecutor),
                                alignof(PythonExecutor)>::type impl;
};

// Exception references (owned by the module).
static PyObject* MicroxError{nullptr};
static PyObject* InstructionDecodeError{nullptr};
static PyObject* InstructionFetchError{nullptr};
static PyObject* AddressFaultError{nullptr};
static PyObject* UnsupportedError{nullptr};

// Initialize the exception references.
static bool CreateExceptions(PyObject* module) {
  MicroxError = PyErr_NewException("microx_core.MicroxError", nullptr, nullptr);
  if (!MicroxError ||
      PyModule_AddObjectRef(module, "MicroxError", MicroxError) < 0) {
    return false;
  }

  InstructionDecodeError = PyErr_NewException(
      "microx_core.InstructionDecodeError", MicroxError, nullptr);
  if (!InstructionDecodeError ||
      PyModule_AddObjectRef(module, "InstructionDecodeError",
                            InstructionDecodeError) < 0) {
    return false;
  }

  InstructionFetchError = PyErr_NewException(
      "microx_core.InstructionFetchError", MicroxError, nullptr);
  if (!InstructionFetchError ||
      PyModule_AddObjectRef(module, "InstructionFetchError",
                            InstructionFetchError) < 0) {
    return false;
  }

  AddressFaultError =
      PyErr_NewException("microx_core.AddressFaultError", MicroxError, nullptr);
  if (!AddressFaultError || PyModule_AddObjectRef(module, "AddressFaultError",
                                                  AddressFaultError) < 0) {
    return false;
  }

  UnsupportedError =
      PyErr_NewException("microx_core.UnsupportedError", MicroxError, nullptr);
  if (!UnsupportedError ||
      PyModule_AddObjectRef(module, "UnsupportedError", UnsupportedError) < 0) {
    return false;
  }

  return true;
}

// Map an architecture name to the `Arch` enum. Only the host architecture (the
// one this extension was built for) is supported.
static bool ArchFromName(const char* name, size_t addr_size, Arch& out) {
  if (0 == std::strcmp(name, "auto")) {
    name = MICROX_HOST_ARCH_STR;
  }
  if (0 != std::strcmp(name, MICROX_HOST_ARCH_STR)) {
    return false;
  }
  if (0 == std::strcmp(name, "aarch64")) {
    out = Arch::kAArch64;
  } else {  // "x86_64"
    out = (32 == addr_size) ? Arch::kX86 : Arch::kAMD64;
  }
  return true;
}

static int Executor_init(PyObject* self_, PyObject* args, PyObject* kwargs) {
  static const char* kwlist[] = {"addr_size", "arch", nullptr};
  unsigned addr_size = 0;
  const char* arch_name = "auto";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "I|s",
                                   const_cast<char**>(kwlist), &addr_size,
                                   &arch_name)) {
    return -1;
  }

  if (64 != addr_size && 32 != addr_size) {
    PyErr_Format(
        PyExc_ValueError,
        "Invalid address size %u. Expected 32 or 64 as the address size.",
        addr_size);
    return -1;
  }

  Arch arch;
  if (!ArchFromName(arch_name, addr_size, arch)) {
    PyErr_Format(UnsupportedError,
                 "microx was built for '%s'; cannot execute '%s' instructions.",
                 MICROX_HOST_ARCH_STR, arch_name);
    return -1;
  }

  if (Arch::kAArch64 == arch && 32 == addr_size) {
    PyErr_Format(PyExc_ValueError,
                 "AArch64 (A64) only supports a 64-bit address size.");
    return -1;
  }

  auto self = reinterpret_cast<PythonExecutorObject*>(self_);
  if (self->executor) {  // Guard against __init__ being called twice.
    self->executor->~PythonExecutor();
    self->executor = nullptr;
  }
  self->executor = new (&(self->impl)) PythonExecutor(self_, arch, addr_size);
  return 0;
}

static void Executor_dealloc(PyObject* self_) {
  PyTypeObject* type = Py_TYPE(self_);
  auto self = reinterpret_cast<PythonExecutorObject*>(self_);
  if (self->executor) {
    self->executor->~PythonExecutor();
    self->executor = nullptr;
  }
  auto tp_free = reinterpret_cast<freefunc>(PyType_GetSlot(type, Py_tp_free));
  tp_free(self_);
  Py_DECREF(type);  // Heap-type instances hold a reference to their type.
}

// Emulate an instruction.
static PyObject* Executor_Execute(PyObject* self_, PyObject* args) {
  unsigned long long num_execs = 0;

  if (!PyArg_ParseTuple(args, "K", &num_execs)) {
    PyErr_SetString(PyExc_TypeError,
                    "Invalid value passed to 'execute' method.");
    return nullptr;
  }

  auto self = reinterpret_cast<PythonExecutorObject*>(self_);

  self->executor->has_error = false;
  self->executor->error = nullptr;
  switch (auto error_code =
              self->executor->Execute(static_cast<size_t>(num_execs))) {
    case ExecutorStatus::kGood:
      break;

    case ExecutorStatus::kErrorNotInitialized:
      PyErr_SetString(PyExc_ValueError,
                      "Micro-execution environment is not initialized.");
      return nullptr;

    case ExecutorStatus::kErrorUnsupportedArch:
      PyErr_SetString(UnsupportedError,
                      "This microx build does not support the requested "
                      "architecture.");
      return nullptr;

    case ExecutorStatus::kErrorDecode:
      PyErr_SetString(InstructionDecodeError, "Unable to decode instruction.");
      return nullptr;
    case ExecutorStatus::kErrorUnsupportedFeatures:
    case ExecutorStatus::kErrorUnsupportedCFI:
    case ExecutorStatus::kErrorUnsupportedStack:
      PyErr_SetString(UnsupportedError,
                      "Instruction is not supported by microx.");
      return nullptr;
    case ExecutorStatus::kErrorExecute:
      PyErr_SetString(MicroxError, "Unable to micro-execute instruction.");
      return nullptr;

    case ExecutorStatus::kErrorFault:
      PyErr_SetString(AddressFaultError,
                      "Instruction faulted during micro-execution.");
      return nullptr;

    case ExecutorStatus::kErrorFloatingPointException:
      PyErr_SetString(PyExc_FloatingPointError,
                      "Instruction faulted during micro-execution.");
      return nullptr;

    case ExecutorStatus::kErrorReadInstMem:
      if (!PyErr_Occurred() && !self->executor->error) {
        PyErr_SetString(InstructionFetchError,
                        "Could not read instruction bytes.");
      }
      // fallthrough

    default:
      if (PyErr_Occurred()) {
        // Do nothing, we've got an error already.

      } else if (self->executor->error) {
        PyErr_SetString(self->executor->error, self->executor->error_message);
        self->executor->error = nullptr;

      } else {
        PyErr_Format(PyExc_RuntimeError,
                     "Unable to micro-execute instruction with status %u.",
                     static_cast<unsigned>(error_code));
      }
      return nullptr;
  }

  Py_RETURN_TRUE;
}

static PyMethodDef gExecutorMethods[] = {
    {"execute", Executor_Execute, METH_VARARGS,
     "Interpret a string of bytes as a machine instruction and perform a "
     "micro-execution of the instruction."},
    {nullptr, nullptr, 0, nullptr} /* Sentinel */
};

static PyType_Slot gExecutorSlots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyType_GenericNew)},
    {Py_tp_init, reinterpret_cast<void*>(Executor_init)},
    {Py_tp_dealloc, reinterpret_cast<void*>(Executor_dealloc)},
    {Py_tp_methods, reinterpret_cast<void*>(gExecutorMethods)},
    {Py_tp_doc, const_cast<char*>("Instruction micro-executor.")},
    {0, nullptr},
};

static PyType_Spec gExecutorSpec = {
    "microx_core.Executor",
    sizeof(PythonExecutorObject),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    gExecutorSlots,
};

PythonExecutor::PythonExecutor(PyObject* self_, Arch arch, size_t addr_size)
    : Executor(arch, addr_size), self(self_), error(nullptr) {}

PythonExecutor::~PythonExecutor(void) {}

// Convert a Python value into a `Data` object.
bool PythonExecutor::ReadValue(PyObject* res, size_t num_bits, Data& val,
                               const char* usage) const {
  if (has_error) {
    return false;
  }

  const auto num_bytes = std::min(sizeof(val), (num_bits + 7) / 8);
  if (PyBytes_Check(res)) {
    auto res_size = static_cast<size_t>(PyBytes_Size(res));
    if (num_bytes != res_size) {
      has_error = true;
      error = PyExc_ValueError;
      snprintf(error_message, sizeof(error_message),
               "Incorrect number of bytes returned for value from '%s'; "
               "wanted %zu bytes but got %zu bytes.",
               usage, num_bytes, res_size);
      return false;
    } else {
      memcpy(&(val.bytes[0]), PyBytes_AsString(res), num_bytes);
    }

  } else if (PyLong_Check(res)) {
    // Convert via `int.to_bytes`, which is Stable-ABI-safe and handles the full
    // 512-bit `Data` width. Raises OverflowError for negative / too-wide.
    PyObject* as_bytes = PyObject_CallMethod(
        res, "to_bytes", "is", static_cast<int>(sizeof(val)), "little");
    if (!as_bytes) {
      has_error = true;
      return false;
    }
    memcpy(&(val.bytes[0]), PyBytes_AsString(as_bytes), sizeof(val));
    Py_DECREF(as_bytes);
    return true;

  } else if (PyFloat_Check(res)) {
    if (32 == num_bits) {
      auto f = static_cast<float>(PyFloat_AsDouble(res));
      memcpy(&(val.bytes[0]), &f, sizeof(f));
    } else {
      auto d = PyFloat_AsDouble(res);
      memcpy(&(val.bytes[0]), &d, sizeof(d));
    }
  } else {
    error = PyExc_TypeError;
    PyObject* type_name = PyObject_GetAttrString(
        reinterpret_cast<PyObject*>(Py_TYPE(res)), "__name__");
    const char* tn =
        type_name ? PyUnicode_AsUTF8AndSize(type_name, nullptr) : nullptr;
    snprintf(error_message, sizeof(error_message),
             "Cannot convert type '%s' into a byte sequence from '%s'.",
             tn ? tn : "<unknown>", usage);
    Py_XDECREF(type_name);
    return false;
  }
  memset(&(val.bytes[num_bytes]), 0, sizeof(val) - num_bytes);
  return true;
}

// Perform address computation. The segment register name is passed in so
// that the extender can perform segmented address calculation.
uintptr_t PythonExecutor::ComputeAddress(const char* seg_name, uintptr_t base,
                                         uintptr_t index, uintptr_t scale,
                                         uintptr_t displacement, size_t size,
                                         MemRequestHint hint) const {
  if (has_error) {
    return 0;
  }

  char usage[256];
  auto res = PyObject_CallMethod(
      self, "compute_address", "(s,K,K,K,K,I,i)", seg_name,
      static_cast<unsigned long long>(base),
      static_cast<unsigned long long>(index),
      static_cast<unsigned long long>(scale),
      static_cast<unsigned long long>(displacement),
      static_cast<unsigned>(size / 8), static_cast<int>(hint));

  auto ret_addr = this->Executor::ComputeAddress(seg_name, base, index, scale,
                                                 displacement, size, hint);

  if (res) {
    snprintf(usage, sizeof(usage),
             "compute_address(\"%s\", 0x%08" PRIx64 ", 0x%08" PRIx64
             ", 0x%08" PRIx64 ", 0x%08" PRIx64 ", %u, %d)",
             seg_name, static_cast<uint64_t>(base),
             static_cast<uint64_t>(index), static_cast<uint64_t>(scale),
             static_cast<uint64_t>(displacement),
             static_cast<unsigned>(size / 8), static_cast<int>(hint));
    Data val;
    auto ret = ReadValue(res, addr_size, val, usage);
    Py_DECREF(res);

    if (ret) {
      ret_addr = *reinterpret_cast<uintptr_t*>(val.bytes);
    }

  } else if (PyErr_Occurred()) {
    has_error = true;
  }

  return ret_addr;
}

// Read a register from the environment. The name of the register should make
// the size explicit.
bool PythonExecutor::ReadReg(const char* name, size_t size, RegRequestHint hint,
                             Data& val) const {
  if (has_error) {
    return false;
  }

  char usage[256];
  auto res = PyObject_CallMethod(self, "read_register", "(s,i)", name,
                                 static_cast<int>(hint));
  if (res) {
    snprintf(usage, sizeof(usage), "read_register(\"%s\")", name);
    auto ret = ReadValue(res, size, val, usage);
    Py_DECREF(res);
    return ret;
  } else {
    return false;
  }
}

bool PythonExecutor::WriteReg(const char* name, size_t size,
                              const Data& val) const {
  if (has_error) {
    return false;
  }

  // Build the bytes object explicitly (with an explicit Py_ssize_t length)
  // rather than using a `y#` format code, which behaves differently across
  // Python versions under the limited API.
  PyObject* payload =
      PyBytes_FromStringAndSize(reinterpret_cast<const char*>(val.bytes),
                                static_cast<Py_ssize_t>((size + 7) / 8));
  if (!payload) {
    return false;
  }
  auto ret = PyObject_CallMethod(self, "write_register", "(sO)", name, payload);
  Py_DECREF(payload);
  Py_XDECREF(ret);
  return nullptr != ret;
}

bool PythonExecutor::ReadMem(uintptr_t addr, size_t size, MemRequestHint hint,
                             Data& val) const {
  if (has_error) {
    return false;
  }

  char usage[256];
  auto res = PyObject_CallMethod(
      self, "read_memory", "(K,I,i)", static_cast<unsigned long long>(addr),
      static_cast<unsigned>(size / 8), static_cast<int>(hint));
  if (res) {
    snprintf(usage, sizeof(usage), "read_memory(0x%08" PRIx64 ", %u, %d)",
             static_cast<uint64_t>(addr), static_cast<unsigned>(size / 8),
             static_cast<int>(hint));
    auto ret = ReadValue(res, size, val, usage);
    Py_DECREF(res);
    return ret;
  }

  // Instruction fetches probe descending lengths and fail benignly at the end
  // of mapped memory; swallow that exception so it does not accumulate. All
  // other failures are latched for the caller to raise.
  if (MemRequestHint::kReadExecutable == hint) {
    PyErr_Clear();
  } else if (PyErr_Occurred()) {
    has_error = true;
  }
  return false;
}

bool PythonExecutor::WriteMem(uintptr_t addr, size_t size,
                              const Data& val) const {
  if (has_error) {
    return false;
  }

  PyObject* payload =
      PyBytes_FromStringAndSize(reinterpret_cast<const char*>(val.bytes),
                                static_cast<Py_ssize_t>(size / 8));
  if (!payload) {
    return false;
  }
  auto ret =
      PyObject_CallMethod(self, "write_memory", "(KO)",
                          static_cast<unsigned long long>(addr), payload);
  Py_DECREF(payload);
  Py_XDECREF(ret);
  return nullptr != ret;
}

bool PythonExecutor::ReadFPU(FPU& val) const {
  if (has_error) {
    return false;
  }

  auto res = PyObject_CallMethod(self, "read_fpu", "()");
  if (res) {
    if (!PyBytes_Check(res)) {
      Py_DECREF(res);
      has_error = true;
      error = PyExc_ValueError;
      snprintf(error_message, sizeof(error_message),
               "Expected 'read_fpu' to return string.");
      return false;
    }
    auto res_size = static_cast<size_t>(PyBytes_Size(res));
    if (sizeof(FPU) != res_size) {
      Py_DECREF(res);
      if (!error) {
        error = PyExc_ValueError;
        snprintf(
            error_message, sizeof(error_message),
            "Incorrect number of bytes returned for value from 'read_fpu'; "
            "wanted %zu bytes but got %zu bytes.",
            sizeof(FPU), res_size);
      }
      return false;
    } else {
      memcpy(&(val.bytes[0]), PyBytes_AsString(res), sizeof(FPU));
    }
  }
  Py_XDECREF(res);
  return nullptr != res;
}

bool PythonExecutor::WriteFPU(const FPU& val) const {
  PyObject* payload =
      PyBytes_FromStringAndSize(reinterpret_cast<const char*>(val.bytes),
                                static_cast<Py_ssize_t>(sizeof(val)));
  if (!payload) {
    return false;
  }
  auto ret = PyObject_CallMethod(self, "write_fpu", "(O)", payload);
  Py_DECREF(payload);
  Py_XDECREF(ret);
  return nullptr != ret;
}

static struct PyModuleDef gMicroxModuleDef = {
    PyModuleDef_HEAD_INIT,
    "microx_core",
    "Single-instruction micro-execution support (x86, x86-64, AArch64).",
    -1,  // Global state; not safe for multiple interpreters.
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

}  // namespace
}  // namespace microx

PyMODINIT_FUNC PyInit_microx_core(void) {
  using namespace microx;

  if (!Executor::Init()) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to initialize microx.");
    return nullptr;
  }

  auto module = PyModule_Create(&gMicroxModuleDef);
  if (!module) {
    return nullptr;
  }

  if (!CreateExceptions(module)) {
    Py_DECREF(module);
    return nullptr;
  }

  PyObject* executor_type =
      PyType_FromModuleAndSpec(module, &gExecutorSpec, nullptr);
  if (!executor_type ||
      PyModule_AddObjectRef(module, "Executor", executor_type) < 0) {
    Py_XDECREF(executor_type);
    Py_DECREF(module);
    return nullptr;
  }
  Py_DECREF(executor_type);

  if (PyModule_AddStringConstant(module, "HOST_ARCH", MICROX_HOST_ARCH_STR) <
      0) {
    Py_DECREF(module);
    return nullptr;
  }

  return module;
}

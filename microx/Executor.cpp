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

// This file is the architecture-independent micro-execution driver. It owns the
// phase ordering of a single instruction's execution, the global lock, and the
// POSIX signal-recovery scaffold that catches hardware faults raised while an
// instruction runs natively. Everything architecture-specific lives behind the
// `Backend` interface (see backends/*/), so this translation unit contains no
// decoder, no inline assembly, and no CPython coupling.

#include "microx/Executor.h"

#include <pthread.h>
#include <setjmp.h>
#include <signal.h>

#include <cstdlib>

#include "Backend.h"

namespace microx {
namespace {

// Guards accesses to the process-wide backend state. Using pthreads for
// portability, so that libc++ / libstdc++ doesn't need to be linked in
// (otherwise `std::mutex` would be nicer).
class LockGuard {
 public:
  explicit LockGuard(pthread_mutex_t& lock_) : lock(&lock_) {
    pthread_mutex_lock(lock);
  }
  ~LockGuard(void) { pthread_mutex_unlock(lock); }
  LockGuard(const LockGuard&) = delete;
  LockGuard& operator=(const LockGuard&) = delete;

 private:
  pthread_mutex_t* lock;
};

static pthread_mutex_t gExecutorLock = PTHREAD_MUTEX_INITIALIZER;

// True once `Executor::Init` has run successfully.
static bool gIsInitialized = false;

// The signal number caught during the most recent native execution, or 0.
static int gSignal = 0;

// Recovery point for `siglongjmp` out of the fault handler. Execution of a
// single instruction is wrapped in `sigsetjmp(gRecoveryTarget, ...)`, so a
// hardware fault unwinds here instead of terminating the process.
static sigjmp_buf gRecoveryTarget;

// The handler installed around native execution, and saved originals.
static struct sigaction gFaultHandler;
static struct sigaction gPrevSIGILL;
static struct sigaction gPrevSIGSEGV;
static struct sigaction gPrevSIGBUS;
static struct sigaction gPrevSIGFPE;
static struct sigaction gPrevSIGTRAP;

// An alternate signal stack, so a fault caused by a bad stack pointer in the
// executed instruction is still deliverable. A fixed size is used because
// `SIGSTKSZ` is not a compile-time constant on modern glibc.
alignas(16) static uint8_t gSignalStack[256 * 1024];

// Recover from a signal raised by executing an instruction natively.
[[noreturn]] static void RecoverFromError(int sig) {
  gSignal = sig;
  siglongjmp(gRecoveryTarget, 1);
}

static void InstallSignalHandlers(void) {
  sigaction(SIGILL, &gFaultHandler, &gPrevSIGILL);
  sigaction(SIGSEGV, &gFaultHandler, &gPrevSIGSEGV);
  sigaction(SIGBUS, &gFaultHandler, &gPrevSIGBUS);
  sigaction(SIGFPE, &gFaultHandler, &gPrevSIGFPE);
  sigaction(SIGTRAP, &gFaultHandler, &gPrevSIGTRAP);
}

static void RestoreSignalHandlers(void) {
  sigaction(SIGILL, &gPrevSIGILL, nullptr);
  sigaction(SIGSEGV, &gPrevSIGSEGV, nullptr);
  sigaction(SIGBUS, &gPrevSIGBUS, nullptr);
  sigaction(SIGFPE, &gPrevSIGFPE, nullptr);
  sigaction(SIGTRAP, &gPrevSIGTRAP, nullptr);
}

static ExecutorStatus SignalToStatus(int sig) {
  switch (sig) {
    case 0:
      return ExecutorStatus::kGood;
    case SIGSEGV:
    case SIGBUS:
      return ExecutorStatus::kErrorFault;
    case SIGFPE:
      return ExecutorStatus::kErrorFloatingPointException;
    default:  // SIGILL, SIGTRAP, ...
      return ExecutorStatus::kErrorExecute;
  }
}

}  // namespace

Executor::Executor(Arch arch_, size_t addr_size_, bool has_avx_,
                   bool has_avx512_)
    : arch(arch_),
      addr_size(addr_size_),
      has_avx(has_avx_),
      has_avx512(has_avx512_),
      backend(Backend::Create(arch_, addr_size_, has_avx_, has_avx512_)),
      create_status(backend ? ExecutorStatus::kGood
                            : ExecutorStatus::kErrorUnsupportedArch) {}

Executor::~Executor(void) { delete backend; }

bool Executor::Init(void) {
  LockGuard locker(gExecutorLock);
  if (gIsInitialized) {
    return true;
  }

  if (!Backend::GlobalInit()) {
    return false;
  }

  // Install the alternate signal stack and the fault handler template once.
  stack_t alt_stack;
  alt_stack.ss_sp = gSignalStack;
  alt_stack.ss_size = sizeof(gSignalStack);
  alt_stack.ss_flags = 0;
  sigaltstack(&alt_stack, nullptr);

  gFaultHandler.sa_handler = RecoverFromError;
  gFaultHandler.sa_flags = SA_ONSTACK;
  sigfillset(&(gFaultHandler.sa_mask));

  gIsInitialized = true;
  return true;
}

uintptr_t Executor::ComputeAddress(const char*, uintptr_t base, uintptr_t index,
                                   uintptr_t scale, uintptr_t displacement,
                                   size_t, MemRequestHint) const {
  return base + (index * scale) + displacement;
}

ExecutorStatus Executor::Execute(size_t max_num_executions) {
  if (!max_num_executions) {
    return ExecutorStatus::kGood;
  }

  LockGuard locker(gExecutorLock);

  if (!gIsInitialized) {
    return ExecutorStatus::kErrorNotInitialized;
  }

  if (!backend) {
    return create_status;  // kErrorUnsupportedArch.
  }

  for (size_t num_executed = 0; num_executed < max_num_executions;
       ++num_executed) {
    backend->Reset();

    uintptr_t pc = 0;
    if (!backend->ReadProgramCounter(this, pc)) {
      return ExecutorStatus::kErrorReadReg;
    }

    ExecutorStatus status = backend->FetchDecode(this, pc);
    if (ExecutorStatus::kGood != status) {
      return status;
    }

    status = backend->CheckSupported(this);
    if (ExecutorStatus::kGood != status) {
      return status;
    }

    status = backend->ReadInputs(this);
    if (ExecutorStatus::kGood != status) {
      return status;
    }

    if (!backend->IsNoOp()) {
      status = backend->ReadMemory(this);
      if (ExecutorStatus::kGood != status) {
        return status;
      }
    }

    uintptr_t next_pc = 0;
    ExecutorStatus cf_status = ExecutorStatus::kGood;
    switch (backend->ComputeControlFlow(this, next_pc, cf_status)) {
      case Action::kError:
        return cf_status;

      case Action::kEmulated:
        break;

      case Action::kNative:
        if (!backend->EncodeNative(this)) {
          return ExecutorStatus::kErrorExecute;
        }
        gSignal = 0;
        InstallSignalHandlers();
        if (!sigsetjmp(gRecoveryTarget, 1)) {
          backend->RunNative();
        }
        RestoreSignalHandlers();
        status = SignalToStatus(gSignal);
        if (ExecutorStatus::kGood != status) {
          return status;
        }
        break;
    }

    status = backend->WriteOutputs(this, next_pc);
    if (ExecutorStatus::kGood != status) {
      return status;
    }
  }

  return ExecutorStatus::kGood;
}

}  // namespace microx

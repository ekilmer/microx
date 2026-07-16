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

#ifndef MICROX_BACKENDS_AARCH64_AARCH64STATE_H_
#define MICROX_BACKENDS_AARCH64_AARCH64STATE_H_

// Byte offsets into the guest execution-state block that the generated arena
// and the assembly trampoline read and write. These MUST stay in sync with
// `struct State` below (a static_assert enforces it for the C++ side).

#define MICROX_ST_VEC 0    /* uint8_t  vec[32][16]  (v0..v31, 16-byte each) */
#define MICROX_ST_GPR 512  /* uint64_t gpr[31]      (x0..x30) */
#define MICROX_ST_SP 760   /* uint64_t sp */
#define MICROX_ST_NZCV 768 /* uint64_t nzcv (PSTATE.NZCV in bits [31:28]) */
#define MICROX_ST_FPCR 776 /* uint64_t fpcr */
#define MICROX_ST_FPSR 784 /* uint64_t fpsr */
#define MICROX_ST_ARENA                                                    \
  792                     /* uint64_t arena_entry (generated code address) \
                           */
#define MICROX_ST_RET 800 /* uint64_t ret_addr    (&microx_arm64_return) */

#ifndef __ASSEMBLER__

#include <cstddef>
#include <cstdint>

namespace microx {
namespace aarch64 {

// The guest register file, laid out for the generated arena to load/store with
// simple base+offset addressing. Vectors come first so they are 16-byte
// aligned for `ldp/stp q`.
struct alignas(16) State {
  uint8_t vec[32][16];   // v0..v31
  uint64_t gpr[31];      // x0..x30
  uint64_t sp;           // stack pointer (distinct from xzr)
  uint64_t nzcv;         // condition flags
  uint64_t fpcr;         // FP control
  uint64_t fpsr;         // FP status (cumulative exception bits)
  uint64_t arena_entry;  // entry point of the generated instruction sequence
  uint64_t ret_addr;     // address to branch back to (microx_arm64_return)
};

static_assert(offsetof(State, vec) == MICROX_ST_VEC, "State::vec offset");
static_assert(offsetof(State, gpr) == MICROX_ST_GPR, "State::gpr offset");
static_assert(offsetof(State, sp) == MICROX_ST_SP, "State::sp offset");
static_assert(offsetof(State, nzcv) == MICROX_ST_NZCV, "State::nzcv offset");
static_assert(offsetof(State, fpcr) == MICROX_ST_FPCR, "State::fpcr offset");
static_assert(offsetof(State, fpsr) == MICROX_ST_FPSR, "State::fpsr offset");
static_assert(offsetof(State, arena_entry) == MICROX_ST_ARENA,
              "State::arena_entry offset");
static_assert(offsetof(State, ret_addr) == MICROX_ST_RET, "State::ret offset");

}  // namespace aarch64
}  // namespace microx

// The assembly trampoline. `microx_arm64_enter` saves host state, then branches
// to `state->arena_entry`; the generated arena branches back to
// `microx_arm64_return`, whose address the harness stores in `state->ret_addr`.
extern "C" void microx_arm64_enter(microx::aarch64::State* state);
extern "C" void microx_arm64_return(void);

#endif  // __ASSEMBLER__

#endif  // MICROX_BACKENDS_AARCH64_AARCH64STATE_H_

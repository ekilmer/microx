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

/*
 * AArch64 micro-execution trampoline.
 *
 * `microx_arm64_enter(State *state)` (x0 = state) saves the host's
 * callee-saved registers — x19..x28, the frame pointer x29, the link register
 * x30, the reserved platform register x18, and the low 64 bits of the
 * callee-saved SIMD registers v8..v15 (required by the AArch64 PCS) — then
 * branches into the generated arena at `state->arena_entry`, passing `state`
 * in x10. The arena loads the guest register file, runs the single guest
 * instruction, stores the results back, and branches to `state->ret_addr`,
 * which is `microx_arm64_return`. That path restores the host registers and
 * returns.
 *
 * The real stack pointer is never loaded with the guest SP (the arena aliases
 * a guest SP data operand to a scratch register), so the host stack — and
 * hence signal delivery and this epilogue — stays valid throughout.
 */

#include "AArch64State.h"

#if defined(__APPLE__)
#define SYM(name) _##name
#else
#define SYM(name) name
#endif

    .text
    .balign 4

    .globl SYM(microx_arm64_enter)
    .globl SYM(microx_arm64_return)

SYM(microx_arm64_enter):
    bti c
    /* Frame: 176 bytes. Save x19..x30 (host), x18 (platform), v8..v15 (host). */
    stp x19, x20, [sp, #-176]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    stp d8,  d9,  [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    /* x18 is the reserved platform register; preserve it across the call. */
    str x18, [sp, #160]

    mov x10, x0                     /* x10 = state (passed to the arena) */
    ldr x9, [x0, #MICROX_ST_ARENA]  /* x9  = arena entry */
    br  x9                          /* -> generated arena */

SYM(microx_arm64_return):
    bti j
    ldr x18, [sp, #160]
    ldp x19, x20, [sp]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    ldp x29, x30, [sp, #80]
    ldp d8,  d9,  [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    add sp, sp, #176
    ret

#if defined(__linux__)
/* Mark the stack non-executable. */
    .section .note.GNU-stack, "", %progbits
#endif

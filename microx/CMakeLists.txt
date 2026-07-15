cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(microx CXX)

# --- Host-native architecture selection ---------------------------------------
# A given build only supports its host ISA: the x86 backend is compiled only on x86-family hosts,
# the AArch64 backend only on arm64 hosts.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
  set(MICROX_HOST_ARCH "x86_64")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(MICROX_HOST_ARCH "aarch64")
else ()
  message(FATAL_ERROR "microx: unsupported host ISA '${CMAKE_SYSTEM_PROCESSOR}'")
endif ()

# --- Decoder acquisition, per architecture ------------------------------------
add_library(microx_decoder INTERFACE)

if (MICROX_HOST_ARCH STREQUAL "x86_64")
  # Intel XED is built out-of-band by scripts/bootstrap.sh into third_party/. XED_DIR overrides the
  # location; otherwise fall back to the in-repo copy.
  set(_xed_root "$ENV{XED_DIR}")
  if (NOT _xed_root)
    set(_xed_root "${CMAKE_SOURCE_DIR}/third_party")
  endif ()
  find_library(
    LIBXED
    NAMES xed
    PATHS "${_xed_root}"
    PATH_SUFFIXES "lib" REQUIRED
  )
  target_link_libraries(microx_decoder INTERFACE "${LIBXED}")
  target_include_directories(microx_decoder SYSTEM INTERFACE "${_xed_root}/include")
  set(MICROX_BACKEND_SOURCES backends/x86/X86Backend.cpp)
else ()
  # Capstone v6 (next) provides AArch64 decode with register access info. Pinned to a commit that
  # includes the AArch64 register-offset access fix (PR #2929); bump deliberately.
  include(FetchContent)
  set(CAPSTONE_ARCHITECTURE_DEFAULT
      OFF
      CACHE BOOL "" FORCE
  )
  set(CAPSTONE_AARCH64_SUPPORT
      ON
      CACHE BOOL "" FORCE
  )
  set(CAPSTONE_BUILD_SHARED_LIBS
      OFF
      CACHE BOOL "" FORCE
  )
  set(CAPSTONE_BUILD_STATIC_LIBS
      ON
      CACHE BOOL "" FORCE
  )
  set(CAPSTONE_BUILD_CSTOOL
      OFF
      CACHE BOOL "" FORCE
  )
  set(CAPSTONE_INSTALL
      OFF
      CACHE BOOL "" FORCE
  )
  FetchContent_Declare(
    capstone
    GIT_REPOSITORY "https://github.com/capstone-engine/capstone.git"
    GIT_TAG "ca197a0f89bc302a64c85cd21c92947f076dc2b6" # next @ 2026-07, incl. PR #2929
    GIT_SHALLOW FALSE
  )
  FetchContent_MakeAvailable(capstone)
  # Link the real static archive (`capstone` itself is an OBJECT library, whose objects do not embed
  # into our static lib).
  target_link_libraries(microx_decoder INTERFACE capstone_static)
  target_include_directories(microx_decoder SYSTEM INTERFACE "${capstone_SOURCE_DIR}/include")

  enable_language(ASM)
  set(MICROX_BACKEND_SOURCES
      backends/aarch64/AArch64Backend.cpp backends/aarch64/AArch64Harness.cpp
      backends/aarch64/AArch64Trampoline.S
  )
endif ()

# --- Architecture-independent library -----------------------------------------
add_library(microx Executor.cpp ${MICROX_BACKEND_SOURCES})
target_include_directories(
  microx PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_include_directories(microx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(microx PRIVATE microx_decoder)
set_target_properties(
  microx PROPERTIES PUBLIC_HEADER "include/microx/Executor.h" POSITION_INDEPENDENT_CODE ON
)

# Install the C++ library and its public header for standalone consumers.
if (NOT SKBUILD)
  install(
    TARGETS microx
    RUNTIME DESTINATION "bin"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    PUBLIC_HEADER DESTINATION "include/microx"
  )
endif ()

# --- Python extension ---------------------------------------------------------
if (MICROX_PYTHON_BINDINGS)
  # scikit-build-core sets SKBUILD_SABI_COMPONENT; fall back for a plain `cmake
  # -DMICROX_PYTHON_BINDINGS=ON` configure.
  if (NOT DEFINED SKBUILD_SABI_COMPONENT)
    set(SKBUILD_SABI_COMPONENT "Development.SABIModule")
  endif ()
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
  # USE_SABI defines Py_LIMITED_API=0x030A0000; WITH_SOABI yields the microx_core.abi3.so suffix.
  python_add_library(microx_core MODULE WITH_SOABI USE_SABI 3.10 Python.cpp)
  target_compile_definitions(microx_core PRIVATE MICROX_HOST_ARCH_STR="${MICROX_HOST_ARCH}")
  target_link_libraries(microx_core PRIVATE microx)
  install(TARGETS microx_core LIBRARY DESTINATION .)
  install(FILES "${CMAKE_SOURCE_DIR}/microx_core.pyi" DESTINATION .)
endif ()

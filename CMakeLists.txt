cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(microx CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      "${CMAKE_BINARY_DIR}"
      CACHE PATH "Default install directory" FORCE
  )
endif ()

# Build the hand-written CPython extension (microx_core). scikit-build-core turns this on; a plain
# C++ consumer leaves it off and just gets the library.
option(MICROX_PYTHON_BINDINGS "Build the microx_core Python extension" OFF)

add_subdirectory(microx)

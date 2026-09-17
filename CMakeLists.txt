cmake_minimum_required(VERSION 3.24)
project(driveeval LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(DE_NATIVE      "Tune for the host CPU. Off for portable CI binaries." ON)
option(DE_ENABLE_OSQP "Build the OSQP refinement backend." ON)
option(DE_SANITIZE    "Build with address and undefined-behaviour sanitizers." OFF)
option(DE_BUILD_TESTS "Build the test suite." ON)
option(DE_BUILD_FUZZ  "Build libFuzzer targets. Needs clang." OFF)

add_library(drive_flags INTERFACE)
target_compile_options(drive_flags INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast -Wnon-virtual-dtor -Wdouble-promotion
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:RelWithDebInfo>:-O2>)

# Record the exact tuning flag so the harness can print it alongside every
# latency number. An unlabelled number is not a number.
set(DE_ARCH_FLAG "generic")
if(DE_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" DE_HAS_MCPU_NATIVE)
  if(DE_HAS_MCPU_NATIVE)
    target_compile_options(drive_flags INTERFACE -mcpu=native)
    set(DE_ARCH_FLAG "-mcpu=native")
  else()
    check_cxx_compiler_flag("-march=native" DE_HAS_MARCH_NATIVE)
    if(DE_HAS_MARCH_NATIVE)
      target_compile_options(drive_flags INTERFACE -march=native)
      set(DE_ARCH_FLAG "-march=native")
    endif()
  endif()
endif()

if(DE_SANITIZE)
  target_compile_options(drive_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(drive_flags INTERFACE -fsanitize=address,undefined)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
add_library(drive_eigen INTERFACE)
target_include_directories(drive_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

if(DE_ENABLE_OSQP)
  # OSQP 0.6.3 declares cmake_minimum_required(VERSION 3.2), which CMake 4
  # refuses outright. Raising the policy floor for the subproject is the least
  # invasive fix; the alternative is OSQP 1.x, whose C API changed enough to
  # rewrite the backend against.
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
  set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  set(UNITTESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v0.6.3
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(osqp)
endif()

if(DE_ENABLE_OSQP)
  set(DE_OSQP_01 1)
else()
  set(DE_OSQP_01 0)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/driveeval/build_info.hpp @ONLY)

file(GLOB_RECURSE DE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(drive_core STATIC ${DE_SOURCES})
target_include_directories(drive_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(drive_core PUBLIC drive_flags drive_eigen)
if(DE_ENABLE_OSQP)
  # Built via add_subdirectory, so the in-tree target name applies; the
  # osqp:: namespace exists only for an installed export set.
  target_link_libraries(drive_core PUBLIC osqpstatic)
  target_compile_definitions(drive_core PUBLIC DE_WITH_OSQP=1)
else()
  target_compile_definitions(drive_core PUBLIC DE_WITH_OSQP=0)
endif()
find_package(Threads REQUIRED)
target_link_libraries(drive_core PUBLIC Threads::Threads)

file(GLOB DE_APPS CONFIGURE_DEPENDS apps/drive_*.cpp)
foreach(app_src ${DE_APPS})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} PRIVATE drive_core)
endforeach()
# drive_bench links the allocation-counting operator new so that it can report heap
# traffic inside the planning cycle. Nothing else does, so nothing else pays.
target_sources(drive_bench PRIVATE tests/alloc_hook.cpp)

if(DE_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  file(GLOB DE_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
  add_executable(drive_tests ${DE_TEST_SOURCES})
  target_include_directories(drive_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(drive_tests PRIVATE drive_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(drive_tests DISCOVERY_TIMEOUT 60)
endif()

if(DE_BUILD_FUZZ)
  # Apple's clang does not ship libFuzzer, so probe for it rather than failing
  # at link time with a missing-library error that looks like a project bug.
  # tests/test_cache.cpp carries an always-on random-mutation harness over the
  # same entry point, so the property is checked on every build either way;
  # these targets are the deeper, coverage-guided version and CI runs them on
  # Linux where the runtime exists.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-fsanitize=fuzzer")
  check_cxx_source_compiles(
    "#include <cstddef>\n#include <cstdint>\nextern \"C\" int LLVMFuzzerTestOneInput(const uint8_t*, size_t){return 0;}"
    DE_HAS_LIBFUZZER)
  unset(CMAKE_REQUIRED_FLAGS)
  if(DE_HAS_LIBFUZZER)
    foreach(f fuzz_cache_reader fuzz_lane_graph)
      add_executable(${f} fuzz/${f}.cpp)
      target_link_libraries(${f} PRIVATE drive_core)
      target_compile_options(${f} PRIVATE -fsanitize=fuzzer,address,undefined)
      target_link_options(${f} PRIVATE -fsanitize=fuzzer,address,undefined)
    endforeach()
  else()
    message(WARNING
      "DE_BUILD_FUZZ requested but this toolchain has no libFuzzer runtime "
      "(Apple clang does not ship one). Skipping the fuzz targets. The "
      "random-mutation test in tests/test_cache.cpp still runs.")
  endif()
endif()

message(STATUS "driveeval: arch flag ${DE_ARCH_FLAG}, OSQP ${DE_ENABLE_OSQP}, sanitize ${DE_SANITIZE}")

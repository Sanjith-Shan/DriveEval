cmake_minimum_required(VERSION 3.24)
project(switchback LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SB_NATIVE      "Tune for the host CPU. Off for portable CI binaries." ON)
option(SB_ENABLE_OSQP "Build the OSQP refinement backend." ON)
option(SB_SANITIZE    "Build with address and undefined-behaviour sanitizers." OFF)
option(SB_BUILD_TESTS "Build the test suite." ON)
option(SB_BUILD_FUZZ  "Build libFuzzer targets. Needs clang." OFF)

add_library(sb_flags INTERFACE)
target_compile_options(sb_flags INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
  -Wold-style-cast -Wnon-virtual-dtor -Wdouble-promotion
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:RelWithDebInfo>:-O2>)

# Record the exact tuning flag so the harness can print it alongside every
# latency number. An unlabelled number is not a number.
set(SB_ARCH_FLAG "generic")
if(SB_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" SB_HAS_MCPU_NATIVE)
  if(SB_HAS_MCPU_NATIVE)
    target_compile_options(sb_flags INTERFACE -mcpu=native)
    set(SB_ARCH_FLAG "-mcpu=native")
  else()
    check_cxx_compiler_flag("-march=native" SB_HAS_MARCH_NATIVE)
    if(SB_HAS_MARCH_NATIVE)
      target_compile_options(sb_flags INTERFACE -march=native)
      set(SB_ARCH_FLAG "-march=native")
    endif()
  endif()
endif()

if(SB_SANITIZE)
  target_compile_options(sb_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(sb_flags INTERFACE -fsanitize=address,undefined)
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
add_library(sb_eigen INTERFACE)
target_include_directories(sb_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

if(SB_ENABLE_OSQP)
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

if(SB_ENABLE_OSQP)
  set(SB_OSQP_01 1)
else()
  set(SB_OSQP_01 0)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build_info.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/switchback/build_info.hpp @ONLY)

file(GLOB_RECURSE SB_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(sb_core STATIC ${SB_SOURCES})
target_include_directories(sb_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(sb_core PUBLIC sb_flags sb_eigen)
if(SB_ENABLE_OSQP)
  # Built via add_subdirectory, so the in-tree target name applies; the
  # osqp:: namespace exists only for an installed export set.
  target_link_libraries(sb_core PUBLIC osqpstatic)
  target_compile_definitions(sb_core PUBLIC SB_WITH_OSQP=1)
else()
  target_compile_definitions(sb_core PUBLIC SB_WITH_OSQP=0)
endif()
find_package(Threads REQUIRED)
target_link_libraries(sb_core PUBLIC Threads::Threads)

file(GLOB SB_APPS CONFIGURE_DEPENDS apps/sb_*.cpp)
foreach(app_src ${SB_APPS})
  get_filename_component(app_name ${app_src} NAME_WE)
  add_executable(${app_name} ${app_src})
  target_link_libraries(${app_name} PRIVATE sb_core)
endforeach()
# sb_bench links the allocation-counting operator new so that it can report heap
# traffic inside the planning cycle. Nothing else does, so nothing else pays.
target_sources(sb_bench PRIVATE tests/alloc_hook.cpp)

if(SB_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  file(GLOB SB_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
  add_executable(sb_tests ${SB_TEST_SOURCES})
  target_include_directories(sb_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(sb_tests PRIVATE sb_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(sb_tests DISCOVERY_TIMEOUT 60)
endif()

if(SB_BUILD_FUZZ)
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
    SB_HAS_LIBFUZZER)
  unset(CMAKE_REQUIRED_FLAGS)
  if(SB_HAS_LIBFUZZER)
    foreach(f fuzz_cache_reader fuzz_lane_graph)
      add_executable(${f} fuzz/${f}.cpp)
      target_link_libraries(${f} PRIVATE sb_core)
      target_compile_options(${f} PRIVATE -fsanitize=fuzzer,address,undefined)
      target_link_options(${f} PRIVATE -fsanitize=fuzzer,address,undefined)
    endforeach()
  else()
    message(WARNING
      "SB_BUILD_FUZZ requested but this toolchain has no libFuzzer runtime "
      "(Apple clang does not ship one). Skipping the fuzz targets. The "
      "random-mutation test in tests/test_cache.cpp still runs.")
  endif()
endif()

message(STATUS "switchback: arch flag ${SB_ARCH_FLAG}, OSQP ${SB_ENABLE_OSQP}, sanitize ${SB_SANITIZE}")

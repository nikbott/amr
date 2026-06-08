cmake_minimum_required(VERSION 3.20)

# Project Metadata
project(
  AMR_System
  VERSION 1.0.0
  DESCRIPTION "High-performance Parallel Adaptive Mesh Refinement Engine"
  LANGUAGES CXX)

# --- Custom Build Types (Profiling) -------------------------------------------
# -O3: Max optimization -g: Debug symbols (needed for profilers to resolve
# function names) -fno-omit-frame-pointer: Crucial for accurate stack unwinding
# in perf/VTune -DNDEBUG: Disable assertions
set(CMAKE_CXX_FLAGS_PROFILING
    "-O3 -g -fno-omit-frame-pointer -DNDEBUG"
    CACHE STRING "Flags used by the C++ compiler during profiling builds."
          FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILING
    ""
    CACHE STRING "Flags used for linking binaries during profiling builds."
          FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILING
    ""
    CACHE STRING
          "Flags used by the shared libraries linker during profiling builds."
          FORCE)

mark_as_advanced(CMAKE_CXX_FLAGS_PROFILING CMAKE_EXE_LINKER_FLAGS_PROFILING
                 CMAKE_SHARED_LINKER_FLAGS_PROFILING)

# --- Best Practice: Default to Release ----------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' for maximum performance.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Register the custom type so GUI tools pick it up
  set_property(
    CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                    "RelWithDebInfo" "Profiling")
endif()

# --- Options ------------------------------------------------------------------
option(BUILD_TESTING "Build the testing suite" ON)
option(USE_NATIVE_ARCH "Optimize for the current machine architecture" ON)
option(ENABLE_LTO "Enable Link Time Optimization (IPO)" ON)
option(ENABLE_SANITIZERS
       "Enable Address/Undefined Behavior Sanitizers (Debug only)" OFF)
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# Backend feature flags. OpenMP (omp/) always builds; MPI and CUDA are opt-in.
option(AMR_BUILD_MPI "Build the distributed MPI backend (mpi/)" OFF)
option(AMR_BUILD_CUDA "Build the single-GPU CUDA backend (cuda/)" OFF)
set(AMR_CUDA_ARCHITECTURES
    "70"
    CACHE STRING
          "CUDA arch(es) for the cuda/ backend, e.g. 80 (A100), 90 (H100)")

# --- Compiler Configuration ---------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

# --- LTO Configuration (Performance) ------------------------------------------
if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO is not supported: ${output}")
  endif()
endif()

# --- Dependencies -------------------------------------------------------------
if(BUILD_TESTING)
  include(CTest)
  # Prefer a system Catch2 (no network); fall back to fetching it.
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.11.0)
    FetchContent_MakeAvailable(Catch2)
  endif()
endif()

# --- Library Target (AMR Core) ------------------------------------------------
add_library(core INTERFACE)

target_include_directories(core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/omp)

target_link_libraries(core INTERFACE OpenMP::OpenMP_CXX)

# Compile Options & Warnings (Correctness)
if(MSVC)
  target_compile_options(core INTERFACE /W4 /permissive- /Zc:__cplusplus)
  if(WARNINGS_AS_ERRORS)
    target_compile_options(core INTERFACE /WX)
  endif()
else()
  target_compile_options(core INTERFACE -Wall -Wextra -Wpedantic -Wconversion
                                        -Wshadow -Wformat=2)

  if(WARNINGS_AS_ERRORS)
    target_compile_options(core INTERFACE -Werror)
  endif()

  if(USE_NATIVE_ARCH)
    # target_compile_options(core INTERFACE -march=native)
  endif()
endif()

# --- Executable: Simulation ---------------------------------------------------
add_executable(amr omp/main.cpp)
target_link_libraries(amr PRIVATE core)

# --- Executable: Balance benchmark (active-front vs reference)
# -----------------
add_executable(amr_omp_bench omp/bench.cpp)
target_link_libraries(amr_omp_bench PRIVATE core)

# --- Executable: Tests --------------------------------------------------------
if(BUILD_TESTING)
  add_executable(amr_tests omp/tests.cpp)
  target_link_libraries(amr_tests PRIVATE core Catch2::Catch2WithMain)
  add_test(NAME AllTests COMMAND amr_tests)
endif()

# --- Sanitizers (Correctness/Debug) -------------------------------------------
if(ENABLE_SANITIZERS AND NOT MSVC)
  # Automatically disable sanitizers for Profiling builds to ensure accuracy
  if(CMAKE_BUILD_TYPE MATCHES "Profiling")
    message(
      STATUS
        "Sanitizers disabled for Profiling build to ensure accurate timing.")
  else()
    target_compile_options(amr PRIVATE -fsanitize=address,undefined)
    target_link_options(amr PRIVATE -fsanitize=address,undefined)

    if(BUILD_TESTING)
      target_compile_options(amr_tests PRIVATE -fsanitize=address,undefined)
      target_link_options(amr_tests PRIVATE -fsanitize=address,undefined)
    endif()
  endif()
endif()

# --- MPI backend (mpi/) -------------------------------------------------------
# Opt-in: -DAMR_BUILD_MPI=ON. One rank may also use OpenMP threads internally.
if(AMR_BUILD_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  message(STATUS "MPI backend enabled: ${MPIEXEC_EXECUTABLE}")

  add_executable(amr_mpi mpi/main.cpp)
  target_link_libraries(amr_mpi PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_compile_features(amr_mpi PRIVATE cxx_std_20)

  if(BUILD_TESTING)
    add_executable(amr_mpi_tests mpi/tests.cpp)
    target_link_libraries(amr_mpi_tests PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX
                                                Catch2::Catch2)
    target_compile_features(amr_mpi_tests PRIVATE cxx_std_20)
    # mpi/tests.cpp provides its own MPI-aware main (Catch2::Catch2, not
    # WithMain).
    add_test(
      NAME MpiTests
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:amr_mpi_tests> ${MPIEXEC_POSTFLAGS})
  endif()
endif()

# --- CUDA backend (cuda/) -----------------------------------------------------
# Opt-in: -DAMR_BUILD_CUDA=ON. Thrust device lambdas require --extended-lambda.
if(AMR_BUILD_CUDA)
  enable_language(CUDA)
  message(STATUS "CUDA backend enabled (arch ${AMR_CUDA_ARCHITECTURES})")

  add_executable(amr_cuda cuda/main.cu)
  set_target_properties(
    amr_cuda
    PROPERTIES CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               CUDA_ARCHITECTURES "${AMR_CUDA_ARCHITECTURES}")
  target_compile_options(amr_cuda
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

  # Active-front vs reference balance() benchmark (cuda/bench.cu).
  add_executable(amr_cuda_bench cuda/bench.cu)
  set_target_properties(
    amr_cuda_bench
    PROPERTIES CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               CUDA_ARCHITECTURES "${AMR_CUDA_ARCHITECTURES}")
  target_compile_options(amr_cuda_bench
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

  if(BUILD_TESTING)
    # cuda/tests.cu uses a self-contained CHECK/REQUIRE harness (no Catch2).
    add_executable(amr_cuda_tests cuda/tests.cu)
    set_target_properties(
      amr_cuda_tests
      PROPERTIES CUDA_STANDARD 17
                 CUDA_STANDARD_REQUIRED ON
                 CUDA_ARCHITECTURES "${AMR_CUDA_ARCHITECTURES}")
    target_compile_options(
      amr_cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    add_test(NAME CudaTests COMMAND amr_cuda_tests)
  endif()
endif()

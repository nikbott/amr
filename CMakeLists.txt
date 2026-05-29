cmake_minimum_required(VERSION 3.20)

# Project Metadata
project(AMR_System 
    VERSION 1.0.0
    DESCRIPTION "High-performance Parallel Adaptive Mesh Refinement Engine"
    LANGUAGES CXX
)

# --- Custom Build Types (Profiling) -------------------------------------------
# -O3: Max optimization
# -g: Debug symbols (needed for profilers to resolve function names)
# -fno-omit-frame-pointer: Crucial for accurate stack unwinding in perf/VTune
# -DNDEBUG: Disable assertions
set(CMAKE_CXX_FLAGS_PROFILING "-O3 -g -fno-omit-frame-pointer -DNDEBUG" CACHE STRING "Flags used by the C++ compiler during profiling builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILING "" CACHE STRING "Flags used for linking binaries during profiling builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILING "" CACHE STRING "Flags used by the shared libraries linker during profiling builds." FORCE)

mark_as_advanced(
    CMAKE_CXX_FLAGS_PROFILING
    CMAKE_EXE_LINKER_FLAGS_PROFILING
    CMAKE_SHARED_LINKER_FLAGS_PROFILING
)

# --- Best Practice: Default to Release ----------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' for maximum performance.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Register the custom type so GUI tools pick it up
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Profiling")
endif()

# --- Options ------------------------------------------------------------------
option(BUILD_TESTING "Build the testing suite" ON)
option(USE_NATIVE_ARCH "Optimize for the current machine architecture" ON)
option(ENABLE_LTO "Enable Link Time Optimization (IPO)" ON)
option(ENABLE_SANITIZERS "Enable Address/Undefined Behavior Sanitizers (Debug only)" OFF)
option(WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

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
include(FetchContent)

if(BUILD_TESTING)
    include(CTest)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.11.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# --- Library Target (AMR Core) ------------------------------------------------
add_library(core INTERFACE)

target_include_directories(core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/omp
)

target_link_libraries(core INTERFACE OpenMP::OpenMP_CXX)

# Compile Options & Warnings (Correctness)
if(MSVC)
    target_compile_options(core INTERFACE /W4 /permissive- /Zc:__cplusplus)
    if(WARNINGS_AS_ERRORS)
        target_compile_options(core INTERFACE /WX)
    endif()
else()
    target_compile_options(core INTERFACE 
        -Wall -Wextra -Wpedantic 
        -Wconversion -Wshadow -Wformat=2
    )
    
    if(WARNINGS_AS_ERRORS)
        target_compile_options(core INTERFACE -Werror)
    endif()
    
    if(USE_NATIVE_ARCH)
        #target_compile_options(core INTERFACE -march=native)
    endif()
endif()

# --- Executable: Simulation ---------------------------------------------------
add_executable(amr omp/main.cpp)
target_link_libraries(amr PRIVATE core)

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
        message(STATUS "Sanitizers disabled for Profiling build to ensure accurate timing.")
    else()
        target_compile_options(amr PRIVATE -fsanitize=address,undefined)
        target_link_options(amr PRIVATE -fsanitize=address,undefined)
        
        if(BUILD_TESTING)
            target_compile_options(amr_tests PRIVATE -fsanitize=address,undefined)
            target_link_options(amr_tests PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endif()
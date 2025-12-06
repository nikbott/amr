cmake_minimum_required(VERSION 3.18)
project(AMR_CUDA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA acceleration" OFF)

# Source files
set(SOURCES
    main.cpp
)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # Adjust based on target GPU
    
    add_definitions(-DUSE_CUDA)
    
    list(APPEND SOURCES
        tree_kernels.cu
    )
endif()

add_executable(amr ${SOURCES})

if(USE_CUDA)
    set_target_properties(amr PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    # Linking cudart adds the include directories for cuda_runtime.h to main.cpp
    target_link_libraries(amr PRIVATE CUDA::cudart)
endif()

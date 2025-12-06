cmake_minimum_required(VERSION 3.18)
project(AMR_CUDA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    main.cpp
)

add_executable(amr ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(amr PUBLIC OpenMP::OpenMP_CXX)
endif()

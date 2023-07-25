cmake_minimum_required(VERSION 3.11)
project(llama.c)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS_DEBUG "-g")

option(LLAMA_C_OFAST   "llama.c: enable -Ofast flag"               ON)
option(LLAMA_C_NATIVE  "llama.c: enable -march=native flag"        OFF)

# Enable flags
if (LLAMA_C_OFAST)
    add_compile_options(-Ofast)
endif()
if (LLAMA_C_NATIVE)
    add_compile_options(-march=native)
endif()

# Add the executable target
add_executable(run run.c)

# The OpenMP-enabled build
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Clean target
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove run
)

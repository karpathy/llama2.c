cmake_minimum_required (VERSION 3.16.3)
# option(BUILD_TESTS ON "Build unit tests")
project(llama2cpp
        VERSION 0.1.0
        LANGUAGES CXX 
        DESCRIPTION "llama2 inference in C++"
        )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

include_directories(include)

add_executable(
    run_llama src/main.cpp
    include/llama2cpp/ops.hpp
    include/llama2cpp/utils.hpp
    include/llama2cpp/sampler.hpp
    include/llama2cpp/tokenizer.hpp
    include/llama2cpp/transformer.hpp
    include/llama2cpp/llama2.hpp
)
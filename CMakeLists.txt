cmake_minimum_required(VERSION 3.12)
project(llama2c)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

add_executable(llama2c run.c)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(llama2c PRIVATE -Wall -Wextra -Ofast)
endif()

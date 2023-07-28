cmake_minimum_required(VERSION 3.12)
project(llama2c)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

option(USE_OPENMP "Build with openmp support" OFF)

add_executable(llama2c run.c)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_C_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(llama2c PRIVATE -Wall -Wextra -Ofast)
  target_link_libraries(llama2c PRIVATE m)
endif()

if(USE_OPENMP)
  find_package(OpenMP)

  if(OpenMP_C_FOUND)
    target_compile_options(llama2c PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(llama2c PRIVATE ${OpenMP_C_LIBRARIES})
  else()
    message(WARNING "OpenMP not found")
  endif()
endif()

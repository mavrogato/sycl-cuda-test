
cmake_minimum_required(VERSION 3.29)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Wno-unknown-cuda-version")

project(sycl-cuda-test)

add_executable(sycl-cuda-test
  sycl-cuda-test.cc)

add_custom_target(run
  DEPENDS ./sycl-cuda-test
  COMMAND ./sycl-cuda-test)

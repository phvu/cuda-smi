cmake_minimum_required(VERSION 2.6)

cmake_policy(SET CMP0048 NEW)
project(cuda-semi LANGUAGES CXX C)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

file(GLOB SRC_FILES  *.cpp)
cuda_add_executable(cuda-semi ${SRC_FILES})

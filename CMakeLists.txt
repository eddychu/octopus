cmake_minimum_required(VERSION 3.21)
project(octopus CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(octopus main.cu)

#add cuda architecture
set_property(TARGET octopus PROPERTY CUDA_ARCHITECTURES 75)

set_target_properties(octopus PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

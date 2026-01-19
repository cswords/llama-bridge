cmake_minimum_required(VERSION 3.20)
project(llama_chat_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Path to llama.cpp (passed via -DLLAMA_DIR=...)
if(NOT DEFINED LLAMA_DIR)
    set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp")
endif()

message(STATUS "LLAMA_DIR: ${LLAMA_DIR}")

# Include directories
include_directories(
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
)

# Link directories
link_directories(
    ${LLAMA_DIR}/build
)

# Create the pybind11 module
pybind11_add_module(llama_chat
    llama_chat_wrapper.cpp
)

# Link against llama.cpp libraries
target_link_libraries(llama_chat PRIVATE
    llama
    common
)

# Set output name
set_target_properties(llama_chat PROPERTIES
    OUTPUT_NAME "llama_chat"
)

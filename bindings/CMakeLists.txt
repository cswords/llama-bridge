cmake_minimum_required(VERSION 3.20)
project(llama_chat_bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    ${LLAMA_DIR}/ggml/src
    ${LLAMA_DIR}/vendor
)

# Link directories - libraries are in build/bin
set(LLAMA_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/llama.cpp/build")
link_directories(
    ${LLAMA_BUILD_DIR}/bin
)

# Create the pybind11 module
pybind11_add_module(llama_chat
    llama_chat_wrapper.cpp
)

# Link against llama.cpp libraries
target_link_libraries(llama_chat PRIVATE
    ${LLAMA_BUILD_DIR}/bin/libllama.dylib
    ${LLAMA_BUILD_DIR}/common/libcommon.a
)

# macOS specific: set rpath for finding dylibs at runtime
if(APPLE)
    set_target_properties(llama_chat PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path/lib"
    )
endif()

# Set output name
set_target_properties(llama_chat PROPERTIES
    OUTPUT_NAME "llama_chat"
)

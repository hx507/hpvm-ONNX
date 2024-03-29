# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

if(NOT ONNXRUNTIME_ROOTDIR)
    set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/lib/onnxruntime")
endif()

#TODO: we should only need one of them.
message(STATUS "Use onnxruntime at: ${ONNXRUNTIME_ROOTDIR}")
include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# On Linux the samples use libjpeg and libpng for decoding images.
# On Windows they use Windows Image Component(WIC)
find_package(JPEG)
if(LIBPNG_ROOTDIR)
    set(PNG_FOUND true)
    set(PNG_LIBRARIES png16)
    set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
    set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
else()
    find_package(PNG)
endif()

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_OPENVINO)
    add_definitions(-DUSE_OPENVINO)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
    add_definitions(-DUSE_NNAPI)
endif()
if(onnxruntime_USE_DNNL)
    add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_NUPHAR)
    add_definitions(-DUSE_NUPHAR)
endif()
if(onnxruntime_USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
    message("Enabling DML")
    add_definitions(-DUSE_DML)
endif()

add_subdirectory(src)

################################################################################
#
# MIT License
#
# Copyright (c) 2018 - 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.10)
project(migraphx_node_test)
set(CMAKE_CXX_STANDARD 14)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/mivisionx)
link_directories(${ROCM_PATH}/lib)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
find_package(migraphx QUIET)
find_package(OpenCV REQUIRED)
if(migraphx_FOUND)
    add_executable(${PROJECT_NAME} migraphx_node_test.cpp)
    target_link_libraries(${PROJECT_NAME} migraphx::c openvx vx_amd_migraphx)
else()
    message("-- ERROR!: MIGraphX not Found, please install MIGrpahX!-- ${PROJECT_NAME} excluded${ColourReset}")
endif()
if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
    target_compile_definitions(migraphx_node_test PUBLIC ENABLE_OPENCV=1)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(migraphx_node_test ${OpenCV_LIBRARIES})
    if(${OpenCV_VERSION_MAJOR} EQUAL 4)
        target_compile_definitions(migraphx_node_test PUBLIC USE_OPENCV_4=1)
    else()
        target_compile_definitions(migraphx_node_test PUBLIC USE_OPENCV_4=0)
    endif()
    else()
    target_compile_definitions(migraphx_node_test PUBLIC ENABLE_OPENCV=0)
    message("-- NOTE: migraphx_node_test -- OpenCV Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
    endif()
else()
    message("-- ERROR!: OpenCV not Found, please install OpenCV!-- ${PROJECT_NAME} excluded${ColourReset}")
endif()


# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)
project(runcl)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL QUIET)

list(APPEND SOURCES runcl.cpp)

add_executable(runcl ${SOURCES})
# install MIVisionX executables -- {ROCM_PATH}/bin
install(TARGETS runcl DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
install(TARGETS runcl DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT dev)

if(OpenCL_FOUND)
	target_compile_definitions(runcl PUBLIC ENABLE_OPENCL=1)
	include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
	target_link_libraries(runcl ${OpenCL_LIBRARIES})
else(OpenCL_FOUND)
	target_compile_definitions(runcl PUBLIC ENABLE_OPENCL=0)
endif(OpenCL_FOUND)

if(WIN32)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")
	find_package(OpenSSL QUIET)
	if(OPENSSL_FOUND)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OpenSSL=1")
		target_link_libraries(runcl crypto)
	endif(OPENSSL_FOUND)
endif()

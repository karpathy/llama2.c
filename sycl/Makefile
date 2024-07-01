 # Modifications Copyright (C) 2024 Intel Corporation
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom
 # the Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included
 # in all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 # OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 # OR OTHER DEALINGS IN THE SOFTWARE.
 #
 # SPDX-License-Identifier: MIT
 #

CC := icpx

LD := $(CC)

LIB :=

FLAGS := -O2 -fsycl -ffast-math

ifeq ($(shell which $(CC)),)
    $(error ERROR - $(CC) compiler not found)
endif

ROOT_DIR     := $(shell dirname $(shell which $(CC)))
INCLUDE_SYCL := $(ROOT_DIR)/../include
INCLUDE_CL   := $(ROOT_DIR)/../include/sycl

TARGET_0_SRC_1 = ./run.sycl.cpp
TARGET_0_FLAG_1 = -std=c++17 -I $(INCLUDE_SYCL) -I $(INCLUDE_CL) ${FLAGS}

.PHONY: sycl
sycl:
	icpx $(TARGET_0_FLAG_1) ${TARGET_0_SRC_1} -o runsycl

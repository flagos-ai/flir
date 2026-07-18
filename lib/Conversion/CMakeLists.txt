# Copyright 2026 FlagOS Contributors
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

if(NOT FLAGTREE_BACKEND STREQUAL "tsingmicro")
  add_subdirectory(TritonToLinalgExperimental)
  add_subdirectory(TritonArithToLinalg)
  add_subdirectory(StructuredToMemref)
  add_subdirectory(ReconcilePtrCasts)
endif()
add_subdirectory(TritonToLinalg)
add_subdirectory(TritonToStructured)
add_subdirectory(TritonToUnstructured)
add_subdirectory(TritonPtrToMemref)
add_subdirectory(UnstructuredToMemref)
add_subdirectory(MemrefCopyToDMA_FlagTree)
add_subdirectory(NoBufferize_FlagTree)
if (FLIR_BUILD_INCUBATED)
  add_subdirectory(TritonToLinalgIncubated)
  add_subdirectory(DiscreteMaskAccessConversion)
  add_subdirectory(TritonToUnstructureIncubated)
  add_subdirectory(TritonToStructuredIncubated)
endif()

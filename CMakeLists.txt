cmake_minimum_required(VERSION 3.24)
project(2022_SEAI_project_SVM)

set(CMAKE_CXX_STANDARD 14)
include_directories(external)
include_directories(external/cppposit_private)
include_directories(external/cppposit_private/examples)
include_directories(external/cppposit_private/experimental)
include_directories(external/cppposit_private/experimental/simd)
include_directories(external/cppposit_private/experimental/vector)
include_directories(external/cppposit_private/extern)
include_directories(external/cppposit_private/include)
include_directories(external/cppposit_private/include/floats)
include_directories(external/cppposit_private/include/interface)
include_directories(external/cppposit_private/include/interface/cereal)
include_directories(external/cppposit_private/include/interface/logger)
include_directories(external/cppposit_private/src)
include_directories(external/cppposit_private/tests)
include_directories(external/cppposit_private/tests/vect_test_pack)

add_executable(2022_SEAI_project_SVM
         main_wb.cpp)



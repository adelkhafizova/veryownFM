cmake_minimum_required(VERSION 3.10)
project(FM)

set(CMAKE_CXX_STANDARD 14)

add_executable(FM main.cpp src/factorization_machine.h src/utils.h src/fm_sgd.h src/utils.cpp src/fm_als.h src/factorization_machine.cpp src/fm_sgd.cpp)

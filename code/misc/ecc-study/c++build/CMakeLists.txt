cmake_minimum_required(VERSION 3.15)
project(dynamics_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(_dynamics
    src/dynamics.cpp
    src/bindings.cpp
)

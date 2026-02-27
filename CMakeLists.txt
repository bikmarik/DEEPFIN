cmake_minimum_required(VERSION 3.12)
project(FinancialAI VERSION 0.1.0 LANGUAGES CXX)

# 1. Force C++11 or higher (Required by pybind11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find Python and pybind11
# This looks for the headers (Python.h) and the development libraries
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 3. Create the Python Module
# Note: 'calculate_data' must match the name in your PYBIND11_MODULE(name, m)
pybind11_add_module(calculate_data dataGenius/calculate_data.cpp)

# 4. Link the Python headers
target_include_directories(calculate_data PRIVATE ${Python_INCLUDE_DIRS})
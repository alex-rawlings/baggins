
# Needed for C++17 support
cmake_minimum_required(VERSION 3.8)
# Require standard C++17. When this is set here, the requirement will be
# automatically enabled for all targets.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(PYBIND11_PYTHON_VERSION 3.8)

project(bagginsCXX LANGUAGES CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Release, RelWithDebInfo, Debug or MinSizeRel)."
      FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

# Warning flags based on the compiler
set(WARNING_FLAGS
    $<$<CXX_COMPILER_ID:GNU>:
        -Wall -Wextra -Wpedantic
    >
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -Wall -Wextra -Wpedantic -Wno-missing-braces 
    >
)

find_package(OpenMP)
include(FindPythonInterp)
include(FindPythonLibs)

add_subdirectory("thirdparty/pybind11")
pybind11_add_module(bagginsCXX "src/python_bindings.cpp")
target_include_directories(bagginsCXX PRIVATE "include/" ${Boost_INCLUDE_DIRS})
target_compile_options(bagginsCXX PRIVATE ${WARNING_FLAGS})
target_link_libraries(bagginsCXX PRIVATE OpenMP::OpenMP_CXX)

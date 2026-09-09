cmake_minimum_required(VERSION 3.18)
project(gapese_cpp VERSION 0.1.0 LANGUAGES CXX)

option(GAPESE_BUILD_PYTHON "Build the pybind11 Python module" ON)
option(GAPESE_BUILD_TESTS "Build C++ tests" ON)
option(GAPESE_FETCH_DEPENDENCIES "Download Eigen/pybind11 when not found" ON)
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path containing the Eigen headers, e.g. <path>/eigen-3.4.0")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    if(EIGEN3_INCLUDE_DIR)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    elseif(GAPESE_FETCH_DEPENDENCIES)
        message(STATUS "Eigen3 not found; fetching Eigen 3.4.0")
        FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(eigen)
    else()
        message(FATAL_ERROR "Eigen3 was not found. Set EIGEN3_INCLUDE_DIR or enable GAPESE_FETCH_DEPENDENCIES.")
    endif()
endif()

add_library(gapese_core STATIC
    src/types.cpp
    src/dgap.cpp
    src/fb.cpp
    src/kkt.cpp
    src/merit.cpp
    src/solver.cpp)
add_library(gapese::core ALIAS gapese_core)

target_include_directories(gapese_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(gapese_core PUBLIC Eigen3::Eigen)
set_target_properties(gapese_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(gapese_core PRIVATE /O2 /W4)
else()
    target_compile_options(gapese_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(GAPESE_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        if(GAPESE_FETCH_DEPENDENCIES)
            message(STATUS "pybind11 not found; fetching pybind11 2.12.0")
            FetchContent_Declare(
                pybind11
                URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
            FetchContent_MakeAvailable(pybind11)
        else()
            message(FATAL_ERROR "pybind11 was not found. Install pybind11 or enable GAPESE_FETCH_DEPENDENCIES.")
        endif()
    endif()

    pybind11_add_module(gapese python/pybind_module.cpp)
    target_link_libraries(gapese PRIVATE gapese_core)
    install(TARGETS gapese LIBRARY DESTINATION . RUNTIME DESTINATION .)
endif()

if(GAPESE_BUILD_TESTS)
    enable_testing()

    add_executable(test_gapese_simple tests/cpp/test_simple.cpp)
    target_link_libraries(test_gapese_simple PRIVATE gapese_core)
    add_test(NAME gapese_simple COMMAND test_gapese_simple)

    add_executable(test_gapese_options tests/cpp/test_options.cpp)
    target_link_libraries(test_gapese_options PRIVATE gapese_core)
    add_test(NAME gapese_options COMMAND test_gapese_options)
endif()

install(TARGETS gapese_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

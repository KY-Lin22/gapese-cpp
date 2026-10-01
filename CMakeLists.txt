cmake_minimum_required(VERSION 3.18)
project(gapese_cpp VERSION 0.1.0 LANGUAGES C CXX)

# CMake 3.24+ prefers extraction-time timestamps for URL-based FetchContent.
# Guard the policy so the project remains compatible with CMake 3.18.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_GAPESE_IS_TOP_LEVEL ON)
else()
    set(_GAPESE_IS_TOP_LEVEL OFF)
endif()

option(GAPESE_BUILD_PYTHON "Build the pybind11 Python module" OFF)
option(GAPESE_BUILD_TESTS "Build C++ tests" OFF)
option(GAPESE_BUILD_EXAMPLES "Build C++ examples" OFF)
option(GAPESE_FETCH_DEPENDENCIES "Download Eigen/pybind11 when not found" ON)
option(GAPESE_INSTALL "Generate Gapese install and package rules" ${_GAPESE_IS_TOP_LEVEL})
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path containing the Eigen headers, e.g. <path>/eigen-3.4.0")

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Eigen is part of the public C++ interface, so the exported Gapese target
# keeps Eigen3::Eigen as a transitive dependency.
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 QUIET CONFIG)
endif()

if(NOT TARGET Eigen3::Eigen)
    if(EIGEN3_INCLUDE_DIR)
        if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
            message(FATAL_ERROR
                "EIGEN3_INCLUDE_DIR='${EIGEN3_INCLUDE_DIR}' does not contain Eigen/Core.")
        endif()
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    elseif(GAPESE_FETCH_DEPENDENCIES)
        message(STATUS "Eigen3 not found; fetching Eigen 3.4.0 headers")
        # Eigen is header-only. Populate its source tree without adding Eigen's
        # own CMake project, so Gapese's install/export remains self-contained.
        FetchContent_Declare(
            gapese_eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            SOURCE_SUBDIR _gapese_header_only)
        FetchContent_MakeAvailable(gapese_eigen)
        if(NOT EXISTS "${gapese_eigen_SOURCE_DIR}/Eigen/Core")
            message(FATAL_ERROR "Fetched Eigen source does not contain Eigen/Core.")
        endif()
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${gapese_eigen_SOURCE_DIR}")
    else()
        message(FATAL_ERROR
            "Eigen3 was not found. Set EIGEN3_INCLUDE_DIR or enable GAPESE_FETCH_DEPENDENCIES.")
    endif()
endif()

add_library(gapese_core STATIC
    src/types.cpp
    src/dgap.cpp
    src/fb.cpp
    src/kkt.cpp
    src/merit.cpp
    src/solver.cpp
    third_party/qdldl/src/qdldl.c)
add_library(gapese::core ALIAS gapese_core)
add_library(gapese::gapese ALIAS gapese_core)

target_compile_features(gapese_core PUBLIC cxx_std_17)
target_include_directories(gapese_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdldl/include)
target_link_libraries(gapese_core PUBLIC Eigen3::Eigen)
set_target_properties(gapese_core PROPERTIES
    CXX_EXTENSIONS OFF
    EXPORT_NAME gapese
    POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(gapese_core PRIVATE /W4)
else()
    target_compile_options(gapese_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(GAPESE_BUILD_PYTHON)
    if(NOT TARGET pybind11::module)
        find_package(pybind11 2.13 CONFIG QUIET)
    endif()
    if(NOT TARGET pybind11::module)
        if(GAPESE_FETCH_DEPENDENCIES)
            message(STATUS "pybind11 not found; fetching pybind11 2.13.6")
            FetchContent_Declare(
                pybind11
                URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz)
            FetchContent_MakeAvailable(pybind11)
        else()
            message(FATAL_ERROR
                "pybind11 2.13 or newer was not found. Install pybind11 or enable GAPESE_FETCH_DEPENDENCIES.")
        endif()
    endif()

    pybind11_add_module(gapese python/pybind_module.cpp)
    target_link_libraries(gapese PRIVATE gapese::gapese)
    if(GAPESE_INSTALL)
        install(TARGETS gapese
            LIBRARY DESTINATION . COMPONENT Python
            RUNTIME DESTINATION . COMPONENT Python)
    endif()
endif()

if(GAPESE_BUILD_EXAMPLES)
    add_executable(gapese_example_simple examples/cpp/simple_ccqp.cpp)
    target_link_libraries(gapese_example_simple PRIVATE gapese::gapese)
endif()

if(GAPESE_BUILD_TESTS)
    enable_testing()

    add_executable(test_gapese_simple tests/cpp/test_simple.cpp)
    target_link_libraries(test_gapese_simple PRIVATE gapese::gapese)
    add_test(NAME gapese_simple COMMAND test_gapese_simple)

    add_executable(test_gapese_options tests/cpp/test_options.cpp)
    target_link_libraries(test_gapese_options PRIVATE gapese::gapese)
    add_test(NAME gapese_options COMMAND test_gapese_options)
endif()

if(GAPESE_INSTALL)
    set(GAPESE_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/gapese")

    install(TARGETS gapese_core
        EXPORT gapeseTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Development)
    install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT Development)
    install(FILES LICENSE
        DESTINATION "${CMAKE_INSTALL_DATADIR}/gapese"
        COMPONENT Development)
    install(FILES third_party/qdldl/LICENSE
        DESTINATION "${CMAKE_INSTALL_DATADIR}/gapese/licenses"
        RENAME QDLDL-LICENSE
        COMPONENT Development)

    configure_package_config_file(
        cmake/gapeseConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/gapeseConfig.cmake"
        INSTALL_DESTINATION "${GAPESE_CMAKE_INSTALL_DIR}")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/gapeseConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMinorVersion)

    install(EXPORT gapeseTargets
        FILE gapeseTargets.cmake
        NAMESPACE gapese::
        DESTINATION "${GAPESE_CMAKE_INSTALL_DIR}"
        COMPONENT Development)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/gapeseConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/gapeseConfigVersion.cmake"
        DESTINATION "${GAPESE_CMAKE_INSTALL_DIR}"
        COMPONENT Development)
endif()

cmake_minimum_required(VERSION 3.16)

project(vamm_variational_tests LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to repository root (vamm/cpp/tests -> vamm/)
get_filename_component(VAMM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

include(GoogleTest)

enable_testing()

# ---------------------------------------------------------------------------
# Validate dependencies
# ---------------------------------------------------------------------------

if(NOT EXISTS "${VAMM_ROOT}/vamm/extern/eigen/Eigen")
    message(FATAL_ERROR
        "Eigen not found. Run: git submodule update --init --recursive")
endif()

# ---------------------------------------------------------------------------
# Test sources
# ---------------------------------------------------------------------------

set(VAMM_TEST_SOURCES
    test_variational_pure.cpp
    test_variational_init.cpp
    test_variational_io.cpp
    test_variational_em.cpp
    test_mixture.cpp
    test_diagonal.cpp
    test_mfa.cpp
    test_full.cpp
)

# ---------------------------------------------------------------------------
# Shared test configuration
# ---------------------------------------------------------------------------

add_library(vamm_test_config INTERFACE)

target_compile_features(
    vamm_test_config
    INTERFACE
        cxx_std_17
)

target_include_directories(
    vamm_test_config
    INTERFACE
        "${VAMM_ROOT}/vamm/extern/eigen"
        "${VAMM_ROOT}/vamm/cpp/include"
)

set(BOOST_COMPONENTS
    unordered
    assert
    container_hash
    config
    core
    predef
    throw_exception
    mp11
    describe
    static_assert
)

foreach(component IN LISTS BOOST_COMPONENTS)
    target_include_directories(
        vamm_test_config
        INTERFACE
            "${VAMM_ROOT}/vamm/extern/boost/${component}/include"
    )
endforeach()

target_compile_definitions(
    vamm_test_config
    INTERFACE
        PRECISION_T=double
        EIGEN_DONT_PARALLELIZE
        BOOST_ALLOW_DEPRECATED_HEADERS
)

# if(NOT MSVC)
target_compile_options(
    vamm_test_config
    INTERFACE
        -Wall
        -Wextra
        -pedantic
)
# endif()

# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

foreach(test_source IN LISTS VAMM_TEST_SOURCES)

    get_filename_component(
        test_name
        "${test_source}"
        NAME_WE
    )

    add_executable(
        "${test_name}"
        "${test_source}"
    )

    target_link_libraries(
        "${test_name}"
        PRIVATE
            vamm_test_config
            GTest::gtest_main
            OpenMP::OpenMP_CXX
    )

    gtest_discover_tests("${test_name}")

endforeach()
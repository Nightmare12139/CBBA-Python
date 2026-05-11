cmake_minimum_required(VERSION 3.16)

project(cbba_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cbba_cpp
    src/CBBA.cpp
    src/Config.cpp
    src/HelperLibrary.cpp
)

target_include_directories(cbba_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cbba_example_01
    examples/run_cbba_example_01.cpp
)

target_link_libraries(cbba_example_01 PRIVATE cbba_cpp)

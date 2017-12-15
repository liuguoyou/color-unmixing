cmake_minimum_required(VERSION 3.1)

project(ColorUnmixing)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(unmixing)

add_executable(ColorUnmixingTest test.cpp)
target_link_libraries(ColorUnmixingTest ColorUnmixing)

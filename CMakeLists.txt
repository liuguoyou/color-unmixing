cmake_minimum_required(VERSION 3.1)
enable_testing()

project(ColorUnmixing)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(unmixing)

add_executable(ColorUnmixingTest test.cpp)
target_link_libraries(ColorUnmixingTest ColorUnmixing)

add_test(
  NAME test
  COMMAND $<TARGET_FILE:ColorUnmixingTest> ${CMAKE_SOURCE_DIR}/resources/leaves.jpg ${PROJECT_BINARY_DIR}
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

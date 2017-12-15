cmake_minimum_required(VERSION 2.8)

find_package(Eigen3 REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(NLopt REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Qt5Gui_INCLUDE_DIRS})
include_directories(${NLopt_INCLUDE_DIRS})

file(GLOB files *.cpp *.h)
add_library(ColorUnmixing STATIC ${files})
target_link_libraries(ColorUnmixing Qt5::Gui ${NLopt_LIBRARIES})

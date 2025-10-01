cmake_minimum_required(VERSION 3.10)
project(ceres_hello)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(fit fit.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(fit ${CERES_LIBRARIES} ${OpenCV_LIBS})





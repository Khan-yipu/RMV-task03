cmake_minimum_required(VERSION 3.10)
project(ceres_hello)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(trajectory trajectory.cpp)
target_link_libraries(trajectory ${OpenCV_LIBS})

add_executable(fit fit.cpp)
target_link_libraries(fit ${CERES_LIBRARIES})




cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
# 判断EIGEN3_INCLUDE_DIR环境变量是否定义
if(DEFINED ENV{EIGEN3_INCLUDE_DIR})
    message("EIGEN3_INCLUDE_DIR: $ENV{EIGEN3_INCLUDE_DIR}")
    include_directories($ENV{EIGEN3_INCLUDE_DIR})
else()
    message("NOT DEFINED EIGEN3_INCLUDE_DIR VARIABLES")
endif()

# 判断OpenCV_INCLUDES环境变量是否定义
if(DEFINED ENV{OpenCV_INCLUDES})
    message("OpenCV_INCLUDES: $ENV{OpenCV_INCLUDES}")
    include_directories($ENV{OpenCV_INCLUDES})
else()
    message("NOT DEFINED OpenCV_INCLUDES VARIABLES")
endif()

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp)

# 判断OpenCV_LIBRARIES环境变量是否定义
if(DEFINED ENV{OpenCV_LIBRARIES})
    message("OpenCV_LIBRARIES: $ENV{OpenCV_LIBRARIES}")
    target_link_libraries(Rasterizer $ENV{OpenCV_LIBRARIES})
else()
    message("NOT DEFINED OpenCV_LIBRARIES VARIABLES")
endif()


cmake_minimum_required(VERSION 2.8.3)
project(pcd_calib)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf
  cv_bridge
  image_geometry
  image_transport
  image_view
)
#find_package(Threads REQUIRED)
catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${TF_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
link_directories(${TF_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})


add_executable(pcl_calib src/pcl_calib.cpp)




target_link_libraries(pcl_calib ${catkin_LIBRARIES}
 ${PCL_LIBRARIES}
 ${TF_LIBRARIES}
 )

#target_link_libraries(pcl_calib PRIVATE Threads::Threads)





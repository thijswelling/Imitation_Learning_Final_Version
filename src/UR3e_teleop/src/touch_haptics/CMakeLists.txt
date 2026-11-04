cmake_minimum_required(VERSION 3.8)
project(touch_haptics)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(touch_haptic_bridge src/touch_haptic_bridge.cpp)
ament_target_dependencies(touch_haptic_bridge rclcpp geometry_msgs)
target_link_libraries(touch_haptic_bridge HD HDU rt pthread)

install(TARGETS
  touch_haptic_bridge
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

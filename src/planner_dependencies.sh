#!/bin/bash

# Update package list
sudo apt-get update

# ROS Navigation Planner dependencies
sudo apt-get install -y ros-noetic-navigation

sudo apt-get install -y ros-noetic-laser-drivers
sudo apt-get install -y ros-noetic-depth-camera-drivers
sudo apt-get install ros-noetic-octomap-server
sudo apt-get install ros-noetic-map-server
sudo apt-get install ros-noetic-amcl
sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-rviz
sudo apt-get install ros-noetic-dwa-local-planner

echo "ROS Navigation Planner dependencies installed successfully."

source /opt/ros/noetic/setup.bash

exit 0


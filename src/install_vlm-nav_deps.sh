#!/bin/bash
#
# This script is used to install all the system dependencies for the pff_vlm


# Update and upgrade the system packages
sudo apt update
sudo apt upgrade -y

# Install system packages
sudo apt install -y python3 python3-pip python3-dev python3-tk

# Install ROS dependencies
source /opt/ros/noetic/setup.bash
rosdep update
rosdep init
rosdep install --from-paths ./ --ignore-src -r -y

echo "System-level dependencies installed and ROS dependencies set up."



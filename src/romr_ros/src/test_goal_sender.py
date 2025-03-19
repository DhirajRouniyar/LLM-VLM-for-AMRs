#!/usr/bin/env python

# Common functions for Image Processing with OpenCV
#
# Copyright (c) 2024 Piaggio Fast Forward (PFF), Inc.
# All Rights Reserved. Reproduction, publication,
# or re-transmittal not allowed without written permission of PFF, Inc.

# import the necessary packages

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def send_goal(x, y, z, w):
    rospy.init_node('test_goal_sender')
    
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()
    
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = 0
    goal.target_pose.pose.orientation.z = z
    goal.target_pose.pose.orientation.w = w
   
    
    client.send_goal(goal)
    client.wait_for_result()
    
    return client.get_result()

if __name__ == '__main__':
    try:
        result = send_goal(0, 0, 0.95, 1)
        rospy.loginfo("Goal result: %s", result)
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")


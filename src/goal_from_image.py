#!/usr/bin/env python

# Common functions for Image Processing with OpenCV
#
# Copyright (c) 2024 Piaggio Fast Forward (PFF), Inc.
# All Rights Reserved. Reproduction, publication,
# or re-transmittal not allowed without written permission of PFF, Inc.

# import the necessary packages

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
import tf
import numpy as np
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError

class ObjectNavigator:
    def __init__(self):
        rospy.init_node('object_navigator', anonymous=True)
        
        self.bridge = CvBridge()
        self.intrinsics = None

        self.llm_output_sub = rospy.Subscriber('/llm_output', String, self.llm_output_callback)
        self.camera_info_sub = rospy.Subscriber('/rgb/camera_info', CameraInfo, self.camera_info_callback)
        
        self.tf_listener = tf.TransformListener()
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server()

    def camera_info_callback(self, msg):
        self.intrinsics = np.array(msg.K).reshape(3, 3)

    def llm_output_callback(self, msg):
        data = msg.data
        try:
            # Parsing the message
            text_index_start = data.index('Text ') + len('Text ')
            text_index_end = data.index(':', text_index_start)
            text_index = data[text_index_start:text_index_end].strip()
            
            label_start = data.index('Image ') + len('Image ')
            label_end = data.index(',', label_start)
            label = data[label_start:label_end].strip()
            
            score_start = data.index('Score ') + len('Score ')
            score_end = data.index(',', score_start)
            score = float(data[score_start:score_end].strip())
            
            bbox_start = data.index('Bounding Box [') + len('Bounding Box [')
            bbox_end = data.index(']', bbox_start)
            bbox = eval(data[bbox_start:bbox_end].strip())
            
            center_start = data.index('Center (') + len('Center (')
            center_end = data.index(')', center_start)
            center = eval(data[center_start:center_end].strip())
            u, v = center
            print("Here", u, v)
            depth_start = data.index('Depth') + len('Depth ')
            depth_end = data.index("\n", depth_start) if '\n' in data[depth_start:] else len(data)
            depth = float(data[depth_start:depth_end].strip())

            # Convert image coordinates to camera coordinates
            camera_coords = self.image_to_camera_coords(u, v, depth)

            # Transform camera coordinates to world coordinates
            world_coords = self.transform_camera_to_world(camera_coords)
            rospy.loginfo(f"world_coords_in_map: {world_coords}")

            self.publish_goal(world_coords)

        except Exception as e:
            rospy.logerr(f"Failed to parse message: {e}")

    def image_to_camera_coords(self, u, v, depth):
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        print("x:", x)
        print("y", y)
        print("Depth:", depth)
        return np.array([x, y, z])

    def transform_camera_to_world(self, camera_coords):
        try:
            # Get the transformation from camera frame to world frame
            self.tf_listener.waitForTransform('map', 'camerad435i_link', rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform('map', 'camerad435i_link', rospy.Time(0))
            rospy.loginfo(f"Trans, Rot: {trans, rot}")
            rospy.loginfo(f"Camera_coords: {camera_coords}")
            # Create transformation matrices
            rotation_matrix = tf.transformations.quaternion_matrix(rot)[:3, :3]
            translation_vector = np.array(trans)
            
            # Apply the transformation
            world_coords = np.dot(rotation_matrix, camera_coords) + translation_vector

            return world_coords
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
            return None

    def publish_goal(self, world_coords):
        goal = MoveBaseGoal()
        
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = world_coords[0]
        goal.target_pose.pose.position.y = world_coords[2] - 1.5 # chair - 1.5 and for others - 1.2
        goal.target_pose.pose.position.z = 0  # Typically, z is 0 in the map frame
        goal.target_pose.pose.orientation.w = 1
        
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
        rospy.loginfo("Goal sent to the robot")

        return self.move_base_client.get_result()

if __name__ == '__main__':
    try:
        ObjectNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

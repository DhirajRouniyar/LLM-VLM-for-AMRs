#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
import os
import cv2
from threading import Timer
import time
import rospkg

class TrackerNode:
    def __init__(self):
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.depth_topic = rospy.get_param("~depth_topic", "/depth/image")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", None)
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ultralytics_ros')
        self.save_path = rospy.get_param("~save_path", os.path.join(package_path, "Yolo_detected_files"))
        if not os.path.exists(self.save_path):
           os.makedirs(self.save_path)

        # Clear the save path directory
        for file in os.listdir(self.save_path):
            file_path = os.path.join(self.save_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                
            except Exception as e:
                rospy.logerr(f"Failed to delete {file_path}. Reason: {e}")
        self.snapshot_save_path = rospy.get_param("~snapshot_save_path", os.path.join(package_path,"Snapshots"))
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(f"{path}/models/{yolo_model}")
        self.model.fuse()
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_topic,
            Image,
            self.depth_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(
            self.result_image_topic, Image, queue_size=1
        )
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model.endswith(".pt")
        self.latest_depth_image = None
        time.sleep(5)    
        # Timer to shut down the node after 20 seconds
        self.timer = Timer(20, self.shutdown_node)
        self.timer.start()

    def shutdown_node(self):
        if self.latest_rgb_image is not None:
            snapshot_name = os.path.join(self.snapshot_save_path, "snapshot.png")
            cv2.imwrite(snapshot_name, cv2.cvtColor(self.latest_rgb_image, cv2.COLOR_RGB2BGR))
            rospy.loginfo(f"Snapshot saved at {snapshot_name}")
        rospy.signal_shutdown("Shutting down node after 20 seconds")

    def depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")


    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_rgb_image = cv_image

        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)
            self.save_detected_objects(cv_image, results)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="rgb8")
        return result_image_msg

    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg
    
    def save_detected_objects(self, cv_image, results):
        highest_confidences = {}
        bounding_box = results[0].boxes.xyxy
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        label = results[0].names


        # Find the highest confidence score for each class
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            if cls not in highest_confidences or conf > highest_confidences[cls][1]:
                highest_confidences[cls] = (bbox, conf)

        # Save the highest confidence detections
        for cls, (bbox, conf) in highest_confidences.items():
            x_min, y_min, x_max, y_max = map(int, bbox)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            class_name = label[int(cls)]  # Replace with actual class name mapping if available
            image_name = f"{class_name}.png"
            image_path = os.path.join(self.save_path, image_name)
            object_image = cv_image[y_min:y_max, x_min:x_max]
            cv2.imwrite(image_path, cv2.cvtColor(object_image, cv2.COLOR_RGB2BGR))
            depth_value = "N/A"
            if self.latest_depth_image is not None:
                depth_value = float(self.latest_depth_image[y_center, x_center])
               
            else:
                depth_value = "N/A"

            with open(os.path.join(self.save_path, f"{class_name}.txt"), 'w') as f:
                f.write(f"Class: {class_name}\n")
                f.write(f"Confidence: {conf:.2f}\n")
                f.write(f"Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}]\n")
                f.write(f"Center:[{x_center}, {y_center}]\n")
                f.write(f"Depth at Center: {depth_value:.2f}\n")

if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()

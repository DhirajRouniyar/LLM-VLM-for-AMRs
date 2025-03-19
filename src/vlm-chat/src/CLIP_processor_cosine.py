#!/usr/bin/env python

# Common functions for Image Processing with OpenCV
#
# Copyright (c) 2024 Piaggio Fast Forward (PFF), Inc.
# All Rights Reserved. Reproduction, publication,
# or re-transmittal not allowed without written permission of PFF, Inc.

# import the necessary packages

import rospy
import clip 
import torch
import numpy as np
from std_msgs.msg import String
import os
from PIL import Image
import matplotlib.pyplot as plt
import rospkg

class CLIPProcessor:
    def __init__(self):
        rospy.init_node('clip_processor', anonymous = True)

        # Load CLIP model and preprocess function
        self.model, self.preprocess = clip.load("ViT-L/14")
        self.model.cuda().eval()
        
        # Create publishers and subscribers
        self.llm_output_publisher = rospy.Publisher('/llm_output', String, queue_size = 10)
        rospy.Subscriber('/llm_input', String, self.process_input)

    def process_input(self, msg):
        # Extract text input from message
        user_input = str(msg.data).split(', ')
        print("txt_input = ", user_input)
       
        # Create text tokens
        text_tokens = clip.tokenize(user_input).cuda()
        input_image = self.preprocess_images()
        
        # Process with CLIP
        with torch.no_grad():
            image_features = self.model.encode_image(input_image).float() 
            text_features = self.model.encode_text(text_tokens).float()
        
        # Normalize features
        image_features /= image_features.norm(dim = -1, keepdim = True)
        text_features /= text_features.norm(dim = -1, keepdim = True)

        # Calculate Cosine similarity
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        print("similarity:", similarity)
        similarity_round = np.round(similarity, 3)

        # Get the path to the 'ultralytics_ros' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ultralytics_ros')

        # Build the path to the 'Yolo_detected_files' folder inside the package
        yolo_detected_files_path = os.path.join(package_path, 'Yolo_detected_files')

        # List and sort files in the 'Yolo_detected_files' folder
        test_dir = sorted(os.listdir(yolo_detected_files_path))
        print("Test_dir", test_dir)

        for text_index in range(similarity_round.shape[0]):
            max_similarity = np.max(similarity_round[text_index, :])
            
            # Get the corresponding indices for the highest similarity score for the current text
            max_indices = np.where(similarity_round[text_index, :] == max_similarity)[0]
            
            labels = []
            scores = []
            bounding_boxes = []
            centers = []
            depth_bbox = []
            
            # Retrieve the corresponding filename and bounding boxes
            for index in max_indices:
                
                filename_without_extension = test_dir[index*2].rsplit('.', 1)[0]
                labels.append(filename_without_extension)
                score = similarity_round[text_index, index]
                scores.append(score)
                
                # Read bounding box from the corresponding .txt file
                txt_file_path = os.path.join(yolo_detected_files_path, filename_without_extension + ".txt")
                bbox, center, dpth_bbox = self.read_bounding_box(txt_file_path)
                bounding_boxes.append(bbox)
                centers.append(center)
                depth_bbox.append(dpth_bbox)

            # Print the results for the current text
            for label, score, bbox, center, dpth_bbox in zip(labels, scores, bounding_boxes, centers, depth_bbox):
                response_message = f"Text {text_index}: Image {label}, Score {score:.3f}, Bounding Box {bbox}, Center {center}, Depth {dpth_bbox}"
                print("Reponse:", response_message)
                self.llm_output_publisher.publish(response_message)

        # Save Similarity Plot
        self.save_similarity_plot(user_input, test_dir, similarity, yolo_detected_files_path)

    def preprocess_images(self):
        # Extract images from the folder
        original_images = []
        images = []
        # Get the path to the 'ultralytics_ros' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ultralytics_ros')

        # Build the path to the 'Yolo_detected_files' folder inside the package
        yolo_detected_files_path = os.path.join(package_path, 'Yolo_detected_files')
        test_dir = sorted(os.listdir(yolo_detected_files_path))
        for filename in [filename for filename in test_dir if filename.endswith(".png") or filename.endswith(".jpg")]:    
            image = Image.open(os.path.join(yolo_detected_files_path, filename)).convert("RGB")
            original_images.append(image)
            images.append(self.preprocess(image))
        image_input = torch.tensor(np.stack(images)).cuda()
        return image_input

    def read_bounding_box(self, txt_file_path):
        # Read the bounding box from the .txt file and calculate its center
        if not os.path.isfile(txt_file_path):
            return None, None, None

        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
        
        bbox = None
        center = None
        depth = None

        for line in lines:
            print("Line", line)
            if line.startswith('Bounding Box:'):
                bbox_str = line.replace('Bounding Box: ', '').strip('[]\n')
                bbox = list(map(int, bbox_str.split(', ')))
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center = (center_x, center_y)
            
        # for line in lines:
            if line.startswith('Depth at Center:'):
                
                depth_str = line.replace('Depth at Center: ', '').strip()
                depth = float(depth_str)
                print('Depth:', depth)
            
        
        return bbox, center, depth

    def save_similarity_plot(self, user_input, test_dir, similarity, yolo_detected_files_path):
        original_images = []
        for filename in [filename for filename in test_dir if filename.endswith(".png") or filename.endswith(".jpg")]: 
            image = Image.open(os.path.join(yolo_detected_files_path, filename)).convert("RGB")
            original_images.append(image)
        
        count_txt = len(user_input)
        count_img = len(original_images)
        plt.figure(figsize=(15, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        plt.yticks(range(count_txt), user_input, fontsize=18)
        plt.xticks([])
        for i, image in enumerate(original_images):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.3f}", ha="center", va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count_img - 0.5])
        plt.ylim([count_txt + 0.5, -2])

        plt.title("Cosine similarity between text and image features", size=20)
        # Get the path to the 'ultralytics_ros' package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ultralytics_ros')

        # Build the path to the 'Yolo_detected_files' folder inside the package
        # yolo_detected_files_path = os.path.join(package_path, 'Plot')
        # plt.savefig(os.path.join(yolo_detected_files_path, 'similarity_plot.png'))
        plt.show()
        # plt.close()  # Close the plot to free resources

if __name__ == '__main__':
    processor = CLIPProcessor()
    rospy.spin()

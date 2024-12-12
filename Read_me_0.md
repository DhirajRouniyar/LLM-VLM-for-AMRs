

## VLM_nav
This framework synergically exploits the capabilities of pre-trained large language models (LLMs) and a multimodal vision-language model (VLM) to enable humans to interact naturally with autonomous robots through human high level commands. It leverages the LLMs to decode the high-level natural language instructions from humans and abstract them into precise robot actionable commands or queries.


## VLM_nav installation
The following instructions are necessary to set up VLM_nav.

**Install ROS and the navigation planner:**

VLM_nav can work with any ROS-based mobile robot publishing standard ROS topics. The whole framework is implemented using ROS Noetic. To install ROS, follow the instructions at the scripts/x86_installers/install_ros.sh or [ROS Wiki](http://wiki.ros.org/ROS/Installation). You will need to install or ensure that you have the ROS navigation planner and its dependencies installed as stated in steps below.

**Install conda for Linux if not already**
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
  ```
If you get 'conda: command not found' error

In terminal cd to the path where environment.yml is present
Then type/copy the following command

export PATH="$HOME/miniconda3/bin:$PATH"

**Download YOLO models** 
Download the model "yolov8n.pt" from below drive link and save it to "/ultralytics_ros/models" :

https://drive.google.com/drive/folders/16Z-92z2GTl3UR8UJQNuSlQAo2wGKuZDy?usp=sharing

**1. Create a ROS workspace:**
 ```bash
  mkdir -p ~/catkin_ws/src
  cd ~/catkin_ws/src
  ```
 **2. Clone the VLM_nav repository files to your workspace:**
Clone all the packages and files from features/vlm-robot-navigation to the /catkin_ws/src folder

 **3. Make conda env and install VLM_nav dependencies:**
3.1 Source the conda path in bashrc
-Below change dhirajkumarrouniyar to your username.
```bash
nano ~/.bashrc
# Conda initialization
export PATH="/home/dhirajkumarrouniyar/miniconda3/bin:$PATH"
source /home/dhirajkumarrouniyar/miniconda3/etc/profile.d/conda.sh
```
3.2 Make conda Environment
```bash
conda env create -f environment.yml
```

3.3 close and reopen terminal

3.4 Activate the conda venv that you recently created 
```bash
cd ~/catkin_ws/src
conda activate pff_vlm
```
3.5 Required CUDA 10.2 and above versions

If you use CUDA 10.2 then install below pytorch, torchvision and torchaudio versions

- torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
- torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
- torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

3.6 Source the CUDA path in bashrc

FYI - Below, change cuda-10.2 to the version of cuda you work on
```bash
nano ~/.bashrc
# CUDA Setup 
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}

# ROS setup
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
3.7 Set permissions for .sh files
```bash
chmod +x install_vlm-nav_deps.sh
chmod +x planner_dependencies.sh
```
3.8 Install system level dependencies
```bash
bash install_vlm-nav_deps.sh
bash planner_dependencies.sh
```
3.9 Update the package installed files
```bash
sudo apt update
```
**4. Build the workspace:**

4.1 Go to catkin_ws directory and build
```bash
cd ~/catkin_ws
catkin_make
```
Error handling (only when error)

a. If you get error for yaml then deactivate conda env pff_vlm and install following in system level
```bash
pip install pyyaml
```
b. If you get error for empy then remain in the pff_vlm conda env and do following
```bash
sudo apt-get update
pip uninstall em         (do this at system level too if required)
pip install empy==3.3.4
```
4.2 Go to catkin_ws directory
```bash
cd ~/catkin_ws
(In case of error type "rosdep install --from-paths src --ignore-src -r -y" )
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
source devel/setup.bash
```
**5. Install some packages again**
```bash
pip install rospkg
pip install defusedxml
```

## Run VLM_nav Example Demos
### Simulation
Open three terminal windows (T1-T3) in your workspace directory and run the following: 
## Bot will reach to safe distance near to Gazebo objects.
The objects used in the simulation environment have different inflation boxes to prevent the bot from colliding with them. Hence, the body center depth coming from the YOLO box is reduced by the radius of the objects' inflation boxes. This ensures the bot does not collide with objects and maintains a safe distance from them.

**T1:**
```bash
cd ~/catkin_ws/
conda activate pff_vlm
conda install libffi==3.3
conda install numpy
catkin_make
source devel/setup.bash
cd ~/catkin_ws/src/
bash set_permission.sh
roslaunch romr_ros romr_navigation.launch
```


**T2 :**
```bash
cd ~/catkin_ws/
conda activate pff_vlm
source devel/setup.bash
cd ~/catkin_ws/src/
rosrun romr_ros test_goal_sender.py
```

After executing T1 and T2 execute T3, and start interacting with the robot writing human commands in chatGUI interaction platform like Go to Chair, Go to Girl, etc based on the robot field of view (see Gazebo camera RGB output window).


**T3 :**
```bash
cd ~/catkin_ws/
conda activate pff_vlm
source devel/setup.bash
cd ~/catkin_ws/src/
rosrun romr_ros goal_from_image.py
```

After sending the human command the robot will go to the object. 

Before sending the next human command press 'ctrl + c' on T3 to stop the running code. Then run T2 to move robot to home position. Give next command on chatGUI. You will also need to close 'Figure 1 ' that gives similarity score information that appears after running the command.

## YOLO models 
Download the model "yolov8n.pt" from below drive link and save it to "/ultralytics_ros/models" :

https://drive.google.com/drive/folders/16Z-92z2GTl3UR8UJQNuSlQAo2wGKuZDy?usp=sharing

The yolo model used is "yolov8n.pt" which does the intended work here. However, I explored more models as below and user can use it based on their requirements.

1. yolov8n.pt: YOLOv8 Nano Detection Model: This model is designed for object detection, identifying objects within an image and drawing bounding boxes around them. The "n" stands for "nano," indicating a very small model optimized for speed and low computational cost, typically at the expense of accuracy.

2. yolov8m-cls.pt: YOLOv8 Medium Classification Model: This model is trained specifically for image classification tasks. The "m" indicates it's a medium-sized model, which generally balances speed and accuracy.

3. yolov8m-seg.pt: YOLOv8 Medium Segmentation Model: This model is used for image segmentation tasks, where the goal is to classify each pixel in the image. The "m" again signifies a medium-sized model.

4. yolov8n-cls.pt: YOLOv8 Nano Classification Model: This model is for image classification, like the "m-cls" version but smaller (nano). It's optimized for speed and efficiency in classifying images into predefined categories.

5. yolov8n-pose.pt: YOLOv8 Nano Pose Estimation Model: This model is specialized for human pose estimation tasks, where the model predicts keypoints on a human body (like elbows, knees, etc.). The "n" indicates it's a nano model, optimized for speed and efficiency.

## CLIP models

In "vlm-chat/src/CLIP_processor_cosine.py" the clip model used is "ViT-L/14" which does the intended work here. However, based on requirement user can select other models too as described below.

The terms "ViT-B/32", "ViT-B/16", and "ViT-L/14" refer to different variants of the Vision Transformer (ViT) models used in the CLIP (Contrastive Language-Image Pretraining) framework. These models vary in architecture, size, and performance characteristics.

1. ViT-B/32 is suitable if you have limited computational resources or need a faster model that still performs reasonably well.
2. ViT-B/16 offers a good balance between speed and performance, suitable for many general-purpose applications.
3. ViT-L/14 is ideal when high accuracy is paramount and you have sufficient computational resources to handle the increased complexity and memory requirements.









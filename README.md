# LLM-VLM-for-AMRs

Note: Stage 2 code is not published here since paper is under review - Targeted CVPR. Please contact dkrouniyar@wpi.edu 
      In Stage 2 -> LLM is guided by VLM to command bot to navigate in unknown environment based on human commands.
      In Stage 1 -> Only VLM is used for bot to navigate in unknown environment.
    
# Guided_LLM_nav

Generative language models like GPT-2/3 are highly effective at generating text when prompted. However, how these models can be guided by non-text inputs, such as images, remains an open question. I propose a training-free framework called Guided_LLM_nav, which integrates visual controls into the text generation process, enabling multimodal tasks like zero-shot image captioning. Guided_LLM_nav leverages an off-the-shelf language model (e.g., GPT-2) and an image-text matching model VLM to guide text generation. During decoding, a ‘score’ ensures that the generated content aligns with the provided image and maintains coherence with prior context, all without gradient updates, ensuring computational efficiency. This framework allows LLMs to create accurate and contextually relevant descriptions of target image concepts while filtering out irrelevant details, with quality assessed by evaluating how well a receiver model identifies the described object.
There are two sections to run:
# Secton 1: Multimodal sanity check
Desktop run for sanity check of VLM_guide_LLM_robnav pipeline. Here, the data from bot is available from previous runs, refer below steps. Sanity check passes if the bbox coordinates output from Robot indicates the object to which the human command indicates.
Eg: 
Human command : Go near to person with a laptop.
Bot output : I reached to location body center(.............).

# Result_Stage 1

![alt text](https://github.com/DhirajRouniyar/LLM-VLM-for-AMRs/blob/main/Output/demo_stage1.gif)

# Result_Stage 2

![alt text](https://github.com/DhirajRouniyar/LLM-VLM-for-AMRs/blob/main/Output/Stage2_output.png)

# Setup

The following instructions are necessary to set up Guided_LLM_nav.

**Install ROS at system level:**

Guided_LLM_nav can work with any ROS-based mobile robot publishing standard ROS topics. The whole framework is implemented using ROS Noetic. To install ROS, follow the instructions at the scripts/x86_installers/install_ros.sh or [ROS Wiki](http://wiki.ros.org/ROS/Installation). You will need to install conda, create ROS workspace and create a virtual environment as stated in steps below.

**Install conda for Linux if not already**
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
``` 
If you get 'conda: command not found' error

In terminal cd to the path where environment.yml is present Then type/copy the following command
```shell
export PATH="$HOME/miniconda3/bin:$PATH"
``` 
**1. Create a ROS workspace:**
```shell
 mkdir -p ~/catkin_ws/src
 cd ~/catkin_ws/src
```
**2. Clone the Guided_LLM_nav repository files to your workspace: Clone guided_llm_nav package and files from feature/vlm_guide_llm_robnav to the /catkin_ws/src folder**

**3. Make conda env and install Guided_LLM_nav dependencies:** 

3.1 Source the conda path in bashrc -Below change dhirajkumarrouniyar to your username.
```shell
nano ~/.bashrc
# Conda initialization
export PATH="/home/dhirajkumarrouniyar/miniconda3/bin:$PATH"
source /home/dhirajkumarrouniyar/miniconda3/etc/profile.d/conda.sh
```
3.2 Make conda Environment
```shell
conda env create -f environment.yml
```
3.3 Close and reopen terminal

3.4 Activate the conda venv that you recently created
```shell
cd ~/catkin_ws/src
conda activate pff_gllm
```
3.5 Required CUDA 10.2 and above versions

3.6 Source the CUDA path in bashrc

Below, change cuda-10.2 to the version of cuda you work on
```shell
nano ~/.bashrc
# CUDA Setup 
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}

# ROS setup
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
3.7 Update the package installed files
```shell
sudo apt update
```
**4. Build the workspace:**
4.1 Go to catkin_ws directory and build
```shell
cd ~/catkin_ws
catkin_make
```
**Error handling (only when error)**

a. If you get error for yaml then deactivate conda env pff_gllm and install following in system level
```shell
pip install pyyaml
```
b. If you get error for empy then remain in the pff_gllm conda env and do following
```shell
sudo apt-get update
pip uninstall em         (do this at system level too if required)
pip install empy==3.3.4
```
If still the issue persists then run following, make sure to update the name dhirajkumarrouniyar as per your username.
```shell
export PYTHONPATH="/home/dhirajkumarrouniyar/miniconda3/envs/pff_gllm/lib/python3.9/site-packages:$PYTHONPATH"
```
4.2 Go to catkin_ws directory
```shell
cd ~/catkin_ws
(In case of error type "rosdep install --from-paths src --ignore-src -r -y" )
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
source devel/setup.bash
```
**5. Install some packages again**
```shell
pip install rospkg
pip install defusedxml
```
# Download and copy files to the file location:
**Download link:**
https://drive.google.com/drive/folders/1hgYYn3bEuoWVV2wory55pkMtcmbYPxg2?usp=sharing
**File Location:**
/catkin_ws/src/guided_llm_nav/data/from_bot

# Run Guided_LLM_nav Example Demos

**Terminal:**
```shell
cd ~/catkin_ws
conda activate pff_gllm
source source /opt/ros/noetic/setup.bash
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
cd catkin_ws/src/guided_llm_nav/
python3 setup.py install
cd ~/catkin_ws
catkin_make
     
source devel/setup.bash
```
```shell
roslaunch guided_llm_nav guided_llm_nav.launch
```
A RGB Image with multiple objects and a GUI will appear on the screen to interact with the model. 
Type commands inside the GUI like go near to person holding a laptop, go near to blue gita, etc as per the object present in the RGB image.

# Output
In the GUI you can see that Robot is publishing the goal location of the object we referred for it to navigate.
Further, the file location /catkin_ws/src/guided_llm_nav/data/input will be populated with goal.txt file including the goal coordinates for the bot to navigate along, which is later transferred to the bot as goal location.
![Guided_LLM_output](https://github.com/user-attachments/assets/22493387-de7b-41bf-abf6-4fe695311884)



# Section 2: Run on Bot
**1. Get real time data from bot, follow steps below:**
**On local terminal type**
```shell
cd ~/.dev/pmc_software
```
Change bot name beta-5 to your bot name 
```shell
./scripts/smart_copy_to_robot.sh beta-5 
```
**On bot terminal**
```shell
ssh beta-5
sudo systemctl stop pff-bringup
sudo ./bin/navigator
```
Type password where required then enter Ctrl+C
**2. Update run.py code to switch it from sanity check to actual run**
a. Open guided_lm_nav package and edit run.py
- Uncommnet line from 202 to 221
- Change path details on line 204 accordingly
- Change ip address of bot in line 215

**3. Initialize model processing and command bot to navigate to goal position, follow steps below:**
- Follow **Section 1** commands mentioned above.
- Input commands in GUI for objects detected in the bot's FOV like: go near to person with a laptop, go near to blue gita, etc.
- Wait for 30 sec and see the bot going near to desired object.

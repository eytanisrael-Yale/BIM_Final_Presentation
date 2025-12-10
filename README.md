# Hey You: Using LLMs to Address People in a Group Setting

Sivan Almogy, Eytan Israel, Elizabeth Stanish

Building Interactive Machines, Fall 2025

## Background

Robots are increasingly present in social and group settings, from
photo assistants to event helpers. In many cases, the usefulness
and acceptance of a robot in a group social setting depends on
how natural and trustworthy the robot is. Thus, it is essential
to create interactions that feel natural and intuitive with a robot
to build trust between the robot and the subjects interacting with
it. This will allow the robot to be more useful and accepted in
real-world group tasks.

One way to create more natural interactions in group settings is
to personalize the instructions and the way a robot addresses a given
person in a group of people. If the robot is able to uniquely address
each person in the group, as a human would, the interaction might
become more natural and intuitive, leading to greater acceptance
and use of the robot. By focusing on distinguishing details about
a person when addressing them, such as their appearance and
relative position in a group of people, our robot will be able to
call out specific individuals in a larger group and personalize its
interactions.

The goal we set for our robot is to align everyone in a line, 1.5
meters away from the camera. The Robot will talk to the individuals
one by one and give them instructions until it reaches its goal.

## Running the Code

First, we recommend setting up a virtual environment in your ROS directory. Please refer to [this repo]([https://example.com](https://github.com/Yale-BIM/f25-assignments/blob/master/SETUP0_ROSWorkspace.md) to set up your ROS repository. 

Please also ensure you have installed and set up Shutter, the Robot Photographer. Instructions for setting up Shutter can be found in [this repo](https://gitlab.com/interactive-machines/shutter/shutter-ros2/-/blob/real_robot/REAL_ROBOT.md?ref_type=heads).

Next, ensure you have set up a Gemini API key and Amazon Polly account. Instructions to set up your Gemini API key can be found [here](https://ai.google.dev/gemini-api/docs/api-key), and instructions to set up Amazon Polly can be found [here](https://aws.amazon.com/pm/polly/?trk=5eaa77bb-c289-4641-9e1c-98f3f9179bcc&sc_channel=ps&s_kwcid=AL!4422!10!71331071887391!!!!71331599959072!!483456648!1141294741007044&ef_id=fe51437db0b813464f582a446e0cc6d5:G:s).


In order to run the code, you must run several nodes, one after another. Please follow these directions in order:
1) ros2 launch azure_kinect_ros_driver driver.launch.py body_tracking_enabled:=true
2) ros2 launch shutter_bringup shutter_with_face.launch.py
3) ros2 run shutter_lineup lineup_node
4) ros2 run shutter_lineup virtual_camera
   
Then, once these four are running, launch the following as many times as you want. Running it will make a full process run. If you want to run it again, simply click command/control C, and rerun the command

5) ros2 run shutter_lineup control


## Group Contributions
**Sivan Almogy:** Wrote code to create the bounding boxes and cropping from the original images, ran tests to determine in which format the LLMs responses are more accurate, contributed in gathering the intitial LLM testing dataset and prompt engineering, wrote most of the control node, debugged implementation, and helped record the final video and write the final report. 

**Eytan Israel:** Wrote code to generate descriptions based on three test prompts, ran tests to grade the prompts with a judge LLM, determined the best prompt, contributed in gathering the dataset and prompt engineering, wrote most of the shutter_lineup node, debugged implementation, helped conduct user surveys, and helped record the final video and write the final report. 

**Elizabeth Stanish:** Wrote initial plan, conducted literature review, contributed in gathering the dataset and prompt engineering, wrote most of the virtual_camera node, debugged implementation, formulated survey questions and helped conduct user surveys, and helped record the final video and write the final report. 

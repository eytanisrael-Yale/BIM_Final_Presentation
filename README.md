****Hey You: Using LLMs to Address People in a Group Setting****

**Background**

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

**Running the Code**

In order to run the code, you must run several nodes, one after another. Please follow these directions in order:
1) ros2 launch azure_kinect_ros_driver driver.launch.py body_tracking_enabled:=true
2) ros2 launch shutter_bringup shutter_with_face.launch.py
3) ros2 run shutter_lineup lineup_node
4) ros2 run shutter_lineup virtual_camera
   
Then, once these four are running, launch the following as many times as you want. Running it will make a full process run. If you want to run it again, simply click command/control C, and rerun the command

5) ros2 run shutter_lineup control


**Group Contributions**
Sivan Almogy: Wrote code to create the bounding boxes and cropping from the original images, ran tests to determine in which format the LLMs responses are more accurate, and equally contributed in gathering the dataset and prompt engineering.
Eytan Israel: Wrote code to generate descriptions based on the three prompts, and ran tests to grade the prompts with a judge LLM. Computed the mean to determine the best prompt, and equally contributed in gathering the dataset and prompt engineering.
Elizabeth Stanish: Wrote plan and did research to scope out the next steps in using the Kinect sensors, ROS modules, and physical robot. Conducted literature review, wrote introduction, and equally contributed in gathering the dataset and prompt engineering

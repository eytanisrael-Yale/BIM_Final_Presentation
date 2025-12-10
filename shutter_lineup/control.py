import sys
print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHON PATH:", sys.path)

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray, Float64, Int16
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import cv2

import numpy as np

import time

from . import speak

FOOT_IN_METERS = 0.3048

class ControlNode(Node):
    def __init__(self):
        super().__init__('Control_Node')
        self.create_subscription(Float64MultiArray, 'kinect_bounding_box', self.kinect_cb, 10)
        self.create_subscription(Image, "/bounded_image_raw", self.bounded_image_cb, 10)
        self.control_pub = self.create_publisher(Int16, "control", 10)
        # self.target_pub = self.create_publisher(PoseStamped, 'target', 10)

        self.bounded_image = False
        self.bridge = CvBridge()

        self.kinect_data = None
        self.target_idx = -1
        self.num_pictures = 0

        self.get_logger().info("Control started")

        getTarget = Int16()
        getTarget.data = 1
        getImage = Int16()
        getImage.data = 2
        self.messages = [getTarget, getImage]

        speak.polly_autoplay("Hi, I am shutter, a personal camera robot. I am going to help you guys get in a line one meter away for a great picture!")
        self.control_pub.publish(getTarget)
        
        self.get_logger().info(f"PUBLISHED 1")

        self.awaitGoal = False
        self.complete = False
        self.noTargetReceivedCount = 0

        self.targetX = 0
        self.targetZ = 1

        # Publish joints
        self.cmd_pub = self.create_publisher(Float64MultiArray,
                                             "/joint_group_controller/command", 10)
        
        # Publisher to send bounding box to virtual camera
        self.target_bbox_pub = self.create_publisher(Float64MultiArray, '/target_bbox', 10)

        init = Float64MultiArray()
        init.data = [float(0.0), float(-1.0), float(-0.7), float(0.0)]
        self.cmd_pub.publish(init)
        # time.sleep(2)

        self.getNewTarget = True

       


    # Callback with Kinect Data on Current Person Farthest Away
    def kinect_cb(self, msg:Float64MultiArray):
        if not self.complete:
            self.kinect_data = msg.data

            # First, check if goal was met (receives -1 index)
            if self.kinect_data[0] == -1:
                # self.complete = True
                self.noTargetReceivedCount += 1

                if self.noTargetReceivedCount == 5:
                    self.get_logger().info("Goal reached")
                    speak.polly_autoplay("All done, good job!")
                    self.complete = True
                    return
            else:
                self.noTargetReceivedCount = 0


            if self.getNewTarget and self.kinect_data[0] != -1:
                # Save target position
                self.targetX = -(self.kinect_data[0] + self.kinect_data[1]) / 2
                self.targetZ = self.kinect_data[5] 

                # Send bounding box to virtual camera
                self.target_bbox_pub.publish(msg)

        
                
                self.getNewTarget = False
    

    # Callback for Once We've Projected the Image
    def bounded_image_cb(self, msg):
        if not self.complete:
            # Save Image
            # self.bounded_image = True
            self.get_logger().info("Received bounded image")
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(f"frame_{self.num_pictures}.jpg", image)

            # Turn towards target
            self.get_logger().info(f"Target x: {self.targetX}, target z: {self.targetZ}")
            angle = np.arctan2(self.targetX, self.targetZ)
            self.get_logger().info(f"Angle: {angle}")
            # publish 4-joint command (others static)
            cmd = Float64MultiArray()
            cmd.data = [angle, float(-1.0), float(-0.7), float(0.0)]
            self.cmd_pub.publish(cmd)

            self.get_logger().info("Calling Gemini")
            # Await Call Gemini
            speak.polly_autoplay("Hold on while I think!")
            speak.call_gemini(f"frame_{self.num_pictures}.jpg", int(round((self.kinect_data[5] - 1)/.3048)))
            self.num_pictures += 1

            self.getNewTarget = True
            self.awaitGoal = True
        
      
       

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

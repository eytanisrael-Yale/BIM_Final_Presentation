import sys
print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHON PATH:", sys.path)

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray, Float64, Int16
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

import numpy as np

from . import speak

class ControlNode(Node):
    def __init__(self):
        super().__init__('Control_Node')
        self.create_subscription(Float64MultiArray, 'kinect_bounding_box', self.kinect_cb, 10)
        self.create_subscription(Image, "/bounded_image_raw", self.bounded_image_cb, 10)
        self.control_pub = self.create_publisher(Int16, "control", 10)

        self.bounded_image = False
        self.bridge = CvBridge()

        self.kinect_data = None
        self.target_idx = None

        self.get_logger().info("Control started")

        getTarget = Int16()
        getTarget.data = 1
        getImage = Int16()
        getImage.data = 2
        self.messages = [getTarget, getImage]

        self.control_pub.publish(getTarget)
       

    # Callback with Kinect Data on Current Person Farthest Away
    def kinect_cb(self, msg:Float64MultiArray):
        self.kinect_data = msg.data
        if self.target_idx is None or self.kinect_data[0] != self.target_idx:
            self.get_logger().info(f"Switching target to {self.kinect_data[0]}")
            self.target_idx = self.kinect_data[0]
            self.control_pub.publish(self.messages[1])

    # Callback for Once We've Projected the Image
    def bounded_image_cb(self, msg):
        # Save Image
        # self.bounded_image = True
        self.get_logger().info("Received bounded image")
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite("frame.jpg", image)

        # ADD GAZE HERE :)

        self.get_logger().info("Calling Gemini")
        # Await Call Gemini
        speak.call_gemini("frame.jpg", str(self.kinect_data[5] - 1))

        # Wait for target to be within acceptable range
        while rclpy.ok() and abs(self.kinect_data[5]-1) > 0.3048:
            self.get_logger().info(f"Waiting for target to reach goal, current offset: {self.kinect_data[5] - 1:.3f}")
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().info("Publishing Switch target signal (1)")
        # Get new target
        self.control_pub.publish(self.messages[0])

      
       

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

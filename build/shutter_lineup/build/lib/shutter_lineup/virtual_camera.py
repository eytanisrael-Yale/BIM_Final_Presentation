import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2

class VirtualCameraNode(Node):
    
    def __init__(self):
        super().__init__('virtual_camera')

        fx, fy, cx, cy = 976, 976, 1021, 769

        # Intrinsic camera parameters matrix
        self.K = np.zeros((3, 3))
        self.K[0, 0] = fx
        self.K[1, 1] = fy
        self.K[0, 2] = cx
        self.K[1, 2] = cy
        self.K[2, 2] = 1

        self.recent_image = None

        self.create_subscription(Float64MultiArray, 'kinect_bounding_box', self.bound_box_callback, 10)
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, "/rgb/image_raw", self.image_cb, 10
        )

        self.image_pub = self.create_publisher(Image, "/bounded_image_raw", 10)

    def image_cb(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.recent_image = image
        self.get_logger().info(f"Image recieved")


        

    def bound_box_callback(self, bound_msg):
        bound_list = bound_msg.data
        target_idx, x_min, x_max, y_min, y_max, depth = bound_list[0], bound_list[1], bound_list[2], bound_list[3], bound_list[4], bound_list[5]

        if target_idx == -1:
            self.get_logger().info("No target")
            return

        self.get_logger().info(f"X min: {x_min}\nX max: {x_max}\nY min: {y_min}\nY max: {y_max}\nDepth: {depth}\n")

        camera_left, camera_top = self.project_3d_point(x_min, y_max, depth)
        camera_right, camera_bottom = self.project_3d_point(x_max, y_min, depth)
        self.get_logger().info(f"Projected left: {camera_left}, right: {camera_right}, top: {camera_top}, bottom: {camera_bottom}")

        if self.recent_image is not None:
            cv2.rectangle(self.recent_image, (int(camera_left), int(camera_bottom)), (int(camera_right), int(camera_top)), (0, 255, 0), 5)
            img_msg = self.bridge.cv2_to_imgmsg(self.recent_image, encoding="bgr8")
            self.image_pub.publish(img_msg)
            self.get_logger().info(f"PUBLISHED")


    
    def project_3d_point(self, x, y, z):
        KX = self.K @ np.array([x, y, z], dtype=np.float64).reshape((-1, 1))
        x_proj = KX[0, 0] / KX[2, 0]
        y_proj = KX[1, 0] / KX[2, 0]
        return x_proj, y_proj

def main(args=None):
    rclpy.init(args=args)
    node = VirtualCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

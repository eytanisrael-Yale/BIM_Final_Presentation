import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray, Float64, Int16

import numpy as np

FOOT_IN_METERS = 0.3048

class LineupNode(Node):
    def __init__(self):
        super().__init__('lineup_node')
        self.sub = self.create_subscription(
            MarkerArray,
            '/body_tracking_data',
            self.body_tracking_cb,
            10
        )
        # self.create_subscription(Int16, 'control', self.control_cb, 10)
        self.get_logger().info("LineupNode started, listening to /body_tracking_data")
        self.bounding_coords_publisher = self.create_publisher(Float64MultiArray, 'kinect_bounding_box', 10)
        self.switchTarget = False
        self.target_idx = -1

    def control_cb(self, msg: Int16):
        self.get_logger().info(f"ENTERED CONTROL CB")
        if msg.data == 1:
            self.switchTarget = True
        self.get_logger().info(f"self.switchTarget = {self.switchTarget}")
        

    def body_tracking_cb(self, msg: MarkerArray):
        # self.get_logger().info(f"Running body_tracking_cb, self.switchTarget = {self.switchTarget}")
        markers = msg.markers
        if not markers:
            return

      
        JOINTS_PER_PERSON = 32  
        num_people = len(markers) // JOINTS_PER_PERSON
        if num_people == 0:
            return

        person_depths = []
        min_x = []
        max_x = []
        min_y = []
        max_y = []
        for i in range(num_people):
            group = markers[i * JOINTS_PER_PERSON:(i + 1) * JOINTS_PER_PERSON]
            zs = [m.pose.position.z for m in group]
            mean_z = float(np.mean(zs))
            person_depths.append(mean_z)

            min_x_curr = float('inf')
            max_x_curr = -float('inf')
            min_y_curr = float('inf')
            max_y_curr = -float('inf')

            for joint in group:
                if joint.pose.position.x < min_x_curr:
                    min_x_curr = joint.pose.position.x
                if joint.pose.position.x > max_x_curr:
                    max_x_curr = joint.pose.position.x
                if joint.pose.position.y < min_y_curr:
                    min_y_curr = joint.pose.position.y
                if joint.pose.position.y > max_y_curr:
                    max_y_curr = joint.pose.position.y
            
            min_x.append(min_x_curr)
            max_x.append(max_x_curr)
            min_y.append(min_y_curr)
            max_y.append(max_y_curr)

        # find most out-of-line person
        diffs = [abs(z - 1) for z in person_depths]

        
        # Always set target to furthest
        self.target_idx = np.argmax(diffs)


        max_offset = diffs[self.target_idx]

        self.get_logger().info(
            f"curreny idx={self.target_idx}, depth = {person_depths[self.target_idx]}, offset={max_offset:.2f} m"
        )


        if max_offset > FOOT_IN_METERS:
            direction = "forward" if person_depths[self.target_idx] > 1 else "back"
            distance_ft = max_offset / FOOT_IN_METERS
            self.get_logger().info(
                f"Person {self.target_idx} needs to move {direction} "
                f"by about {distance_ft:.1f} ft."
            )

            bounding_coords = Float64MultiArray()
            bounding_coords.data = [self.target_idx, min_x[self.target_idx], max_x[self.target_idx], min_y[self.target_idx], max_y[self.target_idx], person_depths[self.target_idx]]
            self.bounding_coords_publisher.publish(bounding_coords)
        else:
            self.get_logger().info("Everyone is roughly in line (<= 1 ft offset).")
            no_target_response = Float64MultiArray()
            no_target_response.data = [-1, -1, -1, -1, -1, -1]
            self.bounding_coords_publisher.publish(no_target_response)
       

def main(args=None):
    rclpy.init(args=args)
    node = LineupNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

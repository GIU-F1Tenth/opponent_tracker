import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, Point
from image_geometry import PinholeCameraModel
from visualization_msgs.msg import Marker
from nav2_msgs.msg import CostmapUpdate
from builtin_interfaces.msg import Time


class KalmanFilter2D:
    def __init__(self):
        self.dt = 0.1
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.P = np.eye(4) * 500
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 0.1
        self.Q = np.eye(4) * 0.01

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class OpponentTracker(Node):
    def __init__(self):
        super().__init__('opponent_tracker')

        # Declare parameters with default values
        self.declare_parameter('yolo_topic', '/yolo_bboxes')
        self.declare_parameter('depth_topic', '/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('pose_topic', '/opponent_pose_estimate')
        self.declare_parameter('marker_topic', '/opponent_obstacle_marker')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('timer_period', 0.1)
        self.declare_parameter('marker_scale', 0.5)
        self.declare_parameter('marker_lifetime', 1.0)

        # Get parameters
        yolo_topic = self.get_parameter('yolo_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        timer_period = self.get_parameter('timer_period').get_parameter_value().double_value
        marker_scale = self.get_parameter('marker_scale').get_parameter_value().double_value
        marker_lifetime = self.get_parameter('marker_lifetime').get_parameter_value().double_value

        self.frame_id = frame_id
        self.marker_scale = marker_scale
        self.marker_lifetime = marker_lifetime

        self.bridge = CvBridge()
        self.camera_model = PinholeCameraModel()
        self.kf = KalmanFilter2D()
        self.last_detection_time = None

        self.subscription1 = self.create_subscription(
            Detection2DArray, yolo_topic, self.yolo_callback, 10)
        self.subscription2 = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10)
        self.subscription3 = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.publisher = self.create_publisher(
            PoseStamped, pose_topic, 10)
        self.marker_pub = self.create_publisher(
            Marker, marker_topic, 10)

        self.latest_depth_image = None

        self.timer = self.create_timer(timer_period, self.predict_and_publish)

    def camera_info_callback(self, msg):
        self.camera_model.fromCameraInfo(msg)

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def yolo_callback(self, msg):
        if not msg.detections or self.latest_depth_image is None:
            return

        detection = msg.detections[0]
        bbox = detection.bbox
        cx = int(bbox.center.x)
        cy = int(bbox.center.y)

        try:
            depth = self.latest_depth_image[cy, cx]
            if np.isnan(depth) or depth <= 0.1 or depth > 10.0:
                return

            ray = self.camera_model.projectPixelTo3dRay((cx, cy))
            pos = np.array(ray) * depth  # scale the unit ray by depth

            # Update KF
            self.kf.update([pos[0], pos[2]])  # x, z plane
            self.last_detection_time = self.get_clock().now()

        except Exception as e:
            self.get_logger().warn(f"Failed to process detection: {e}")

    def predict_and_publish(self):
        now = self.get_clock().now()
        if self.last_detection_time is None:
            return

        estimate = self.kf.predict()

        pose = PoseStamped()
        pose.header.stamp = now.to_msg()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(estimate[0])
        pose.pose.position.y = 0.0
        pose.pose.position.z = float(estimate[1])
        pose.pose.orientation.w = 1.0

        self.publisher.publish(pose)

        # Publish a Marker for visualizing the obstacle
        marker = Marker()
        marker.header.stamp = now.to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = 'opponent_obstacle'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(estimate[0])
        marker.pose.position.y = 0.0
        marker.pose.position.z = float(estimate[1])
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = int(self.marker_lifetime)
        marker.lifetime.nanosec = int((self.marker_lifetime - int(self.marker_lifetime)) * 1e9)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = OpponentTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

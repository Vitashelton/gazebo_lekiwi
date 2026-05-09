#!/usr/bin/env python3
"""Record synchronized RGB-D + odom + cmd_vel episodes for imitation learning."""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__('episode_recorder')

        self.declare_parameter('output_dir', 'episodes')
        self.declare_parameter('episode_id', 1)
        self.declare_parameter('duration_sec', 10.0)
        self.declare_parameter('rate_hz', 10.0)

        self.output_dir = self.get_parameter('output_dir').value
        self.episode_id = self.get_parameter('episode_id').value
        self.duration_sec = self.get_parameter('duration_sec').value
        self.rate_hz = self.get_parameter('rate_hz').value

        self.bridge = CvBridge()

        self._setup_episode_dir()
        self._init_storage()
        self._setup_synchronizer()

        self.get_logger().info(
            f'EpisodeRecorder ready: {self.episode_path}\n'
            f'  duration={self.duration_sec}s, rate={self.rate_hz}Hz'
        )

    def _setup_episode_dir(self):
        ep_id = int(self.episode_id)
        self.episode_path = Path(self.output_dir) / f'episode_{ep_id:04d}'
        self.rgb_dir = self.episode_path / 'rgb'
        self.depth_dir = self.episode_path / 'depth'
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.episode_path / 'metadata.json'

    def _init_storage(self):
        self.actions = []
        self.poses = []
        self.timestamps = []
        self.frame_idx = 0
        self.start_time = None
        self.sync_latencies = []

    def _setup_synchronizer(self):
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.cmd_sub = Subscriber(self, Twist, '/cmd_vel', qos_profile=qos)
        self.odom_sub = Subscriber(self, Odometry, '/odom', qos_profile=qos)
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw', qos_profile=qos)

        self.sync = ApproximateTimeSynchronizer(
            [self.cmd_sub, self.odom_sub, self.rgb_sub, self.depth_sub],
            queue_size=100,
            slop=0.05,
        )
        self.sync.registerCallback(self._sync_callback)
        self.get_logger().info('Synchronizer registered (slop=0.05s)')

    def _sync_callback(self, cmd_msg, odom_msg, rgb_msg, depth_msg):
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info(f'Recording started at {self.start_time.nanoseconds * 1e-9:.3f}')

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            return

        period = 1.0 / self.rate_hz
        if self.frame_idx > 0 and elapsed - self.timestamps[-1] < period:
            return

        now = time.time()
        stamp_sec = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        latency = abs(now - stamp_sec)
        self.sync_latencies.append(latency)

        # RGB
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            rgb_path = self.rgb_dir / f'{self.frame_idx:06d}.jpg'
            cv2.imwrite(str(rgb_path), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            self.get_logger().error(f'RGB save error frame {self.frame_idx}: {e}')
            return

        # Depth — handle both 32FC1 (meters) and 16UC1 (mm)
        try:
            if depth_msg.encoding == '32FC1':
                depth_float = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                depth_mm = (depth_float * 1000.0).astype(np.uint16)
            elif depth_msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            else:
                self.get_logger().error(f'Unexpected depth encoding: {depth_msg.encoding}')
                return

            depth_path = self.depth_dir / f'{self.frame_idx:06d}.png'
            cv2.imwrite(str(depth_path), depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        except Exception as e:
            self.get_logger().error(f'Depth save error frame {self.frame_idx}: {e}')
            return

        # Action
        self.actions.append([
            float(cmd_msg.linear.x),
            float(cmd_msg.linear.y),
            float(cmd_msg.angular.z),
        ])

        # Pose (x, y, theta)
        q = odom_msg.pose.pose.orientation
        theta = 2.0 * np.arctan2(float(q.z), float(q.w))
        self.poses.append([
            float(odom_msg.pose.pose.position.x),
            float(odom_msg.pose.pose.position.y),
            float(theta),
        ])

        # Timestamp
        self.timestamps.append(elapsed)
        self.frame_idx += 1

        if self.frame_idx % 50 == 0:
            self.get_logger().info(f'Frame {self.frame_idx}: elapsed={elapsed:.1f}s')

    def save_episode(self):
        if self.frame_idx == 0:
            self.get_logger().error('No frames recorded. Check topic availability.')
            return

        end_time = self.timestamps[-1]

        np.save(str(self.episode_path / 'actions.npy'), np.array(self.actions, dtype=np.float32))
        np.save(str(self.episode_path / 'poses.npy'), np.array(self.poses, dtype=np.float32))
        np.save(str(self.episode_path / 'timestamps.npy'), np.array(self.timestamps, dtype=np.float32))

        latencies = np.array(self.sync_latencies)
        metadata = {
            'start_time': self.start_time.nanoseconds * 1e-9,
            'end_time': self.start_time.nanoseconds * 1e-9 + end_time,
            'duration': end_time,
            'num_frames': self.frame_idx,
            'output_dir': str(self.episode_path),
            'mean_sync_latency': float(latencies.mean()),
            'max_sync_latency': float(latencies.max()),
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(
            f'Episode saved: {self.episode_path}\n'
            f'  frames={self.frame_idx}, duration={end_time:.1f}s\n'
            f'  sync latency: mean={metadata["mean_sync_latency"]*1000:.1f}ms, '
            f'max={metadata["max_sync_latency"]*1000:.1f}ms'
        )


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            elapsed = (node.get_clock().now() - node.start_time).nanoseconds * 1e-9 if node.start_time else 0.0
            if node.start_time and elapsed >= node.duration_sec:
                break
    except KeyboardInterrupt:
        node.get_logger().info('Recording interrupted by user')
    finally:
        node.save_episode()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

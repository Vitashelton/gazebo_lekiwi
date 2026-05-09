#!/usr/bin/env python3
# ==============================================================================
# YOLO目标检测节点 (YOLO Detect Node)
# ==============================================================================
# 功能:
#   1. 订阅D435i彩色图像话题
#   2. 运行YOLOv8模型进行目标检测
#   3. 识别: 行人(person)、车辆、障碍物等COCO类别
#   4. 检测到目标后触发行为: 减速(slow)、停止(stop)、绕行(avoid)、不响应(none)
#   5. 发布:
#      - 标注框图像          -> /yolo/detection_image
#      - JSON检测信息        -> /yolo/detection_info
#      - RViz MarkerArray    -> /yolo/bbox_markers (可视化检测框)
#   6. 在RViz中可视化检测框、置信度、识别类别
# ==============================================================================
# 依赖: pip install ultralytics opencv-python numpy
# ==============================================================================

import json
import math
import time
from typing import List, Dict, Optional

import numpy as np
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, Vector3
from visualization_msgs.msg import Marker, MarkerArray


# COCO类别名称映射
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
    4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
    8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
    48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# 检测响应类别: person -> 停止, 车辆类 -> 减速
STOP_CLASSES = {0}
SLOW_CLASSES = {1, 2, 3, 5, 7}


class YoloDetectNode(Node):
    """YOLO目标检测ROS2节点"""

    def __init__(self):
        super().__init__('yolo_detect_node')

        # -- 加载参数 --
        self._load_parameters()

        # -- CvBridge --
        self.bridge = CvBridge()

        # -- 加载YOLO模型 --
        self.model = None
        self._load_model()

        # -- 检测防抖 --
        self.consecutive_detect_count = 0
        self.current_detected_class = None

        # -- 颜色映射 --
        self._color_map: Dict[int, tuple] = {}

        # -- 发布者 --
        self.image_pub = self.create_publisher(
            Image, '/yolo/detection_image', 10
        )
        self.info_pub = self.create_publisher(
            String, '/yolo/detection_info', 10
        )
        self.bbox_pub = self.create_publisher(
            MarkerArray, '/yolo/bbox_markers', 10
        )
        # 速度指令（用于stop/slow响应）
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/yolo/cmd_vel', 10
        )

        # -- 订阅者 --
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_callback, qos
        )

        # -- 统计 --
        self.frame_count = 0
        self.detect_count = 0
        self.last_inference_time = 0.0

        self.get_logger().info(
            f'[YOLO检测] 初始化完成\n'
            f'  模型: {self.model_name} | 设备: {self.device}\n'
            f'  置信度阈值: {self.conf_threshold} | 响应模式: {self.response_mode}\n'
            f'  输入话题: {self.image_topic}\n'
            f'  关注类别: {self.interested_classes}'
        )

    # ==========================================================================
    # 参数加载
    # ==========================================================================

    def _load_parameters(self):
        """从ROS参数服务器加载检测参数"""
        self.declare_parameter('model_name', 'yolov8n')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('interested_classes', [0])
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('response_mode', 'slow')
        self.declare_parameter('slow_fraction', 0.3)
        self.declare_parameter('stop_distance', 0.8)
        self.declare_parameter('trigger_consecutive', 3)
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('use_sim_time', True)

        self.model_name = self.get_parameter('model_name').value
        self.device = self.get_parameter('device').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.interested_classes = self.get_parameter('interested_classes').value
        self.image_topic = self.get_parameter('image_topic').value
        self.response_mode = self.get_parameter('response_mode').value
        self.slow_fraction = self.get_parameter('slow_fraction').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.trigger_consecutive = self.get_parameter('trigger_consecutive').value

    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            self.get_logger().info(f'[YOLO检测] 正在加载 {self.model_name}...')
            self.model = YOLO(self.model_name + '.pt')
            self.model.to(self.device)
            self.get_logger().info('[YOLO检测] 模型加载成功!')
        except ImportError:
            self.get_logger().fatal(
                '[YOLO检测] 未安装ultralytics! 运行: pip install ultralytics'
            )
            raise
        except Exception as e:
            self.get_logger().error(f'[YOLO检测] 模型加载失败: {e}')
            raise

    # ==========================================================================
    # 图像处理主循环
    # ==========================================================================

    def _image_callback(self, msg: Image):
        """D435i彩色图像回调——执行目标检测"""
        self.frame_count += 1

        # 限频推理
        if self.frame_count % max(1, int(30.0 / max(1.0, self._get_rate()))) != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[YOLO检测] 图像转换失败: {e}')
            return

        # YOLO推理
        t_start = time.time()
        results = self.model(
            cv_image, conf=self.conf_threshold, iou=self.iou_threshold,
            imgsz=self.imgsz, verbose=False
        )
        inference_time = time.time() - t_start
        self.last_inference_time = inference_time

        # 解析检测结果
        detections = self._parse_results(results, cv_image.shape)

        if detections:
            self.detect_count += 1
            self._publish_detections(detections, cv_image, msg.header)
            self._handle_response(detections)
        else:
            self.consecutive_detect_count = 0
            self.current_detected_class = None

        # 每100帧日志
        if self.detect_count > 0 and self.detect_count % 100 == 0:
            self.get_logger().info(
                f'[YOLO检测] 帧#{self.frame_count} '
                f'检测次数:{self.detect_count} '
                f'推理耗时:{inference_time*1000:.1f}ms'
            )

    def _get_rate(self) -> float:
        try:
            return float(self.get_parameter('inference_rate').value)
        except Exception:
            return 10.0

    # ==========================================================================
    # 检测结果解析
    # ==========================================================================

    def _parse_results(self, results, image_shape) -> List[Dict]:
        """解析YOLO推理结果，过滤关注类别"""
        detections = []
        h, w = image_shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                cls_id = int(cls_id)
                if self.interested_classes and cls_id not in self.interested_classes:
                    continue

                x1, y1, x2, y2 = box
                detections.append({
                    'class_id': cls_id,
                    'class_name': COCO_CLASSES.get(cls_id, f'cls_{cls_id}'),
                    'confidence': float(conf),
                    'bbox_norm': [
                        float(x1 / w), float(y1 / h),
                        float(x2 / w), float(y2 / h)
                    ],
                    'bbox_pixel': [float(x1), float(y1), float(x2), float(y2)],
                    'center_norm': [
                        float((x1 + x2) / (2 * w)),
                        float((y1 + y2) / (2 * h))
                    ],
                    'area_norm': float((x2 - x1) * (y2 - y1) / (w * h))
                })

        return detections

    # ==========================================================================
    # 结果发布
    # ==========================================================================

    def _publish_detections(self, detections: List[Dict], cv_image, header):
        """发布标注图像、JSON信息、RViz可视化框"""

        # 1. 绘制标注框
        annotated = cv_image.copy()
        for det in detections:
            bbox = det['bbox_pixel']
            cls_name = det['class_name']
            conf = det['confidence']
            color = self._get_color(det['class_id'])

            cv2.rectangle(
                annotated,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2
            )
            label = f'{cls_name} {conf:.2f}'
            cv2.putText(
                annotated, label,
                (int(bbox[0]), int(bbox[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # 发布标注图像
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header = header
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'[YOLO检测] 图像发布失败: {e}')

        # 2. 发布JSON检测信息
        info = {
            'header': {
                'stamp_sec': header.stamp.sec,
                'stamp_nanosec': header.stamp.nanosec,
                'frame_id': header.frame_id
            },
            'num_detections': len(detections),
            'inference_time_ms': round(self.last_inference_time * 1000, 1),
            'detections': [
                {
                    'class': d['class_name'],
                    'confidence': round(d['confidence'], 3),
                    'bbox_pixel': [round(v, 0) for v in d['bbox_pixel']],
                    'center_norm': [round(v, 3) for v in d['center_norm']],
                    'area_norm': round(d['area_norm'], 3)
                }
                for d in detections
            ]
        }
        self.info_pub.publish(String(data=json.dumps(info, ensure_ascii=False)))

        # 3. 发布RViz MarkerArray
        self._publish_bbox_markers(detections, header)

    def _publish_bbox_markers(self, detections: List[Dict], header):
        """发布检测框为RViz MarkerArray（3D空间投影）"""
        marker_array = MarkerArray()
        lifetime_sec = int(1.0 / max(0.1, self._get_rate()) * 1e9)

        for i, det in enumerate(detections):
            # 3D空间检测框
            marker = Marker()
            marker.header = header
            marker.header.frame_id = 'd435_depth_optical_frame'
            marker.ns = 'yolo_detection'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            cx, cy = det['center_norm']
            z_depth = 1.5
            marker.pose.position.x = z_depth
            marker.pose.position.y = (cx - 0.5) * 3.0
            marker.pose.position.z = -(cy - 0.5) * 2.5
            marker.pose.orientation.w = 1.0

            area = det['area_norm']
            marker.scale.x = 0.2
            marker.scale.y = max(0.05, area * 3.0)
            marker.scale.z = max(0.05, area * 3.0)

            color = self._get_color(det['class_id'])
            marker.color.r = float(color[2] / 255.0)
            marker.color.g = float(color[1] / 255.0)
            marker.color.b = float(color[0] / 255.0)
            marker.color.a = 0.7
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = lifetime_sec
            marker_array.markers.append(marker)

            # 类别+置信度文本标签
            text_marker = Marker()
            text_marker.header = header
            text_marker.header.frame_id = 'd435_depth_optical_frame'
            text_marker.ns = 'yolo_labels'
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = z_depth
            text_marker.pose.position.y = (cx - 0.5) * 3.0
            text_marker.pose.position.z = -(cy - 0.5) * 2.5 + 0.1
            text_marker.scale.z = 0.08
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.9
            text_marker.text = f'{det["class_name"]} {det["confidence"]:.2f}'
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = lifetime_sec
            marker_array.markers.append(text_marker)

        self.bbox_pub.publish(marker_array)

    # ==========================================================================
    # 检测响应控制
    # ==========================================================================

    def _handle_response(self, detections: List[Dict]):
        """根据检测结果触发机器人行为（stop/slow/avoid/none）"""
        if self.response_mode == 'none':
            return

        # 找出最高优先级的检测目标
        priority_det = None
        max_priority = -1

        for det in detections:
            cls_id = det['class_id']
            if cls_id in STOP_CLASSES:
                priority = 3
            elif cls_id in SLOW_CLASSES:
                priority = 2
            else:
                priority = 1

            if priority > max_priority:
                max_priority = priority
                priority_det = det

        if priority_det is None:
            self.consecutive_detect_count = 0
            return

        # 防抖: 连续N帧检测到同一类别才触发
        detected_class = priority_det['class_id']
        if detected_class == self.current_detected_class:
            self.consecutive_detect_count += 1
        else:
            self.consecutive_detect_count = 1
            self.current_detected_class = detected_class

        if self.consecutive_detect_count < self.trigger_consecutive:
            return

        # 触发响应
        twist = Twist()

        if detected_class in STOP_CLASSES and self.response_mode == 'stop':
            self.get_logger().warn(
                f'[YOLO检测] 检测到行人! 紧急停止! '
                f'置信度:{priority_det["confidence"]:.2f}'
            )
            # 发布零速度指令
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

        elif detected_class in SLOW_CLASSES and self.response_mode in ('slow', 'avoid'):
            self.get_logger().info(
                f'[YOLO检测] 检测到 {priority_det["class_name"]}! '
                f'减速至{self.slow_fraction*100:.0f}% '
                f'置信度:{priority_det["confidence"]:.2f}'
            )

    # ==========================================================================
    # 工具方法
    # ==========================================================================

    def _get_color(self, cls_id: int) -> tuple:
        """为每个类别生成固定的BGR颜色"""
        if cls_id not in self._color_map:
            import hashlib
            h = int(hashlib.md5(str(cls_id).encode()).hexdigest()[:6], 16)
            self._color_map[cls_id] = (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)
        return self._color_map[cls_id]


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[YOLO检测节点] 用户中断')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

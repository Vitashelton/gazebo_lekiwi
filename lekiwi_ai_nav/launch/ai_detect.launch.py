#!/usr/bin/env python3
# ==============================================================================
# AI目标检测启动 — YOLOv8 目标检测 + RViz可视化
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav ai_detect.launch.py
# 前提: 需要先启动 gazebo_sim.launch.py (提供 /camera/color/image_raw)
# 效果: 订阅D435i彩色图像 -> YOLO推理 -> 发布检测框/标注图像/JSON信息
# 依赖: pip install ultralytics opencv-python torch
# ==============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成AI目标检测Launch描述"""
    pkg_lekiwi_ai_nav = get_package_share_directory('lekiwi_ai_nav')

    # -- 配置文件 --
    yolo_params = os.path.join(pkg_lekiwi_ai_nav, 'config', 'yolo_params.yaml')

    # -- 声明参数 --
    declare_params = DeclareLaunchArgument(
        'yolo_params_file', default_value=yolo_params,
        description='YOLO检测参数文件路径'
    )
    declare_model = DeclareLaunchArgument(
        'model', default_value='yolov8n',
        description='YOLO模型: yolov5n/s/m, yolov8n/s/m/l'
    )
    declare_device = DeclareLaunchArgument(
        'device', default_value='cpu',
        description='推理设备: cpu / cuda / cuda:0'
    )
    declare_conf = DeclareLaunchArgument(
        'conf', default_value='0.5',
        description='置信度阈值 (0.0~1.0)'
    )
    declare_response = DeclareLaunchArgument(
        'response', default_value='slow',
        description='检测响应: none / stop / slow / avoid'
    )

    # -- YOLO目标检测节点 --
    yolo_node = Node(
        package='lekiwi_ai_nav',
        executable='yolo_detect_node',
        name='yolo_detect_node',
        output='screen',
        parameters=[yolo_params, {
            'model_name': LaunchConfiguration('model'),
            'device': LaunchConfiguration('device'),
            'conf_threshold': LaunchConfiguration('conf'),
            'response_mode': LaunchConfiguration('response'),
            'use_sim_time': True,
        }]
    )

    return LaunchDescription([
        declare_params,
        declare_model,
        declare_device,
        declare_conf,
        declare_response,
        LogInfo(msg='=== YOLO AI目标检测 ==='),
        LogInfo(msg='检测结果话题: /yolo/detection_image, /yolo/detection_info'),
        LogInfo(msg='查看检测画面: ros2 run rqt_image_view rqt_image_view /yolo/detection_image'),
        yolo_node,
    ])

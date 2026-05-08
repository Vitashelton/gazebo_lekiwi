#!/usr/bin/env python3
# ==============================================================================
# SLAM建图启动 — 使用 slam_toolbox 在线建图 + 键盘遥控
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav slam_build.launch.py
# 前提: 先启动 gazebo_sim.launch.py
# 效果: 启动SLAM在线建图 + RViz可视化 + 键盘遥控机器人建图
# 地图保存:
#   ros2 run nav2_map_server map_saver_cli -f ~/maps/lab_map
#   或在RViz中通过slam_toolbox面板保存
# ==============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成SLAM建图Launch描述"""
    pkg_lekiwi_ai_nav = get_package_share_directory('lekiwi_ai_nav')

    # -- 配置文件 --
    slam_params = os.path.join(pkg_lekiwi_ai_nav, 'config', 'slam_toolbox_params.yaml')
    rviz_config = os.path.join(pkg_lekiwi_ai_nav, 'config', 'slam.rviz')

    # -- 声明参数 --
    declare_slam_params = DeclareLaunchArgument(
        'slam_params_file', default_value=slam_params,
        description='SLAM Toolbox参数文件路径'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='使用仿真时间'
    )

    # -- 1. slam_toolbox 在线SLAM建图 --
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params, {'use_sim_time': True}],
        remappings=[
            ('/scan', '/scan'),
            ('/map', '/map'),
        ]
    )

    # -- 2. 键盘遥控（用于建图时手动控制机器人移动） --
    teleop_keyboard = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        output='screen',
        prefix='xterm -e',
        remappings=[('/cmd_vel', '/cmd_vel')]
    )

    # -- 3. RViz2 建图可视化 --
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        declare_slam_params,
        declare_use_sim_time,
        LogInfo(msg='=== SLAM建图模式 ==='),
        LogInfo(msg='用键盘控制机器人移动建图: ijkl 前后左右, u/o 左转/右转'),
        LogInfo(msg='建图完成后保存地图:'),
        LogInfo(msg='  ros2 run nav2_map_server map_saver_cli -f ~/maps/lab_map'),
        slam_toolbox_node,
        teleop_keyboard,
        rviz,
    ])

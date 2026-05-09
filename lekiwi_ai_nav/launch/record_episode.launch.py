#!/usr/bin/env python3
# ==============================================================================
# 数据采集启动 — 启动 episode_recorder 记录 RGB-D + odom + cmd_vel 数据
# ==============================================================================
# 用法:
#   1. 先启动仿真: ros2 launch lekiwi_ai_nav gazebo_sim.launch.py
#   2. 再启动采集: ros2 launch lekiwi_ai_nav record_episode.launch.py
#   3. 自定义参数:
#      ros2 launch lekiwi_ai_nav record_episode.launch.py \
#           output_dir:=/tmp/my_episodes \
#           episode_id:=3 \
#           duration_sec:=30.0
#
# 或者在仿真运行后直接运行 recorder 节点:
#   ros2 run lekiwi_ai_nav episode_recorder --ros-args \
#       -p output_dir:=episodes \
#       -p episode_id:=1 \
#       -p duration_sec:=10.0
# ==============================================================================

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_output_dir = DeclareLaunchArgument(
        'output_dir', default_value='episodes',
        description='Output directory for recorded episodes'
    )
    declare_episode_id = DeclareLaunchArgument(
        'episode_id', default_value='1',
        description='Episode ID number'
    )
    declare_duration = DeclareLaunchArgument(
        'duration_sec', default_value='10.0',
        description='Recording duration in seconds'
    )
    declare_rate = DeclareLaunchArgument(
        'rate_hz', default_value='10.0',
        description='Recording rate in Hz'
    )

    recorder = Node(
        package='lekiwi_ai_nav',
        executable='episode_recorder',
        name='episode_recorder',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'episode_id': LaunchConfiguration('episode_id'),
            'duration_sec': LaunchConfiguration('duration_sec'),
            'rate_hz': LaunchConfiguration('rate_hz'),
            'use_sim_time': True,
        }]
    )

    return LaunchDescription([
        declare_output_dir,
        declare_episode_id,
        declare_duration,
        declare_rate,
        LogInfo(msg='=== 数据采集模式 ==='),
        LogInfo(msg='请确保已在另一个终端启动了 gazebo_sim.launch.py'),
        LogInfo(msg='采集话题: /cmd_vel, /odom, /camera/color/image_raw, /camera/aligned_depth_to_color/image_raw'),
        LogInfo(msg='采集频率: 10 Hz'),
        recorder,
    ])

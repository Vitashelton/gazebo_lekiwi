#!/usr/bin/env python3
# ==============================================================================
# Gazebo仿真启动 — 加载实验室环境 + 生成LeKiwi全向轮机器人 + D435i深度相机
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav gazebo_sim.launch.py
# 效果: 启动Gazebo仿真 + 机器人URDF模型 + D435i传感器话题 + /scan虚拟激光
# 注意: 需要 lekiwi_sim 功能包提供机器人URDF模型
# ==============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, TimerAction, LogInfo
)
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    """生成Gazebo仿真Launch描述"""

    # -- 尝试定位 lekiwi_sim 的 URDF 模型和世界文件 --
    try:
        pkg_lekiwi_sim = get_package_share_directory('lekiwi_sim')
        world_path = os.path.join(pkg_lekiwi_sim, 'worlds', 'lab_environment.world')
        urdf_file = os.path.join(pkg_lekiwi_sim, 'urdf', 'LeKiwi.urdf')
    except Exception:
        pkg_lekiwi_sim = None
        world_path = ''
        urdf_file = ''

    pkg_lekiwi_ai_nav = get_package_share_directory('lekiwi_ai_nav')
    fallback_world = os.path.join(pkg_lekiwi_ai_nav, 'config', 'empty_lab.world')

    # 解析实际的 world 路径
    if world_path and os.path.exists(world_path):
        actual_world = world_path
    else:
        actual_world = fallback_world

    # 解析实际的 URDF 路径
    if not urdf_file or not os.path.exists(urdf_file):
        # 用本功能包内置的占位 URDF（用户需替换为真实的 LeKiwi.urdf）
        urdf_file = os.path.join(pkg_lekiwi_ai_nav, 'config', 'lekiwi.urdf')

    # -- 读取 URDF 文件内容（在 launch 生成阶段，作为字符串注入参数）--
    try:
        with open(urdf_file, 'r') as f:
            robot_desc = f.read()
    except FileNotFoundError:
        robot_desc = '<?xml version="1.0"?><robot name="lekiwi"></robot>'
        print(f'[WARN] URDF文件不存在: {urdf_file}，使用空机器人描述!')

    # -- 声明参数 --
    declare_world = DeclareLaunchArgument(
        'world', default_value=actual_world,
        description='Gazebo世界文件路径'
    )
    declare_urdf = DeclareLaunchArgument(
        'urdf', default_value=urdf_file,
        description='机器人URDF模型文件路径'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='使用仿真时间'
    )
    declare_x = DeclareLaunchArgument('x', default_value='0.0', description='初始X')
    declare_y = DeclareLaunchArgument('y', default_value='0.0', description='初始Y')
    declare_z = DeclareLaunchArgument('z', default_value='0.15', description='初始Z')
    declare_yaw = DeclareLaunchArgument('yaw', default_value='0.0', description='初始Yaw')

    # -- 1. 启动Gazebo仿真环境 --
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', LaunchConfiguration('world'),
             '-s', 'libgazebo_ros_factory.so'],
        output='screen',
        name='gazebo'
    )

    # -- 2. robot_state_publisher（URDF内容作为字符串注入）--
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': True,
        }]
    )

    # -- 3. joint_state_publisher --
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # -- 4. 生成机器人实体到Gazebo（需要文件路径，用LaunchConfiguration支持参数覆盖）--
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'lekiwi',
            '-file', LaunchConfiguration('urdf'),
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
            '-unpause'
        ],
        output='screen',
        name='spawn_lekiwi'
    )

    # -- 5. 无需depth_to_laser和static_tf：URDF内置2D LiDAR直接发布/scan，
    #        planar_move插件发布/odom和odom->base_footprint TF

    return LaunchDescription([
        declare_world,
        declare_urdf,
        declare_use_sim_time,
        declare_x, declare_y, declare_z, declare_yaw,
        gazebo,
        TimerAction(period=2.0, actions=[robot_state_publisher]),
        TimerAction(period=2.0, actions=[joint_state_publisher]),
        TimerAction(period=3.0, actions=[spawn_entity]),
    ])

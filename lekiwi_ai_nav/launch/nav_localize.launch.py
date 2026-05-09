#!/usr/bin/env python3
# ==============================================================================
# 定位导航启动 — AMCL定位 + Nav2导航栈 (ROS2 Humble 标准写法)
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav nav_localize.launch.py
# 前提: 先启动 gazebo_sim.launch.py
# ==============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('lekiwi_ai_nav')

    nav2_yaml = os.path.join(pkg, 'config', 'nav2_params.yaml')
    map_yaml  = os.path.join(pkg, 'config', 'lab_map.yaml')
    rviz_cfg  = os.path.join(pkg, 'config', 'nav.rviz')

    use_sim = {'use_sim_time': True}

    # -- 声明参数 --
    declare_map = DeclareLaunchArgument('map', default_value=map_yaml)
    declare_use_sim = DeclareLaunchArgument('use_sim_time', default_value='true')

    # ======================================================================
    # Nav2 各组件节点
    # ======================================================================
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[nav2_yaml, use_sim, {'yaml_filename': map_yaml}])

    amcl = Node(
        package='nav2_amcl', executable='amcl',
        name='amcl', output='screen',
        parameters=[nav2_yaml, use_sim])

    controller = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', output='screen',
        parameters=[nav2_yaml, use_sim])

    planner = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen',
        parameters=[nav2_yaml, use_sim])

    behavior = Node(
        package='nav2_behaviors', executable='behavior_server',
        name='behavior_server', output='screen',
        parameters=[nav2_yaml, use_sim])

    bt = Node(
        package='nav2_bt_navigator', executable='bt_navigator',
        name='bt_navigator', output='screen',
        parameters=[nav2_yaml, use_sim])

    wp = Node(
        package='nav2_waypoint_follower', executable='waypoint_follower',
        name='waypoint_follower', output='screen',
        parameters=[nav2_yaml, use_sim])

    smoother = Node(
        package='nav2_velocity_smoother', executable='velocity_smoother',
        name='velocity_smoother', output='screen',
        parameters=[nav2_yaml, use_sim])

    # ======================================================================
    # lifecycle_manager — 自动配置+激活所有 Nav2 节点
    #   加大 bond_timeout 和延时，确保节点完全就绪后再激活
    # ======================================================================
    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[use_sim, {
            'autostart': True,
            'node_names': [
                'map_server', 'amcl', 'controller_server', 'planner_server',
                'behavior_server', 'bt_navigator', 'waypoint_follower',
                'velocity_smoother',
            ],
            'bond_timeout': 10.0,           # 等 10 秒让各节点就绪
            'attempt_respawn_reconnection': True,
        }])

    rviz = Node(
        package='rviz2', executable='rviz2',
        name='rviz2', output='screen',
        arguments=['-d', rviz_cfg], parameters=[use_sim])

    return LaunchDescription([
        declare_map, declare_use_sim,

        LogInfo(msg='=== Nav2 定位导航 ==='),
        LogInfo(msg=f'地图: {map_yaml}'),
        LogInfo(msg='启动后 lifecycle_manager 会自动激活所有节点'),

        # 节点按顺序启动
        map_server,
        TimerAction(period=0.5, actions=[amcl]),
        TimerAction(period=1.0, actions=[planner]),
        TimerAction(period=1.0, actions=[controller]),
        TimerAction(period=1.5, actions=[behavior]),
        TimerAction(period=1.5, actions=[bt]),
        TimerAction(period=2.0, actions=[wp]),
        TimerAction(period=2.0, actions=[smoother]),
        # lifecycle_manager 在所有节点启动后 8 秒才开始激活 (bond_timeout=10s)
        TimerAction(period=10.0, actions=[lifecycle_mgr]),
        # RViz
        TimerAction(period=12.0, actions=[rviz]),
    ])

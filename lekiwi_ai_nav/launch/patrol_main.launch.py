#!/usr/bin/env python3
# ==============================================================================
# 自动巡逻总控 — Nav2 + YOLO + 巡逻 + 自定义任务 (ROS2 Humble)
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav patrol_main.launch.py
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
    yolo_yaml = os.path.join(pkg, 'config', 'yolo_params.yaml')
    patrol_yaml = os.path.join(pkg, 'config', 'patrol_waypoints.yaml')
    map_yaml = os.path.join(pkg, 'config', 'lab_map.yaml')
    rviz_cfg = os.path.join(pkg, 'config', 'patrol.rviz')

    use_sim = {'use_sim_time': True}

    declare_map = DeclareLaunchArgument('map', default_value=map_yaml)
    declare_use_sim = DeclareLaunchArgument('use_sim_time', default_value='true')

    # -- Nav2 组件 --
    map_server = Node(
        package='nav2_map_server', executable='map_server',
        name='map_server', output='screen',
        parameters=[nav2_yaml, use_sim, {'yaml_filename': map_yaml}])
    amcl = Node(package='nav2_amcl', executable='amcl', name='amcl',
                output='screen', parameters=[nav2_yaml, use_sim])
    controller = Node(package='nav2_controller', executable='controller_server',
                      name='controller_server', output='screen',
                      parameters=[nav2_yaml, use_sim])
    planner = Node(package='nav2_planner', executable='planner_server',
                   name='planner_server', output='screen',
                   parameters=[nav2_yaml, use_sim])
    behavior = Node(package='nav2_behaviors', executable='behavior_server',
                    name='behavior_server', output='screen',
                    parameters=[nav2_yaml, use_sim])
    bt = Node(package='nav2_bt_navigator', executable='bt_navigator',
              name='bt_navigator', output='screen',
              parameters=[nav2_yaml, use_sim])
    wp = Node(package='nav2_waypoint_follower', executable='waypoint_follower',
              name='waypoint_follower', output='screen',
              parameters=[nav2_yaml, use_sim])
    smoother = Node(package='nav2_velocity_smoother', executable='velocity_smoother',
                    name='velocity_smoother', output='screen',
                    parameters=[nav2_yaml, use_sim])

    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[use_sim, {
            'autostart': True,
            'node_names': ['map_server', 'amcl', 'controller_server',
                           'planner_server', 'behavior_server', 'bt_navigator',
                           'waypoint_follower', 'velocity_smoother'],
            'bond_timeout': 10.0,
            'attempt_respawn_reconnection': True,
        }])

    # -- AI + 巡逻组件 --
    yolo_node = Node(package='lekiwi_ai_nav', executable='yolo_detect_node',
                     name='yolo_detect_node', output='screen',
                     parameters=[yolo_yaml, use_sim])
    patrol_node = Node(package='lekiwi_ai_nav', executable='patrol_node',
                       name='patrol_node', output='screen',
                       parameters=[patrol_yaml, use_sim])
    custom_node = Node(package='lekiwi_ai_nav', executable='custom_task_node',
                       name='custom_task_node', output='screen',
                       parameters=[use_sim])
    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                output='screen', arguments=['-d', rviz_cfg], parameters=[use_sim])

    return LaunchDescription([
        declare_map, declare_use_sim,
        LogInfo(msg='=== 巡逻总控 ==='),
        # Nav2 节点
        map_server, TimerAction(period=0.5, actions=[amcl]),
        TimerAction(period=1.0, actions=[planner, controller]),
        TimerAction(period=1.5, actions=[behavior, bt]),
        TimerAction(period=2.0, actions=[wp, smoother]),
        TimerAction(period=10.0, actions=[lifecycle_mgr]),
        # AI + 巡逻
        TimerAction(period=12.0, actions=[yolo_node]),
        TimerAction(period=13.0, actions=[patrol_node]),
        TimerAction(period=14.0, actions=[custom_node]),
        TimerAction(period=15.0, actions=[rviz]),
    ])

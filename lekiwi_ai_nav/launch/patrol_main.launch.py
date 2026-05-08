#!/usr/bin/env python3
# ==============================================================================
# 自动巡逻总控启动 — Nav2导航 + YOLO检测 + 巡逻任务 + 自定义任务
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav patrol_main.launch.py
# 前提: 先启动 gazebo_sim.launch.py
# 效果: 一次性启动 Nav2全组件 + YOLO检测 + 自动巡逻 + 自定义任务 + RViz
# ==============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成自动巡逻总控Launch描述"""
    pkg_lekiwi_ai_nav = get_package_share_directory('lekiwi_ai_nav')

    # -- 配置文件 --
    nav2_yaml = os.path.join(pkg_lekiwi_ai_nav, 'config', 'nav2_params.yaml')
    yolo_yaml = os.path.join(pkg_lekiwi_ai_nav, 'config', 'yolo_params.yaml')
    patrol_yaml = os.path.join(pkg_lekiwi_ai_nav, 'config', 'patrol_waypoints.yaml')
    default_map = os.path.join(pkg_lekiwi_ai_nav, 'config', 'lab_map.yaml')
    rviz_cfg = os.path.join(pkg_lekiwi_ai_nav, 'config', 'patrol.rviz')

    declare_map = DeclareLaunchArgument(
        'map', default_value=default_map, description='地图文件'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true'
    )

    use_sim = {'use_sim_time': True}

    # ======================================================================
    # Nav2 组件（与 nav_localize 相同结构）
    # ======================================================================
    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen',
        parameters=[nav2_yaml, use_sim,
                    {'yaml_filename': default_map}]
    )
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl',
        output='screen', parameters=[nav2_yaml, use_sim]
    )
    controller = Node(
        package='nav2_controller', executable='controller_server',
        name='controller_server', output='screen',
        parameters=[nav2_yaml, use_sim]
    )
    planner = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen',
        parameters=[nav2_yaml, use_sim]
    )
    behavior = Node(
        package='nav2_behaviors', executable='behavior_server',
        name='behavior_server', output='screen',
        parameters=[nav2_yaml, use_sim]
    )
    bt_navigator = Node(
        package='nav2_bt_navigator', executable='bt_navigator',
        name='bt_navigator', output='screen',
        parameters=[nav2_yaml, use_sim]
    )
    waypoint_follower = Node(
        package='nav2_waypoint_follower', executable='waypoint_follower',
        name='waypoint_follower', output='screen',
        parameters=[nav2_yaml, use_sim]
    )
    velocity_smoother = Node(
        package='nav2_velocity_smoother', executable='velocity_smoother',
        name='velocity_smoother', output='screen',
        parameters=[nav2_yaml, use_sim]
    )
    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[use_sim, {
            'autostart': True,
            'node_names': [
                'map_server', 'amcl', 'controller_server', 'planner_server',
                'behavior_server', 'bt_navigator', 'waypoint_follower',
                'velocity_smoother',
            ],
            'bond_timeout': 4.0,
            'attempt_respawn_reconnection': True,
        }]
    )

    # ======================================================================
    # AI + 巡逻组件
    # ======================================================================
    yolo_node = Node(
        package='lekiwi_ai_nav', executable='yolo_detect_node',
        name='yolo_detect_node', output='screen',
        parameters=[yolo_yaml, use_sim]
    )
    patrol_node = Node(
        package='lekiwi_ai_nav', executable='patrol_node',
        name='patrol_node', output='screen',
        parameters=[patrol_yaml, use_sim]
    )
    custom_task_node = Node(
        package='lekiwi_ai_nav', executable='custom_task_node',
        name='custom_task_node', output='screen',
        parameters=[use_sim]
    )
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen', arguments=['-d', rviz_cfg],
        parameters=[use_sim]
    )

    return LaunchDescription([
        declare_map, declare_use_sim_time,

        LogInfo(msg='========================================'),
        LogInfo(msg='  LeKiwi 自动巡逻总控系统'),
        LogInfo(msg='========================================'),
        LogInfo(msg='巡逻命令 (新终端):'),
        LogInfo(msg='  开始: ros2 topic pub /patrol/command std_msgs/String "data: \'start\'" --once'),
        LogInfo(msg='  暂停: ros2 topic pub /patrol/command std_msgs/String "data: \'pause\'" --once'),
        LogInfo(msg='  恢复: ros2 topic pub /patrol/command std_msgs/String "data: \'resume\'" --once'),
        LogInfo(msg='  停止: ros2 topic pub /patrol/command std_msgs/String "data: \'stop\'" --once'),

        # Nav2核心组件（按依赖链延时启动）
        map_server,
        TimerAction(period=1.0, actions=[amcl]),
        TimerAction(period=2.0, actions=[planner, controller]),
        TimerAction(period=3.0, actions=[behavior, bt_navigator]),
        TimerAction(period=4.0, actions=[waypoint_follower, velocity_smoother]),
        TimerAction(period=5.0, actions=[lifecycle_mgr]),
        # AI + 巡逻（等Nav2就绪后启动）
        TimerAction(period=7.0, actions=[yolo_node]),
        TimerAction(period=8.0, actions=[patrol_node]),
        TimerAction(period=9.0, actions=[custom_task_node]),
        TimerAction(period=10.0, actions=[rviz]),
    ])

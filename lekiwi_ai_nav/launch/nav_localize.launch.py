#!/usr/bin/env python3
# ==============================================================================
# 定位导航启动 — AMCL定位 + Nav2导航栈完整启动（直接启动各组件，不依赖nav2_bringup黑盒）
# ==============================================================================
# 用法: ros2 launch lekiwi_ai_nav nav_localize.launch.py
# 前提: 先启动 gazebo_sim.launch.py
# 效果: 启动 map_server + AMCL + controller/planner/behavior + costmaps + RViz
# ==============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, TimerAction, LogInfo, ExecuteProcess
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """直接启动所有Nav2组件"""
    pkg_lekiwi_ai_nav = get_package_share_directory('lekiwi_ai_nav')

    # -- 配置文件 --
    nav2_yaml = os.path.join(pkg_lekiwi_ai_nav, 'config', 'nav2_params.yaml')
    default_map = os.path.join(pkg_lekiwi_ai_nav, 'config', 'lab_map.yaml')
    rviz_config = os.path.join(pkg_lekiwi_ai_nav, 'config', 'nav.rviz')

    # -- 声明参数 --
    declare_map = DeclareLaunchArgument(
        'map', default_value=default_map,
        description='地图文件 (.yaml)'
    )
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true'
    )

    use_sim = {'use_sim_time': True}

    # ======================================================================
    # 1. map_server — 加载预建地图, 发布 /map 话题
    # ======================================================================
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[nav2_yaml, use_sim,
                    {'yaml_filename': default_map}]
    )

    # ======================================================================
    # 2. AMCL — 自适应蒙特卡洛定位, 发布 map->odom 变换
    # ======================================================================
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 3. controller_server — MPPI 全向路径跟踪
    # ======================================================================
    controller = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 4. planner_server — 全局路径规划 (NavFn)
    # ======================================================================
    planner = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 5. behavior_server — 恢复行为 (spin/backup/wait)
    # ======================================================================
    behavior = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 6. bt_navigator — 行为树导航器
    # ======================================================================
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 7. waypoint_follower — 路点跟随器
    # ======================================================================
    waypoint_follower = Node(
        package='nav2_waypoint_follower',
        executable='waypoint_follower',
        name='waypoint_follower',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 8. velocity_smoother — 速度平滑
    # ======================================================================
    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[nav2_yaml, use_sim]
    )

    # ======================================================================
    # 9. lifecycle_manager — 自动激活所有Nav2节点
    # ======================================================================
    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[use_sim, {
            'autostart': True,
            'node_names': [
                'map_server',
                'amcl',
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'waypoint_follower',
                'velocity_smoother',
            ],
            'bond_timeout': 4.0,
            'attempt_respawn_reconnection': True,
        }]
    )

    # ======================================================================
    # 10. RViz2
    # ======================================================================
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[use_sim]
    )

    return LaunchDescription([
        declare_map,
        declare_use_sim_time,

        LogInfo(msg='========================================'),
        LogInfo(msg='  Nav2 定位导航 — 全组件直接启动'),
        LogInfo(msg='========================================'),
        LogInfo(msg=''),
        LogInfo(msg='组件: map_server + AMCL + controller +'),
        LogInfo(msg='  planner + behavior + bt_navigator +'),
        LogInfo(msg='  waypoint_follower + velocity_smoother'),
        LogInfo(msg=''),
        LogInfo(msg='启动后请在RViz中:'),
        LogInfo(msg='  1. 点击顶部 "2D Pose Estimate" 按钮'),
        LogInfo(msg='  2. 在地图原点(0,0)点击并拖箭头指向前方'),
        LogInfo(msg='  3. 此时地图、代价地图、粒子云将全部出现'),
        LogInfo(msg=''),
        LogInfo(msg='测试导航: 点击 "Nav2 Goal" 在地图上选目标点'),

        # 按依赖顺序启动
        map_server,
        TimerAction(period=1.0, actions=[amcl]),
        TimerAction(period=2.0, actions=[planner, controller]),
        TimerAction(period=3.0, actions=[behavior, bt_navigator]),
        TimerAction(period=4.0, actions=[waypoint_follower, velocity_smoother]),
        TimerAction(period=5.0, actions=[lifecycle_mgr]),
        TimerAction(period=6.0, actions=[rviz]),
    ])

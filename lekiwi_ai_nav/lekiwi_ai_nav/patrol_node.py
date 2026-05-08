#!/usr/bin/env python3
# ==============================================================================
# 自动定点巡逻节点 (Patrol Node)
# ==============================================================================
# 功能:
#   1. 从 patrol_waypoints.yaml 加载路点列表 A/B/C/D
#   2. 通过 Nav2 NavigateToPose Action 依次导航到每个路点
#   3. 在每个路点停留指定时间，然后前往下一个
#   4. 支持通过 /patrol/command 话题随时开启/暂停巡逻
#   5. 永久循环或完成 loop_count 次后停止
# ==============================================================================
# 话题:
#   订阅: /patrol/command     (std_msgs/String) "start"/"pause"/"resume"/"stop"
#   发布: /patrol/status      (std_msgs/String) 当前巡逻状态
#         /patrol/current_wp  (std_msgs/String) 当前到达的路点名
#         /custom_task/command (std_msgs/String) 自定义任务指令(JSON)
# ==============================================================================

import json
import math
import os
import yaml

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String


class PatrolNode(Node):
    """自动巡逻节点——通过Nav2 Action Client 依次导航到预设路点"""

    def __init__(self):
        super().__init__('patrol_node')

        # 回调组（支持并发回调）
        self.callback_group = ReentrantCallbackGroup()

        # -- 加载路点参数 --
        self._load_waypoints()

        # -- 状态变量 --
        self.patrol_active = self.auto_start       # 巡逻是否激活
        self.patrol_paused = False                 # 巡逻是否暂停
        self.current_wp_index = 0                  # 当前路点索引
        self.nav_goal_handle = None                # 当前导航目标句柄
        self.loop_count = 0                        # 已完成的循环次数
        self.dwell_timer = None                    # 路点停留定时器

        # -- Nav2 NavigateToPose Action 客户端 --
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self.callback_group
        )

        # -- 发布者 --
        self.status_pub = self.create_publisher(
            String, '/patrol/status', 10
        )
        self.current_wp_pub = self.create_publisher(
            String, '/patrol/current_wp', 10
        )
        # 发布自定义任务指令到 custom_task_node
        self.custom_task_pub = self.create_publisher(
            String, '/custom_task/command', 10
        )

        # -- 订阅者（巡逻控制命令） --
        self.cmd_sub = self.create_subscription(
            String, '/patrol/command', self._cmd_callback, 10,
            callback_group=self.callback_group
        )

        # -- 主循环定时器（4Hz） --
        self.timer = self.create_timer(0.25, self._timer_callback)

        self.get_logger().info(
            f'[巡逻节点] 初始化完成, 路点数: {len(self.waypoints)}, '
            f'自动启动: {self.auto_start}'
        )
        self.get_logger().info('[巡逻节点] 等待Nav2导航栈就绪...')

    # ==========================================================================
    # 参数加载
    # ==========================================================================

    def _load_waypoints(self):
        """从ROS参数加载路点（参数从 patrol_waypoints.yaml 注入）"""
        self.declare_parameter('loop_count', 0)
        self.declare_parameter('default_wait_sec', 3.0)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('waypoints', [
            {'name': 'A', 'x': 1.0, 'y': 0.0, 'yaw': 0.0, 'wait_sec': 2.0,
             'custom_task': 'rotate', 'task_params': {'angle_deg': 360, 'speed_rads': 0.8}},
            {'name': 'B', 'x': 1.0, 'y': 2.0, 'yaw': 1.57, 'wait_sec': 3.0,
             'custom_task': 'signal', 'task_params': {'message': 'Arrived at B'}},
            {'name': 'C', 'x': -1.0, 'y': 2.0, 'yaw': 3.14, 'wait_sec': 2.0,
             'custom_task': 'wait', 'task_params': {'extra_wait_sec': 3.0}},
            {'name': 'D', 'x': -1.0, 'y': 0.0, 'yaw': -1.57, 'wait_sec': 2.0,
             'custom_task': 'none', 'task_params': {}},
        ])

        self.total_loops = self.get_parameter('loop_count').value
        self.default_wait_sec = self.get_parameter('default_wait_sec').value
        self.auto_start = self.get_parameter('auto_start').value
        self.waypoints = self.get_parameter('waypoints').value

    # ==========================================================================
    # 命令回调
    # ==========================================================================

    def _cmd_callback(self, msg: String):
        """处理外部巡逻控制命令"""
        cmd = msg.data.strip().lower()

        if cmd == 'start':
            self.patrol_active = True
            self.patrol_paused = False
            self.current_wp_index = 0
            self.loop_count = 0
            self.nav_goal_handle = None
            self.get_logger().info('[巡逻] 收到START命令 - 开始巡逻')

        elif cmd == 'pause':
            self.patrol_paused = True
            self.get_logger().info('[巡逻] 收到PAUSE命令 - 暂停巡逻')
            self._cancel_navigation()

        elif cmd == 'resume':
            self.patrol_paused = False
            self.get_logger().info('[巡逻] 收到RESUME命令 - 恢复巡逻')

        elif cmd == 'stop':
            self.patrol_active = False
            self.patrol_paused = False
            self.get_logger().info('[巡逻] 收到STOP命令 - 停止巡逻')
            self._cancel_navigation()

        else:
            self.get_logger().warn(f'[巡逻] 未知命令: {cmd} (支持: start/pause/resume/stop)')

        self._publish_status()

    def _cancel_navigation(self):
        """取消当前导航目标"""
        if self.nav_goal_handle is not None:
            self.get_logger().info('[巡逻] 取消当前导航目标')
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None

    # ==========================================================================
    # 主循环状态机
    # ==========================================================================

    def _timer_callback(self):
        """定时器回调 — 巡逻状态机核心"""
        if not self.patrol_active:
            self._publish_status()
            return

        if self.patrol_paused:
            self._publish_status()
            return

        # 检查是否完成所有循环
        if self.total_loops > 0 and self.loop_count >= self.total_loops:
            self.get_logger().info(
                f'[巡逻] 已完成 {self.total_loops} 次循环, 停止巡逻'
            )
            self.patrol_active = False
            self._publish_status()
            return

        # 空闲时发送下一个路点
        if self.nav_goal_handle is None:
            self._navigate_to_waypoint(self.current_wp_index)

    def _navigate_to_waypoint(self, index: int):
        """向Nav2发送导航目标（路点坐标）"""
        if index >= len(self.waypoints):
            return

        wp = self.waypoints[index]

        # 构造 NavigateToPose 目标
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(wp['x'])
        goal.pose.pose.position.y = float(wp['y'])

        # yaw -> quaternion
        yaw = float(wp['yaw'])
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'[巡逻] 导航到路点 {wp["name"]} '
            f'({wp["x"]:.2f}, {wp["y"]:.2f}, yaw={wp["yaw"]:.2f})'
        )

        # 等待Action服务器就绪
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('[巡逻] NavigateToPose Action服务器未就绪!')
            return

        send_goal_future = self.nav_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    # ==========================================================================
    # Action 回调
    # ==========================================================================

    def _goal_response_callback(self, future):
        """导航目标被Nav2接受/拒绝后的回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('[巡逻] 导航目标被Nav2拒绝, 将重试...')
            self.nav_goal_handle = None
            return

        self.get_logger().info('[巡逻] 导航目标已接受, 机器人开始移动...')
        self.nav_goal_handle = goal_handle
        self._publish_status()

        # 监听导航结果
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_msg):
        """导航过程中的反馈（可扩展：发布剩余距离等）"""
        pass

    def _result_callback(self, future):
        """导航到达（或失败）后的回调"""
        self.nav_goal_handle = None

        result = future.result()
        if result.status == 4:  # SUCCEEDED
            wp = self.waypoints[self.current_wp_index]
            self.get_logger().info(f'[巡逻] 成功到达路点 {wp["name"]}!')

            # 发布到达状态
            self.current_wp_pub.publish(String(data=wp['name']))

            # 触发自定义任务
            self._trigger_custom_task(wp)

            # 计算停留时间 = 默认停留 + 路点额外停留
            wait_total = self.default_wait_sec + float(wp.get('wait_sec', 0))
            self.get_logger().info(
                f'[巡逻] 在路点 {wp["name"]} 停留 {wait_total:.1f} 秒...'
            )

            # 停留后前进到下一个路点
            self.dwell_timer = self.create_timer(
                wait_total, self._advance_to_next_waypoint
            )
        else:
            self.get_logger().warn(
                f'[巡逻] 导航失败 (状态码: {result.status}), 将重试当前路点...'
            )

    def _advance_to_next_waypoint(self):
        """前进到下一个路点（处理循环）"""
        # 销毁一次性定时器
        if self.dwell_timer is not None:
            self.destroy_timer(self.dwell_timer)
            self.dwell_timer = None

        self.current_wp_index += 1
        if self.current_wp_index >= len(self.waypoints):
            self.current_wp_index = 0
            self.loop_count += 1
            self.get_logger().info(
                f'[巡逻] 完成第 {self.loop_count} 轮巡逻循环'
            )

    def _trigger_custom_task(self, wp: dict):
        """发布自定义任务命令到 /custom_task/command 话题"""
        task_type = wp.get('custom_task', 'none')
        if task_type == 'none':
            return

        # 构造JSON任务消息，发布到 custom_task_node
        task_msg = json.dumps({
            'task': task_type,
            'params': wp.get('task_params', {}),
            'waypoint': wp['name']
        })

        self.get_logger().info(f'[巡逻] 触发自定义任务: {task_msg}')
        self.custom_task_pub.publish(String(data=task_msg))

    # ==========================================================================
    # 状态发布
    # ==========================================================================

    def _publish_status(self):
        """发布当前巡逻状态"""
        if not self.patrol_active:
            state = 'stopped'
        elif self.patrol_paused:
            state = 'paused'
        elif self.nav_goal_handle is not None:
            wp = self.waypoints[self.current_wp_index]
            state = f'moving_to_{wp["name"]}'
        else:
            state = 'idle'

        self.status_pub.publish(String(data=state))


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[巡逻节点] 用户中断')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

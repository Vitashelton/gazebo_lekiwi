#!/usr/bin/env python3
# ==============================================================================
# 自定义任务执行节点 (Custom Task Node)
# ==============================================================================
# 功能:
#   1. 订阅巡逻节点发布的自定义任务指令（JSON格式）
#   2. 执行任务:
#      - rotate: 原地旋转指定角度
#      - wait:   纯停留等待
#      - signal: 发布提示话题
#      - standby: 定点待命（等待外部指令恢复）
#   3. 通过 /cmd_vel 发布速度指令控制机器人
# ==============================================================================
# 话题:
#   订阅: /custom_task/command  (std_msgs/String) JSON任务指令
#   发布: /custom_task/status   (std_msgs/String) 任务执行状态
#         /cmd_vel             (geometry_msgs/Twist) 速度指令
# ==============================================================================

import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class CustomTaskNode(Node):
    """自定义任务执行节点——到达路点后执行原地旋转/等待/信号等"""

    def __init__(self):
        super().__init__('custom_task_node')

        # -- 发布者 --
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.task_status_pub = self.create_publisher(
            String, '/custom_task/status', 10
        )

        # -- 订阅者（巡逻节点发布的任务指令）--
        self.cmd_sub = self.create_subscription(
            String, '/custom_task/command', self._task_callback, 10
        )

        # -- 状态 --
        self.is_executing = False       # 是否正在执行任务
        self.current_task = None        # 当前任务类型

        self.get_logger().info('[自定义任务节点] 初始化完成, 等待任务指令...')

    # ==========================================================================
    # 任务调度
    # ==========================================================================

    def _task_callback(self, msg: String):
        """收到任务指令后的调度入口"""
        if self.is_executing:
            self.get_logger().warn('[自定义任务] 当前有任务正在执行, 忽略新任务')
            return

        try:
            task_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f'[自定义任务] JSON解析失败: {msg.data}')
            return

        task_type = task_data.get('task', 'none')
        params = task_data.get('params', {})
        waypoint = task_data.get('waypoint', 'unknown')

        self.get_logger().info(
            f'[自定义任务] 执行: {task_type} @ 路点{waypoint} | 参数: {params}'
        )

        self.is_executing = True
        self.current_task = task_type

        if task_type == 'rotate':
            self._execute_rotate(params)
        elif task_type == 'wait':
            self._execute_wait(params)
        elif task_type == 'signal':
            self._execute_signal(params, waypoint)
        elif task_type == 'standby':
            self._execute_standby(params, waypoint)
        else:
            self.get_logger().warn(f'[自定义任务] 未知任务类型: {task_type}')
            self.is_executing = False

    # ==========================================================================
    # 任务实现
    # ==========================================================================

    def _execute_rotate(self, params: dict):
        """原地旋转任务（全向轮绕Z轴旋转）
        params:
          angle_deg:  旋转角度(度), 默认360
          speed_rads: 旋转角速度(rad/s), 默认0.8
        """
        angle_deg = float(params.get('angle_deg', 360))
        speed = float(params.get('speed_rads', 0.8))
        angle_rad = math.radians(angle_deg)
        duration = abs(angle_rad) / abs(speed) if abs(speed) > 0.001 else 0.0

        self.get_logger().info(
            f'[自定义任务] 原地旋转 {angle_deg}deg '
            f'({angle_rad:.2f}rad) @ {speed:.2f}rad/s, 预计{duration:.1f}s'
        )

        # 发布旋转速度指令
        twist = Twist()
        twist.angular.z = speed
        self.cmd_vel_pub.publish(twist)

        self.task_status_pub.publish(
            String(data=f'rotating:{angle_deg}deg')
        )

        # 定时停止
        self.create_timer(duration, self._stop_rotate)

    def _stop_rotate(self):
        """停止旋转并复位"""
        twist = Twist()
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info('[自定义任务] 旋转完成')
        self.is_executing = False
        self.task_status_pub.publish(String(data='idle'))

    def _execute_wait(self, params: dict):
        """纯等待任务
        params:
          extra_wait_sec: 额外等待秒数
        """
        wait_sec = float(params.get('extra_wait_sec', 3.0))
        self.get_logger().info(f'[自定义任务] 原地等待 {wait_sec:.1f}s...')
        self.task_status_pub.publish(String(data=f'waiting:{wait_sec}s'))

        self.create_timer(wait_sec, self._finish_wait)

    def _finish_wait(self):
        """等待完成"""
        self.get_logger().info('[自定义任务] 等待完成')
        self.is_executing = False
        self.task_status_pub.publish(String(data='idle'))

    def _execute_signal(self, params: dict, waypoint: str):
        """发布提示信号任务
        params:
          message: 提示消息内容
        """
        message = params.get('message', f'Arrived at {waypoint}')
        self.get_logger().info(f'[自定义任务] 发布信号: {message}')
        self.task_status_pub.publish(String(data=f'signal:{message}'))
        # 信号任务立即完成
        self.is_executing = False

    def _execute_standby(self, params: dict, waypoint: str):
        """定点待命任务——停在当前点等待外部resume指令
        params:
          timeout_sec: 超时自动恢复(秒), 0=无限等待
        """
        timeout = float(params.get('timeout_sec', 0))
        self.get_logger().info(
            f'[自定义任务] 定点待命 @ {waypoint} (超时: {timeout}s, 0=无限)'
        )
        self.task_status_pub.publish(String(data=f'standby:{waypoint}'))

        # 先停住
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

        if timeout > 0:
            self.create_timer(timeout, self._finish_standby)
        # timeout=0: 无限等待，由外部通过 /custom_task/command 发送 resume 指令恢复

    def _finish_standby(self):
        """待命结束"""
        self.get_logger().info('[自定义任务] 待命结束')
        self.is_executing = False
        self.task_status_pub.publish(String(data='idle'))


def main(args=None):
    rclpy.init(args=args)
    node = CustomTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('[自定义任务节点] 用户中断')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

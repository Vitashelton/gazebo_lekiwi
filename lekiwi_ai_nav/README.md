# LeKiwi AI Navigation — 数据采集与仿真

基于 ROS2 Humble + Gazebo Classic 11 的 LeKiwi 全向轮移动机器人仿真与模仿学习数据采集系统。

## 环境要求

| 组件 | 版本 |
|------|------|
| Ubuntu | 22.04 |
| ROS2 | Humble |
| Gazebo | Classic 11 |
| Python | 3.10+ |

必需的 ROS2 包：`gazebo_ros`, `gazebo_plugins`, `message_filters`, `cv_bridge`

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-message-filters ros-humble-cv-bridge
```

## 编译

```bash
cd ~/claude_ws
colcon build --packages-select lekiwi_ai_nav
source install/setup.bash
```

## 启动仿真

```bash
ros2 launch lekiwi_ai_nav gazebo_sim.launch.py
```

启动后 Gazebo 会加载 10m×10m 实验室环境并生成机器人，仿真发布以下话题：

| 话题 | 类型 | 说明 |
|------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | 全向运动控制 (vx, vy, wz) |
| `/odom` | `nav_msgs/Odometry` | 里程计 |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB 彩色图像 (640×480) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 彩色相机内参 |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 对齐深度图 (640×480) |
| `/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | 深度相机内参 |
| `/scan` | `sensor_msgs/LaserScan` | 2D 激光扫描 |

TF 树：`odom → base_footprint → base_link`

### 速度限制

| 轴 | 范围 |
|----|------|
| vx (前向) | [-0.3, 0.3] m/s |
| vy (侧向) | [-0.3, 0.3] m/s |
| wz (旋转) | [-1.5, 1.5] rad/s |

## 测试全向运动

```bash
# 前进
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# 侧移（横走）
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# 原地旋转
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.8}}"

# 持续发布（10Hz）
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}" -r 10
```

## 录制数据

### 方式一：Launch 文件

```bash
# 先启动仿真（另一个终端）
ros2 launch lekiwi_ai_nav gazebo_sim.launch.py

# 录制 10 秒
ros2 launch lekiwi_ai_nav record_episode.launch.py \
  output_dir:=episodes \
  episode_id:=1 \
  duration_sec:=10.0
```

### 方式二：直接运行节点

```bash
ros2 run lekiwi_ai_nav episode_recorder --ros-args \
  -p output_dir:=episodes \
  -p episode_id:=1 \
  -p duration_sec:=10.0
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `episodes` | 输出目录 |
| `episode_id` | `1` | 片段编号 |
| `duration_sec` | `10.0` | 录制时长（秒） |
| `rate_hz` | `10.0` | 录制频率（Hz） |

### 输出结构

```
episodes/
  episode_0001/
    metadata.json        # 元数据（时间、帧数、同步延迟）
    rgb/
      000000.jpg         # JPEG 质量 95，640×480
      000001.jpg
      000002.jpg
      ...
    depth/
      000000.png         # 16-bit PNG，单位毫米，640×480
      000001.png
      000002.png
      ...
    actions.npy          # (N, 3) float32 — [vx, vy, wz]
    poses.npy            # (N, 3) float32 — [x, y, theta]
    timestamps.npy       # (N,)  float32 — 相对秒数
```

### metadata.json 示例

```json
{
  "start_time": 0.0,
  "end_time": 9.95,
  "duration": 9.95,
  "num_frames": 100,
  "output_dir": "episodes/episode_0001",
  "mean_sync_latency": 0.003,
  "max_sync_latency": 0.012
}
```

### 录制技巧

录制期间在另一个终端操控机器人，采集多样化的动作数据：

```bash
# 前进 + 旋转
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}" -r 10

# 侧移
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10

# 也可以在另一个终端启动键盘遥控
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## 检查数据

```bash
# 查看所有录制片段
python3 install/lekiwi_ai_nav/lib/lekiwi_ai_nav/data_inspector.py episodes/

# 查看指定片段
python3 install/lekiwi_ai_nav/lib/lekiwi_ai_nav/data_inspector.py episodes/ -e 1

# 仅统计，不画图
python3 install/lekiwi_ai_nav/lib/lekiwi_ai_nav/data_inspector.py episodes/ --no-plots
```

### 输出内容

**终端输出：**
- 每个片段的帧数、时长
- vx / vy / wz 的最小值、最大值、均值、标准差
- 位置范围 (x, y, theta)
- 运动占比（非零速度的帧比例）
- 异常标记：时长不足 3 秒、过多零动作、文件缺失

**图像输出（当前目录）：**
- `action_histograms.png` — vx / vy / wz 分布直方图
- `trajectory_plot.png` — 2D 轨迹图（标注起止点）
- `frame_preview.png` — 数帧 RGB 预览，叠加动作标注

## 目录结构

```
lekiwi_ai_nav/
├── config/
│   ├── lekiwi.urdf              # 机器人模型（全向底盘 + RGB-D + LiDAR）
│   ├── empty_lab.world          # Gazebo 仿真环境（10m×10m 实验室）
│   ├── nav2_params.yaml         # Nav2 导航参数
│   ├── yolo_params.yaml         # YOLO 检测参数
│   ├── patrol_waypoints.yaml    # 巡逻路点
│   └── *.rviz                   # RViz 配置文件
├── launch/
│   ├── gazebo_sim.launch.py     # 仿真启动
│   ├── record_episode.launch.py # 数据录制启动
│   ├── slam_build.launch.py     # SLAM 建图
│   ├── nav_localize.launch.py   # AMCL 定位 + Nav2
│   ├── patrol_main.launch.py    # 自动巡逻
│   └── ai_detect.launch.py      # YOLO 目标检测
├── lekiwi_ai_nav/
│   ├── episode_recorder.py      # 数据采集节点
│   ├── patrol_node.py           # 巡逻节点
│   ├── yolo_detect_node.py      # YOLO 检测节点
│   └── custom_task_node.py      # 自定义任务节点
├── scripts/
│   ├── data_inspector.py        # 数据检查工具
│   └── generate_*.py            # 地图生成脚本
├── package.xml
├── setup.py
└── README.md
```

## 常见问题

**Q: 机器人不侧移？**

确认 URDF 中 planar_move 插件配置了 `<velocity_y_axis>1</velocity_y_axis>`。

**Q: 相机话题没有 camera_info？**

Gazebo 的 `libgazebo_ros_camera.so` 会自动发布 `camera_info` 话题，与 `image_raw` 在同一命名空间下。

**Q: 深度图为空？**

深度相机类型必须设为 `type="depth"`（URDF 第 214 行），Gazebo 才能输出深度数据。

**Q: 录制节点提示 "No frames recorded"？**

检查四个话题是否正常发布：
```bash
ros2 topic list | grep -E "(cmd_vel|odom|camera)"
```

**Q: data_inspector 报错 matplotlib 缺失？**

```bash
pip install matplotlib opencv-python numpy
```

**Q: colcon build 报错找不到 message_filters？**

```bash
sudo apt install ros-humble-message-filters
```

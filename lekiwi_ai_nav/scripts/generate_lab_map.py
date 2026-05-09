#!/usr/bin/env python3
# ==============================================================================
# 根据 Gazebo 世界几何自动生成匹配的 PGM 地图
# ==============================================================================
# 匹配 config/empty_lab.world 的布局:
#   - 10m x 10m 室内, 四周围墙 (西墙有1m缺口)
#   - 中心圆柱障碍物
#   - 3个箱子障碍物
#   - 4个路点标记 (不计入障碍)
# ==============================================================================

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# 地图参数
RESOLUTION = 0.05          # 5cm/pixel
WIDTH_M = 10.0             # 10m 宽
HEIGHT_M = 10.0            # 10m 高
ORIGIN_X = -5.0            # 原点在左下角(-5, -5)
ORIGIN_Y = -5.0

WIDTH_PX = int(WIDTH_M / RESOLUTION)    # 200
HEIGHT_PX = int(HEIGHT_M / RESOLUTION)  # 200


def world_to_pixel(wx, wy):
    """世界坐标(m) -> 像素坐标"""
    px = int((wx - ORIGIN_X) / RESOLUTION)
    py = int((wy - ORIGIN_Y) / RESOLUTION)
    return px, HEIGHT_PX - py  # PGM Y轴翻转


def draw_wall(img, x1, y1, x2, y2, thickness_px):
    """画墙壁 (线段, 带厚度)"""
    draw = ImageDraw.Draw(img)
    px1, py1 = world_to_pixel(x1, y1)
    px2, py2 = world_to_pixel(x2, y2)
    # 用粗线近似
    draw.line([px1, py1, px2, py2], fill=0, width=thickness_px)


def draw_box(img, cx, cy, sx, sy):
    """画矩形障碍物 (世界坐标中心 + 半尺寸)"""
    draw = ImageDraw.Draw(img)
    x1, y1 = cx - sx / 2, cy - sy / 2
    x2, y2 = cx + sx / 2, cy + sy / 2
    px1, py1 = world_to_pixel(x1, y2)  # 注意y翻转
    px2, py2 = world_to_pixel(x2, y1)
    draw.rectangle([px1, py1, px2, py2], fill=0)


def draw_circle(img, cx, cy, r):
    """画圆形障碍物"""
    draw = ImageDraw.Draw(img)
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r
    px1, py1 = world_to_pixel(x1, y2)
    px2, py2 = world_to_pixel(x2, y1)
    draw.ellipse([px1, py1, px2, py2], fill=0)


def generate_map(output_dir):
    """生成匹配empty_lab.world的地图"""

    # 创建空白画布 (全部自由空间 = 254 白色)
    img = Image.new('L', (WIDTH_PX, HEIGHT_PX), 254)

    wall_thickness = int(0.2 / RESOLUTION)  # 墙壁厚度0.2m -> px

    # === 围墙 ===
    # 北墙 y=5.0, 从 x=-5 到 x=5
    draw_wall(img, -5.0, 5.0, 5.0, 5.0, wall_thickness)
    # 南墙 y=-5.0
    draw_wall(img, -5.0, -5.0, 5.0, -5.0, wall_thickness)
    # 东墙 x=5.0
    draw_wall(img, 5.0, -5.0, 5.0, 5.0, wall_thickness)
    # 西墙 x=-5.0: 北段 y=0~5, 南段 y=-5~0 (中间1m缺口)
    draw_wall(img, -5.0, 0.0, -5.0, 5.0, wall_thickness)
    draw_wall(img, -5.0, -5.0, -5.0, -0.8, wall_thickness)
    # 缺口在西墙中间 y=-0.3~0.3

    # === 中心圆柱障碍物 (0.5, 0.5, r=0.25) ===
    draw_circle(img, 0.5, 0.5, 0.25)

    # === 箱子障碍物 ===
    draw_box(img, 2.0, -1.5, 0.5, 0.5)   # 箱1
    draw_box(img, -2.0, 1.5, 0.4, 0.6)   # 箱2
    draw_box(img, -1.5, -2.0, 0.6, 0.4)  # 箱3

    # 保存PGM
    pgm_path = os.path.join(output_dir, 'lab_map.pgm')
    img.save(pgm_path)
    print(f'[地图生成] PGM: {pgm_path} ({WIDTH_PX}x{HEIGHT_PX})')

    # 保存YAML
    yaml_path = os.path.join(output_dir, 'lab_map.yaml')
    yaml_content = f'''image: lab_map.pgm
mode: trinary
resolution: {RESOLUTION}
origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
'''
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f'[地图生成] YAML: {yaml_path}')
    print(f'[地图生成] 地图大小: {WIDTH_M}x{HEIGHT_M}m, 分辨率: {RESOLUTION}m/pixel')
    print(f'[地图生成] 障碍物: 围墙 + 1圆柱 + 3箱体 + 西墙走廊缺口')


if __name__ == '__main__':
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'config'
    )
    os.makedirs(config_dir, exist_ok=True)
    generate_map(config_dir)

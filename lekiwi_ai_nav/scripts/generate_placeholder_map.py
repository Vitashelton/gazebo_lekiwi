#!/usr/bin/env python3
# ==============================================================================
# 地图占位生成脚本
# ==============================================================================
# 运行: python3 generate_placeholder_map.py
# 生成一个空白占位地图，用于首次启动导航系统
# 使用SLAM建图后，用真实地图替换此文件
# ==============================================================================

import os

def generate_placeholder_map(output_dir):
    """生成5x5米的空白占位PGM地图和对应的YAML描述文件"""
    import numpy as np
    from PIL import Image

    # 地图参数
    resolution = 0.05       # 5cm/pixel
    width_m = 10.0          # 10m
    height_m = 10.0         # 10m
    origin_x = -5.0
    origin_y = -5.0

    width_px = int(width_m / resolution)
    height_px = int(height_m / resolution)

    # 创建未知(灰色)地图: 值205表示未知区域
    # Nav2/OccupancyGrid: 0=free, 100=occupied, -1/unknown
    # PGM:  0=occupied, 254=free, 205=unknown
    pgm_data = np.full((height_px, width_px), 205, dtype=np.uint8)

    # 在地图中心画一个自由空间区域(白色=254=free)
    cx, cy = width_px // 2, height_px // 2
    free_radius = int(4.0 / resolution)  # 4m半径自由空间
    for y in range(height_px):
        for x in range(width_px):
            if (x - cx)**2 + (y - cy)**2 < free_radius**2:
                pgm_data[y, x] = 254

    # 绘制边界(黑色=0=occupied)
    for y in range(height_px):
        for x in range(width_px):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if abs(dist - free_radius) < 2:
                pgm_data[y, x] = 0

    # 保存PGM
    pgm_path = os.path.join(output_dir, 'lab_map.pgm')
    img = Image.fromarray(pgm_data)
    img.save(pgm_path)
    print(f'[地图生成] 已创建PGM: {pgm_path}')

    # 保存YAML
    yaml_path = os.path.join(output_dir, 'lab_map.yaml')
    yaml_content = f'''image: lab_map.pgm
mode: trinary
resolution: {resolution}
origin: [{origin_x}, {origin_y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
'''
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f'[地图生成] 已创建YAML: {yaml_path}')
    print(f'[地图生成] 提示: 此为占位地图，请使用SLAM建图后替换')


if __name__ == '__main__':
    config_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'config'
    )
    os.makedirs(config_dir, exist_ok=True)
    generate_placeholder_map(config_dir)

import os
from glob import glob
from setuptools import setup

package_name = 'lekiwi_ai_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        # 资源索引
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # 所有launch文件
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # 所有配置文件 (.yaml, .rviz, .pgm, .world)
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz') +
            glob('config/*.pgm') + glob('config/*.world') +
            glob('config/*.urdf')),
        # 脚本
        (os.path.join('lib', package_name),
            glob('scripts/*.py')),
        # README
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zbx',
    maintainer_email='3467934040@qq.com',
    description='LeKiwi AI Navigation Demo Package - Nav2 + SLAM + AMCL + Patrol + YOLO',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'patrol_node = lekiwi_ai_nav.patrol_node:main',
            'yolo_detect_node = lekiwi_ai_nav.yolo_detect_node:main',
            'custom_task_node = lekiwi_ai_nav.custom_task_node:main',
            'episode_recorder = lekiwi_ai_nav.episode_recorder:main',
        ],
    },
)

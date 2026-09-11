#!/usr/bin/env python

"""
让episode1运动到默认位置，方便安装相机、夹爪

使用方法：
python -m lerobot.episode_default_position [--ip=IP地址] [--port=端口号]

参数说明：
--ip: 可选，指定机器人控制器的IP地址，默认为'localhost'
--port: 可选，指定机器人控制器的端口号，默认为12345
"""

import argparse
import logging
import time

from lerobot.robots.enpei_follower.episode_server import EpisodeAPP
from lerobot.utils.utils import init_logging


def move_to_default_position(ip='localhost', port=12345):
    """
    移动机械臂到默认位置
    
    参数:
        ip (str): 机器人控制器的IP地址，默认为'localhost'
        port (int): 机器人控制器的端口号，默认为12345
    """
    try:
        # 初始化控制器
        controller = EpisodeAPP(ip=ip, port=port)
        logging.info(f"Connected to EnpeiRobot controller at {ip}:{port}")
        
        # 移动到默认位置
        # 使能两个机械臂
        fixed_degrees = [180, 90, 83, 210, 110, 210, 90]
        t = controller.angle_mode(fixed_degrees)
        time.sleep(t)
        fixed_degrees = [180, 90, 83, 210, 110-90, 210, 90]
        t = controller.angle_mode(fixed_degrees)
        time.sleep(t)

        # 夹爪运动到10度
        controller.servo_gripper(40)
        logging.info(f"Moving to default position, estimated time: {t:.2f}s")
        
        # 等待移动完成
        time.sleep(t)
        logging.info("Successfully moved to default position")
        
    except Exception as e:
        logging.error(f"Failed to move to default position: {e}")
        raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='移动机械臂到默认位置')
    parser.add_argument('--ip', type=str, default='localhost',
                        help='机器人控制器的IP地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=12345,
                        help='机器人控制器的端口号 (默认: 12345)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化日志
    init_logging()
    
    # 移动到默认位置
    move_to_default_position(ip=args.ip, port=args.port)

if __name__ == "__main__":
    main()
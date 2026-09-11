# 执行中位位置校准
from lerobot.motors.old_motors.servo_controller import FeetechController
import time
import numpy as np
import json
import os
import logging
from dataclasses import dataclass, asdict
from pprint import pformat

import draccus
from lerobot.utils.utils import init_logging

@dataclass
class SetMiddleConfig:
    # 串口设备路径
    port: str = "/dev/ttyACM0"
    # 电机ID范围，格式为 (起始ID, 结束ID)，例如 (1, 6) 表示电机ID为 1,2,3,4,5,6
    motor_range: tuple = (1, 7)

# Helper functions for non-blocking keyboard input
def input_available():
    """Check if keyboard input is available"""
    try:
        import sys, select
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    except:
        # Fallback method if select doesn't work
        return False

def get_key():
    """Get a single keypress"""
    try:
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    except:
        # Fallback if the above doesn't work
        if input_available():
            return sys.stdin.read(1)
        return ''

def reset_middle_positions(controller, motor_range):
    """
    重置中位，将所有电机设置为中位位置
    类似于 lerobot 中的 reset_middle_positions 函数
    
    Args:
        controller: 电机控制器
        motor_range: 电机ID范围
    """
    print("请手动移动机械臂到新的中位位置，然后按回车...")
    input()
    
    # 写入 128 到所有电机的 Torque_Enable
    motor_names = [f"motor{i}" for i in range(motor_range[0], motor_range[1] + 1)]
    controller.motors_bus.write("Torque_Enable", 128, motor_names)
    print("已重置所有电机中位位置")

def read_raw_position(controller, motor_id):
    """
    读取指定电机的原始位置值（不转换为角度）
    
    Args:
        controller: 电机控制器
        motor_id: 电机ID
    
    Returns:
        原始位置值
    """
    motor_name = f"motor{motor_id}"
    position = controller.motors_bus.read("Present_Position", motor_name)
    return position

def toggle_motor_lock(controller, motor_range, motors_locked):
    """切换电机锁定状态（启用/禁用扭矩）
    
    Args:
        controller: 电机控制器
        motor_range: 电机ID范围
        motors_locked: 当前电机锁定状态
        
    Returns:
        新的电机锁定状态
    """
    # 切换状态
    motors_locked = not motors_locked
    
    # 应用扭矩设置
    torque_value = 1 if motors_locked else 0  # 1 = TorqueMode.ENABLED, 0 = TorqueMode.DISABLED
    
    # 为所有电机设置扭矩
    motor_names = [f"motor{i}" for i in range(motor_range[0], motor_range[1] + 1)]
    controller.motors_bus.write("Torque_Enable", torque_value, motor_names)
    
    # 打印状态信息
    status = "已锁定 (扭矩启用)" if motors_locked else "已解锁 (扭矩禁用)"
    print(f"\n电机 {status}")
    
    return motors_locked

@draccus.wrap()
def set_middle(cfg: SetMiddleConfig):
    """主函数，用于设置电机中位位置
    
    Args:
        cfg: 配置参数
    """
    # 初始化日志
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # 初始化控制器
    controller = FeetechController(port=cfg.port, motor_range=cfg.motor_range)
    
    # 连接到电机
    controller.connect()
    
    # 启动时先解锁所有电机
    motor_names = [f"motor{i}" for i in range(cfg.motor_range[0], cfg.motor_range[1] + 1)]
    controller.motors_bus.write("Torque_Enable", 0, motor_names)
    print("已解锁所有电机（扭矩禁用）")
    
    # 电机锁定状态（True = 锁定/扭矩启用，False = 解锁/扭矩禁用）
    motors_locked = False
    
    try:
        print(f"开始持续监控位置。电机范围: {cfg.motor_range[0]}-{cfg.motor_range[1]}。按 Ctrl+C 停止。")
        print("您可以在监控时手动移动机械臂。")
        print("按 'r' 重置中位位置。")
        print("按 'l' 切换电机锁定状态（启用/禁用扭矩）。")
        
        while True:
            # 读取所有电机的当前位置
            start_time = time.perf_counter()
            positions = {}
            for motor_id in range(cfg.motor_range[0], cfg.motor_range[1] + 1):
                positions[motor_id] = read_raw_position(controller, motor_id)
            
            # 显示原始位置
            lock_status = "🔒" if motors_locked else "🔓"
            positions_str = ", ".join([f"电机 {i}: {positions[i]}" for i in range(cfg.motor_range[0], cfg.motor_range[1] + 1)])
            # print(f"原始位置: {positions_str} {lock_status}", end="\r")
            
            # 检查键盘输入
            if input_available():
                key = get_key()
                if key.lower() == 'r':
                    # 重置中位位置
                    reset_middle_positions(controller, cfg.motor_range)
                elif key.lower() == 'l':
                    # 切换电机锁定状态
                    motors_locked = toggle_motor_lock(controller, cfg.motor_range, motors_locked)
            
            # 小延迟，防止过度轮询
            # time.sleep(0.1)

            # 计算FPS
            end_time = time.perf_counter()
            fps = 1 / (end_time - start_time)
            print(f"原始位置: {positions_str} {lock_status} | FPS: {fps:.2f}", end="\r")
            
    except KeyboardInterrupt:
        print("\n用户停止监控。")
    finally:
        # 确保在断开连接前解锁电机（禁用扭矩）
        if motors_locked:
            motor_names = [f"motor{i}" for i in range(cfg.motor_range[0], cfg.motor_range[1] + 1)]
            controller.motors_bus.write("Torque_Enable", 0, motor_names)
            print("电机已解锁（扭矩禁用）")
        
        # 完成后断开连接
        controller.disconnect()
        print("控制器已断开连接。")


if __name__ == "__main__":
    set_middle()
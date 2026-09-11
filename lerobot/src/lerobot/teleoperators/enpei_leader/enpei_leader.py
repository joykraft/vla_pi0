#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.old_motors.servo_controller import FeetechController
from ..teleoperator import Teleoperator
from .config_enpei_leader import EnpeiLeaderConfig
import json
import os

logger = logging.getLogger(__name__)


class EnpeiLeader(Teleoperator):
    """
    [Enpei episode1 Arm](https://enpeicv.com/) designed by EnpeiCV
    """

    config_class = EnpeiLeaderConfig
    name = "enpei_leader"
    
    # Class variable to store the last gripper angle
    last_gripper_angle = 0.0
    # 夹爪角度范围，根据自己夹爪的实际角度范围设置
    max_gripper_angle = 110.0
    min_gripper_angle = 20.0
    # 预先计算映射比例
    gripper_mapping_ratio = (max_gripper_angle - min_gripper_angle) / 90
    
    # 滤波器配置
    filter_config = {
        "record": 0.99,
        "inference": 0.99,
        "teleop": 0.99
    }
    # 当前模式
    current_mode = "record"

    def __init__(self, config: EnpeiLeaderConfig):
        super().__init__(config)
        self.config = config
        
        # 舵机相关常量
        self.SERVO_MOTOR_RANGE = (1, 7)  # 起始ID和结束ID，1-7表示电机ID为1,2,3,4,5,6,7
        # 电机旋转方向 (1=顺时针增大角度, -1=顺时针减小角度)
        self.SERVO_ROTATION_DIRECTION = {
            1: -1,     # 电机1的旋转方向
            2: 1,      # 电机2的旋转方向
            3: 1,      # 电机3的旋转方向
            4: -1,     # 电机4的旋转方向
            5: 1,      # 电机5的旋转方向
            6: -1,     # 电机6的旋转方向
            7: 1,      # 电机7的旋转方向
        }
        # 角度限制
        self.SERVO_DEGREE_RANGE = {
            1: [0, 340],
            2: [0, 180],
            3: [0, 163],
            4: [0, 335],
            5: [0, 220],
            6: [0, 335],
            7: [0, 110],  # 夹爪的角度范围
        }
        # 电机零点参考位置
        self.SERVO_REFERENCE_POSITIONS = {
            1: 180,     # 电机1的参考位置
            2: 90,      # 电机2的参考位置
            3: 83,      # 电机3的参考位置
            4: 210,     # 电机4的参考位置
            5: 110,     # 电机5的参考位置
            6: 210,     # 电机6的参考位置
            7: 90,      # 电机7的参考位置
        }
        # 校准文件 - 使用绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.SERVO_CALIBRATION_FILE = os.path.join(current_dir, "motor_calibration.json")
        
        # 是否使用弧度（默认使用角度）
        self.use_radian = False
        
        self.bus = FeetechController(port=self.config.port, motor_range=self.SERVO_MOTOR_RANGE)
        # 加载校准数据
        self.zero_references = self.load_calibration()
        print(f"self.zero_references: {self.zero_references}")
        
        # 初始化一阶低通滤波器
        self.filtered_angles = [0] * 7
        # 是否是第一次读取角度
        self.is_first_reading = [True] * 7
        # 滤波器参数，alpha值越小滤波效果越强（0-1之间）
        self.filter_alpha = self.filter_config[self.current_mode]
    
    def normalize_to_degrees(self, position, motor_id, zero_references):
        """
        Convert raw position value to degrees in -180 to +180 range
        Based on the apply_calibration function in feetech.py
        """
        if zero_references[motor_id] is not None:
            if position < 0:
                position = 0
            if position > 4095:
                position = 4095
            # Calculate relative position from zero reference
            relative_position = position - zero_references[motor_id]
            if self.SERVO_ROTATION_DIRECTION[motor_id] == -1:
                relative_position = -relative_position
            degrees =round(relative_position * 360.0 / 4096 + self.SERVO_REFERENCE_POSITIONS[motor_id], 2)

            if motor_id in self.SERVO_DEGREE_RANGE:
                min_degree, max_degree = self.SERVO_DEGREE_RANGE[motor_id]
                if degrees < min_degree:
                    degrees = min_degree
                elif degrees > max_degree:
                    degrees = max_degree

            return degrees
        
        # 如果没有零点参考，则返回原始位置
        return None
    
    def load_calibration(self):
        # 加载校准数据
        with open(self.SERVO_CALIBRATION_FILE, 'r') as f:
            calibration_data = json.load(f)
            # 将键转换回整数
            zero_references = {int(k): v for k, v in calibration_data.items()}
            return zero_references
        
    @property
    def action_features(self) -> dict[str, type]:
        motor_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        return {f"{motor}.pos": float for motor in motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = {}
        motor_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        
        for i, motor_name in enumerate(motor_names, 1):
            position = self.bus.read_position(i)
            angle = self.normalize_to_degrees(position, i, self.zero_references)
            
            # 应用一阶低通滤波 (EMA)
            if angle is not None:
                if self.is_first_reading[i-1]:
                    self.filtered_angles[i-1] = angle
                    self.is_first_reading[i-1] = False
                else:
                    self.filtered_angles[i-1] = self.filter_alpha * angle + (1 - self.filter_alpha) * self.filtered_angles[i-1]
                
                # 使用滤波后的角度，保留2位小数角度
                angle_value = round(self.filtered_angles[i-1], 2)
                
                # Update the class variable for gripper angle (always store in degrees)
                if i == 7:  # If this is the gripper motor
                    # 将主臂的夹爪角度映射到副臂的夹爪角度范围
                    mapped_gripper_angle= int(self.min_gripper_angle + self.gripper_mapping_ratio * angle_value)
                    # 如果超出范围，限制在[min_gripper_angle,max_gripper_angle]
                    if mapped_gripper_angle > self.max_gripper_angle:
                        mapped_gripper_angle = self.max_gripper_angle
                    elif mapped_gripper_angle < self.min_gripper_angle:
                        mapped_gripper_angle = self.min_gripper_angle
                    # 如果使用弧度，则映射到[0,1]
                    if self.use_radian:
                        mapped_gripper_angle = (mapped_gripper_angle - self.min_gripper_angle) / (self.max_gripper_angle - self.min_gripper_angle)

                    action[f"{motor_name}.pos"] = mapped_gripper_angle
                    # 传给类变量，以便给副臂使用
                    EnpeiLeader.last_gripper_angle = mapped_gripper_angle
                else:
                    # 如果使用弧度，则将角度转换为弧度
                    if self.use_radian:
                        angle_value = angle_value * np.pi / 180.0
                    action[f"{motor_name}.pos"] = angle_value
            else:
                # 存在none
                return None
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

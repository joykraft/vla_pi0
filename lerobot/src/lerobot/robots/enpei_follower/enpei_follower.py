#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import random
import time
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.enpei_follower.episode_server import EpisodeAPP
from lerobot.teleoperators.enpei_leader.enpei_leader import EnpeiLeader

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_enpei_follower import EnpeiFollowerConfig

logger = logging.getLogger(__name__)


class EnpeiFollower(Robot):
    """
    enpei episode1 follower arm
    """

    config_class = EnpeiFollowerConfig
    name = "enpei_follower"

    # 不同的速度模式，请不要随意更改
    speed_config = {
            "record":  np.array([90, 40, 100, 100, 100, 10]) * 1,
            "inference":  np.array([90, 40, 100, 100, 100, 10]) * 0.7,
            "teleop": np.array([90, 50, 200, 200, 200, 100]) * 1,
        }
    # 默认模式，准备调节速度，待定
    current_mode = "record"
    
    # 滤波器配置
    filter_config = {
        "record": 0.1,
        "inference": 0.1,
        "teleop": 0.1
    }

    def __init__(self, config: EnpeiFollowerConfig):
        super().__init__(config)
        self.config = config
        self.controller = None
        self.episode1_is_connected = False
       
        self.cameras = make_cameras_from_configs(config.cameras)

        
        
        # 是否使用弧度（默认使用角度）
        self.use_radian = False

        self.gear_ratios = [25, 20, 25, 10, 4, 1] 
        
        # 当前观测到的关节角度对应的脉冲数
        self.current_joint_pulses = [0, 0, 0, 0, 0, 0]

        # 上次的观测到的夹爪角度
        self.last_observation_gripper = 0
        
        # 初始化一阶低通滤波器
        self.filtered_angles = [0] * 6
        # 是否是第一次接收角度
        self.is_first_action = [True] * 6
        # 滤波器参数，alpha值越小滤波效果越强（0-1之间）
        self.filter_alpha = self.filter_config[self.current_mode]

        # 初始化一阶低通滤波器（夹爪）
        self.filtered_angles_gripper = 0
        # 是否是第一次接收角度
        self.is_first_action_gripper = True
        # 夹爪滤波参数
        self.filter_alpha_gripper = 0.9

        # 初始化电机最大速度
        self.motors_max_speed = self.speed_config[self.current_mode]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.controller is not None and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.episode1_is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Initialize the controller
        try:
            
            self.controller = EpisodeAPP(ip=self.config.ip_address, port=self.config.port)
            logging.info("Connected to EnpeiRobot controller")
            self.controller.sync_motor_angles()
            # 移到默认位置
            
            fixed_degrees = [180, 90, 83, 210, 110, 210, 90]
            t = self.controller.angle_mode(fixed_degrees)
            time.sleep(t)
            fixed_degrees = [180, 90, 83, 210, 110-90, 210, 90]
            t = self.controller.angle_mode(fixed_degrees)
            time.sleep(t)

            # 夹爪运动到10度
            self.controller.servo_gripper(10)
            self.episode1_is_connected = True

        except Exception as e:
            logging.error(f"Failed to connect to EnpeiRobot controller: {e}")
            raise
        
        for cam in self.cameras.values():
            cam.connect()

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

    def get_observation(self) -> dict[str, Any]:
        if not self.episode1_is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        
        # 尝试读取3次，如果还是失败则返回None
        max_retries = 3
        for retry in range(max_retries):
            obs_list = self.controller.get_motor_angles()
            if obs_list is not None:
                break
            logger.info(f"读取电机角度失败，重试 {retry+1}/{max_retries}")
            time.sleep(0.01)  # 短暂延迟后重试
        
        if obs_list is None:
            logger.info("多次尝试读取电机角度均失败，请检查电机连接")
            return None
        # 更新当前关节角度对应的脉冲数
        for i in range(6):
            if obs_list[i] is not None:
                # 计算当前关节角度对应的脉冲数
                self.current_joint_pulses[i] = int(3200 * self.gear_ratios[i] * obs_list[i] / 360)

        # 使用上次的夹爪角度
        gripper_angle_mapped = self.last_observation_gripper
        # 添加夹爪角度
        obs_list.append(gripper_angle_mapped)

        # 如果使用弧度，则将角度转换为弧度
        if self.use_radian:
            obs_list = [angle * np.pi / 180.0 for angle in obs_list]
        
        motor_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

        obs_dict = {f"{motor}.pos": val for motor, val in zip(motor_names, obs_list)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.episode1_is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract motor positions from action dictionary
        motor_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        goal_pos = {}
        
        # 应用一阶低通滤波 (EMA)
        for i, motor in enumerate(motor_names, 1):
            key = f"{motor}.pos"
            if key in action:
                angle = action[key]
                
                # 如果使用弧度，先转换为角度再应用滤波
                # 这样可以保证滤波效果一致，因为弧度值比角度值小很多
                angle_in_degrees = angle
                if self.use_radian and i <= 6:  # 只对关节1-6进行转换，夹爪单独处理
                    angle_in_degrees = angle * 180.0 / np.pi
                
                # 应用一阶低通滤波 (EMA)在角度值上
                if self.is_first_action[i-1]:
                    self.filtered_angles[i-1] = angle_in_degrees
                    self.is_first_action[i-1] = False
                else:
                    self.filtered_angles[i-1] = self.filter_alpha * angle_in_degrees + (1 - self.filter_alpha) * self.filtered_angles[i-1]
                
                # 使用滤波后的角度，保留2位小数
                filtered_angle_in_degrees = round(self.filtered_angles[i-1], 2)
                
                # 如果使用弧度，将滤波后的角度转回弧度
                # 这是必要的，因为函数最后返回action字典，需要保持与输入格式一致
                filtered_angle = filtered_angle_in_degrees
                if self.use_radian and i <= 6:  # 只对关节1-6进行转换，夹爪单独处理
                    filtered_angle = filtered_angle_in_degrees * np.pi / 180.0
                
                # 更新action中的值为滤波后的值
                action[key] = filtered_angle  # 保持与输入相同的单位（角度或弧度）
                goal_pos[str(i)] = filtered_angle_in_degrees  # 在goal_pos中始终使用角度值，用于后续计算
        
        # 夹爪单独控制
        if "gripper.pos" in action:
            # 读取原有夹爪角度
            angle = action["gripper.pos"]

             # 如果使用弧度，则将[0，1]转回角度
            if self.use_radian:
                angle = angle *(EnpeiLeader.max_gripper_angle - EnpeiLeader.min_gripper_angle) + EnpeiLeader.min_gripper_angle
                
            # 应用一阶低通滤波 (EMA)在角度值上
            if self.is_first_action_gripper:
                self.filtered_angles_gripper = angle
                self.is_first_action_gripper = False
            else:
                self.filtered_angles_gripper = self.filter_alpha_gripper * angle + (1 - self.filter_alpha_gripper) * self.filtered_angles_gripper

            # 使用滤波后的角度，取整数
            filtered_angle = int(self.filtered_angles_gripper)

            # 添加阈值，减少不必要的命令
            if abs(self.last_observation_gripper - filtered_angle) > 0:  
                # 发送夹爪角度到夹爪舵机
                self.controller.servo_gripper(filtered_angle)
                # 需要保存一个变量给get_observation
                self.last_observation_gripper = filtered_angle

            # 更新action中的值为滤波后的值，但保持单位与输入相同
            if self.use_radian:
                # 转到[0-1]
                angle_normalized = (filtered_angle - EnpeiLeader.min_gripper_angle) / (EnpeiLeader.max_gripper_angle - EnpeiLeader.min_gripper_angle)
            else:
                # 滤波后的角度
                angle_normalized = filtered_angle
            action["gripper.pos"] = angle_normalized  


        # 使用controller中的dynamic_move函数来处理关节运动
        self.controller.dynamic_move(
            goal_pos=goal_pos, # 目标角度
            current_joint_pulses=self.current_joint_pulses, # 当前观测角度脉冲
            motors_max_speed=self.motors_max_speed.tolist(), # 电机最大速度
        )

     
        return action

    def disconnect(self):
        if not self.episode1_is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.controller.__del__()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
        # 因为dynamic_move是开环控制，所以在断开连接时，需要同步一次角度
        self.controller.sync_motor_angles()
        self.controller._close_connection()
        logger.info("已经断开机械臂的链接")
        self.episode1_is_connected = False

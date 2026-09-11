#!/usr/bin/env python3
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

import time
import numpy as np
from typing import List, Union, Optional, Dict, Tuple

# Import the Feetech motor bus from lerobot
# 路径问题
import sys
import os

# Handle imports for both module usage and standalone script execution
# Try relative import first (when used as a module)
from lerobot.motors.old_motors.feetech import FeetechMotorsBus
from lerobot.motors.old_motors.configs import FeetechMotorsBusConfig

class FeetechController:
    """
    A simple controller for Feetech motors used in teleoperating master arm.
    
    This controller provides functions to:
    1. Read current positions of motors with configurable IDs
    2. Set positions for motors with configurable IDs
    3. Batch operations for multiple motors
    """
    
    def __init__(self, port: str, motor_model: str = "sts3215", motor_range: Tuple[int, int] = (1, 6), mock: bool = False):
        """
        Initialize the Feetech controller.
        
        Args:
            port: The serial port for the Feetech motors bus
            motor_model: The model of the motors (default: "sts3215")
            motor_range: Tuple of (start_id, end_id) for motor IDs (default: (1, 6))
            mock: Whether to use mock mode for testing (default: False)
        """
        self.motor_range = motor_range
        
        # 创建电机配置字典
        motors_config = {}
        for i in range(motor_range[0], motor_range[1] + 1):
            motors_config[f"motor{i}"] = (i, motor_model)
        
        self.config = FeetechMotorsBusConfig(
            port=port,
            motors=motors_config,
            mock=mock
        )
        self.motors_bus = FeetechMotorsBus(self.config)
        self.is_connected = False
        
        # Add caching for positions to improve performance
        self._cached_positions = None
        self._cache_timestamp = 0
        self._cache_valid_time = 0.05  # Cache valid for 50ms
        
    def connect(self):
        """Connect to the motors bus"""
        if not self.is_connected:
            self.motors_bus.connect()
            self.is_connected = True
            print(f"Connected to Feetech motors on port {self.config.port}")
    
    def disconnect(self):
        """Disconnect from the motors bus"""
        if self.is_connected:
            self.motors_bus.disconnect()
            self.is_connected = False
            print("Disconnected from Feetech motors")
    
    def read_positions(self, motor_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Read current positions of specified motors.
        
        Args:
            motor_ids: List of motor IDs to read from. If None, read from all motors in range.
            
        Returns:
            numpy array of current positions
        """
        if not self.is_connected:
            self.connect()
        
        if motor_ids is None:
            # Default to reading all motors in range
            motor_names = [f"motor{i}" for i in range(self.motor_range[0], self.motor_range[1] + 1)]
        else:
            # Convert IDs to motor names
            motor_names = [f"motor{id}" for id in motor_ids 
                          if self.motor_range[0] <= id <= self.motor_range[1]]
        
        if not motor_names:
            raise ValueError(f"No valid motor IDs provided. Only IDs {self.motor_range[0]} to {self.motor_range[1]} are supported.")
        
        positions = self.motors_bus.read("Present_Position", motor_names)
        return positions
    
    def refresh_position_cache(self):
        """
        Refresh the internal position cache for all motors.
        This allows for faster individual motor position reads.
        """
        motor_names = [f"motor{i}" for i in range(self.motor_range[0], self.motor_range[1] + 1)]
        self._cached_positions = self.motors_bus.read("Present_Position", motor_names)
        self._cache_timestamp = time.perf_counter()
        
    def read_position(self, motor_id: int) -> Optional[float]:
        """
        读取单个舵机的当前位置
        
        Args:
            motor_id: 舵机ID
            
        Returns:
            舵机位置，如果读取失败则返回None
        """
        if not self.is_connected:
            self.connect()
            
        if not (self.motor_range[0] <= motor_id <= self.motor_range[1]):
            print(f"警告: 电机ID {motor_id} 超出范围 {self.motor_range}")
            return None
        
        # Check if we need to refresh the cache
        current_time = time.perf_counter()
        if self._cached_positions is None or (current_time - self._cache_timestamp) > self._cache_valid_time:
            try:
                self.refresh_position_cache()
            except Exception as e:
                print(f"刷新位置缓存失败: {e}")
                return None
                
        # Get position from cache
        try:
            index = motor_id - self.motor_range[0]
            if 0 <= index < len(self._cached_positions):
                return self._cached_positions[index]
            else:
                print(f"警告: 电机ID {motor_id} 的缓存索引超出范围")
                return None
        except Exception as e:
            print(f"从缓存获取电机 {motor_id} 位置失败: {e}")
            return None
    
    def batch_read_positions(self) -> Dict[int, float]:
        """
        使用同步读取（SyncRead）一次性读取所有舵机角度，返回字典。
        同时更新内部缓存。
        
        Returns:
            Dictionary mapping motor ID to position
        """
        # motor_names: [motor1, motor2, ...]
        motor_names = [f"motor{i}" for i in range(self.motor_range[0], self.motor_range[1] + 1)]
        # 使用 FeetechMotorsBus 的 read("Present_Position", motor_names) 支持同步读取
        positions = self.motors_bus.read("Present_Position", motor_names)
        
        # Update cache
        self._cached_positions = positions
        self._cache_timestamp = time.perf_counter()
        
        # 返回 {id: pos, ...}
        return {i: positions[i - self.motor_range[0]] for i in range(self.motor_range[0], self.motor_range[1] + 1)}
    
    def set_positions(self, positions: Union[List[float], np.ndarray], motor_ids: Optional[List[int]] = None):
        """
        Set positions for specified motors.
        
        Args:
            positions: List or array of positions to set
            motor_ids: List of motor IDs to set. If None, set all motors in range.
        """
        if not self.is_connected:
            self.connect()
        
        if motor_ids is None:
            # Default to setting all motors in range
            motor_names = [f"motor{i}" for i in range(self.motor_range[0], self.motor_range[1] + 1)]
            
            # Make sure we have enough positions
            if len(positions) < (self.motor_range[1] - self.motor_range[0] + 1):
                positions = [positions[0]] * (self.motor_range[1] - self.motor_range[0] + 1)
        else:
            # Convert IDs to motor names and ensure positions match
            motor_names = []
            pos_values = []
            
            for i, id in enumerate(motor_ids):
                if self.motor_range[0] <= id <= self.motor_range[1]:
                    motor_names.append(f"motor{id}")
                    pos_values.append(positions[i] if i < len(positions) else positions[0])
            
            positions = pos_values
        
        if not motor_names:
            raise ValueError(f"No valid motor IDs provided. Only IDs {self.motor_range[0]} to {self.motor_range[1]} are supported.")
        
        # Convert float positions to integers for Feetech motors
        positions = [int(pos) for pos in positions]
        
        self.motors_bus.write("Goal_Position", positions, motor_names)
    
    def batch_set_positions(self, position_dict: Dict[int, float]):
        """
        Set positions for multiple motors using a dictionary.
        
        Args:
            position_dict: Dictionary mapping motor ID to desired position
        """
        motor_ids = []
        positions = []
        
        for motor_id, position in position_dict.items():
            if self.motor_range[0] <= motor_id <= self.motor_range[1]:
                motor_ids.append(motor_id)
                positions.append(int(position))  # Convert to int for Feetech motors
        
        if motor_ids:
            self.set_positions(positions, motor_ids)
    
    def __del__(self):
        """Ensure motors bus is disconnected when object is deleted"""
        self.disconnect()

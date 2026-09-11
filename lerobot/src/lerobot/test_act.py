#!/usr/bin/env python3

"""
Test a trained policy on a robot directly.

Example:
python -m lerobot.test_act \
    --robot.type=enpei_follower \
    --robot.id=enpei_follower \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --policy_path=weights_0720/act/pretrained_model \
    --fps=30 \
    --time_s=120 \
    --single_task="Grab the fruit to the box"
"""

import time
import torch
from pathlib import Path
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import RobotConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# Make sure all robot types are imported so they're registered as choices
from lerobot.robots import hope_jr, koch_follower, enpei_follower, so100_follower, so101_follower, lekiwi


@dataclass
class TestConfig:
    # Robot configuration
    robot: RobotConfig
    # Path to the pretrained policy
    policy_path: str
    # FPS for policy evaluation
    fps: int = 30
    # Time in seconds to run the policy
    time_s: int = 120
    # Display data using rerun
    display_data: bool = False
    # Task description for the policy
    single_task: str = None
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy"""
        return ["policy_path"]


def prepare_observation_for_policy(observation, policy, device):
    """准备观测数据用于策略输入"""
    batch = {}
    
    # 处理图像
    for feature_name, feature in policy.config.input_features.items():
        if feature_name.startswith("observation.image"):
            # 提取相机名称
            if ".images." in feature_name:
                cam_name = feature_name.split("observation.images.")[-1]
            else:
                cam_name = feature_name.split("observation.image.")[-1]
                
            # 尝试不同的相机名称可能性
            possible_names = [cam_name, f"image_{cam_name}", "handeye", "fixed"]
            found_key = None
            
            for name in possible_names:
                if name in observation:
                    found_key = name
                    break
            
            if found_key:
                # 处理图像数据
                img = observation[found_key]
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                
                if img.dim() == 3:  # HWC格式
                    img = img.permute(2, 0, 1)  # 转换为CHW格式
                
                img = img.unsqueeze(0).to(device)
                batch[feature_name] = img
            else:
                # 使用任何可用的相机作为备选
                for obs_key in observation:
                    if isinstance(observation[obs_key], torch.Tensor) and observation[obs_key].dim() >= 3:
                        img = observation[obs_key]
                        if img.dtype == torch.uint8:
                            img = img.float() / 255.0
                        if img.dim() == 3:
                            img = img.permute(2, 0, 1)
                        img = img.unsqueeze(0).to(device)
                        batch[feature_name] = img
                        break
        
        # 处理机器人状态
        elif feature_name == "observation.state":
            state_values = []
            for motor_name in observation:
                if ".pos" in motor_name:
                    state_values.append(observation[motor_name])
            
            if state_values:
                state_tensor = torch.tensor(state_values, dtype=torch.float32)
                state_tensor = state_tensor.unsqueeze(0).to(device)
                batch[feature_name] = state_tensor
    
    return batch


@parser.wrap()
def test_act_policy(cfg: TestConfig):
    init_logging()
    
    # 连接机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # 初始化可视化（如果需要）
    if cfg.display_data:
        _init_rerun(session_name="act_policy_test")
    
    try:
        # 加载策略
        print(f"加载策略模型: {cfg.policy_path}")
        
        policy_config = PreTrainedConfig.from_pretrained(cfg.policy_path)
        policy = ACTPolicy.from_pretrained(cfg.policy_path)
        device = get_safe_torch_device(policy_config.device)
        policy.to(device)
        policy.eval()
        
        print(f"运行策略 {cfg.time_s} 秒，FPS: {cfg.fps}")
        
        # 运行策略
        total_frames = cfg.time_s * cfg.fps
        for i in range(total_frames):
            start_time = time.perf_counter()
            
            # 获取观测
            observation = robot.get_observation()
            if observation is None:
                print("观测为空，退出")
                exit()
            
            # 准备观测数据
            batch = prepare_observation_for_policy(observation, policy, device)
            
            # 使用策略选择动作
            with torch.no_grad():
                action_tensor = policy.select_action(batch)
            
            # 转换动作张量为字典
            action = {}
            action_features = list(robot.action_features.keys())
            
            # 处理不同的动作张量形状
            if action_tensor.dim() > 1:
                for i, key in enumerate(action_features):
                    if i < action_tensor.shape[1]:
                        action[key] = action_tensor[0, i].item()
            else:
                for i, key in enumerate(action_features):
                    if i < action_tensor.shape[0]:
                        action[key] = action_tensor[i].item()
            
            # 发送动作到机器人
            robot.send_action(action)
            
            # 显示数据（如果需要）
            if cfg.display_data:
                log_rerun_data(observation, action)
            
            # 周期性打印状态
            dt_s = time.perf_counter() - start_time
            remaining_time = 1 / cfg.fps - dt_s
            
            # 等待以维持FPS
            busy_wait(remaining_time)

            real_dt_s = time.perf_counter() - start_time
            # 保留两位小数，
            print(f"One loop duration: {real_dt_s:.2f}, Real FPS: {1 / real_dt_s:.2f}, Max FPS: {1 / dt_s:.2f}")

            
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        print("断开机器人连接")
        robot.disconnect()


if __name__ == "__main__":
    test_act_policy() 
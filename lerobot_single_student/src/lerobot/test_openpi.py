#!/usr/bin/env python3
# -- coding: UTF-8

"""Test OpenPI client with real robot observations.

Example usage:
python -m lerobot.test_openpi \
    --robot.type=enpei_follower \
    --robot.id=enpei_follower \
    --robot.cameras="{ handeye: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --host=localhost \
    --port=6006 \
    --instruction="Grab the starfruit to the blue box." \
    --fps=30 \
    --enpei_use_radian=true
"""

import time
import numpy as np
import torch
import threading
import queue
from dataclasses import dataclass
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from lerobot.configs import parser
from lerobot.robots import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.utils import init_logging
from lerobot.utils.robot_utils import busy_wait

# Make sure all robot types are imported so they're registered as choices
from lerobot.robots import hope_jr, koch_follower, enpei_follower, so100_follower, so101_follower, lekiwi

#加入可视化工具rerun
import rerun as rr
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

@dataclass
class ClientConfig:
    # Robot configuration
    robot: RobotConfig
    # Remote server configuration
    host: str = "localhost"
    port: int = 6006
    # Task instruction
    instruction: str = "Grab the starfruit to the blue box."
    # Image size for processing
    img_width: int = 320
    img_height: int = 240
    # FPS for policy evaluation
    fps: int = 30

    # 是否显示 Rerun 可视化窗口
    display_data: bool = False
    # Enpei robot use radian instead of degree
    enpei_use_radian: bool = False
    # 推理时间间隔（秒）- 将在运行时动态更新
    inference_interval: float = 0.4 
    # action_start_index: int = 0


class RemotePI0Client:
    def __init__(self, config: ClientConfig):
        self.latest_observation = None
        # 初始化websocket客户端连接到远程策略服务器
        self.client = websocket_client_policy.WebsocketClientPolicy(host=config.host, port=config.port)
        print(f"Connected to remote policy server at {config.host}:{config.port}")
        print(f"Server metadata: {self.client.get_server_metadata()}")
        
        # 设置默认图像大小和指令
        self.img_size = (config.img_width, config.img_height)
        self.instruction = config.instruction
        self.observation_window = None
        print(f"Using instruction: {self.instruction}")
        
        # 初始化机器人连接
        self.robot = make_robot_from_config(config.robot)
        
        # 设置enpei机器人的角度单位
        if config.robot.type == "enpei_follower" and hasattr(self.robot, "use_radian"):
            self.robot.use_radian = config.enpei_use_radian
            print(f"设置enpei follower机器人角度单位: {'弧度' if config.enpei_use_radian else '角度'}")
        
        self.robot.connect()
        print("Robot connected successfully")
        
        # 多线程相关变量
        self.action_queue = queue.Queue(maxsize=1)  # 用于存储最新的动作
        self.running = True  # 控制线程运行的标志
        self.inference_thread = None  # 推理线程
        self.inference_interval = config.inference_interval  # 推理时间间隔
        self.last_inference_time = 0.0  # 记录最后一次推理的执行时间
        # self.action_start_index = config.action_start_index
        self.trigger_inference = True  # 标记推理是否允许

    
 
    # 使用真实机器人观测更新观察窗口缓冲区 (新方法，使用真实机器人数据)
    def update_observation_window_from_robot(self):
        # 从机器人获取真实观测数据
        observation = self.robot.get_observation()
        if observation is None:
            print("Warning: Failed to get observation from robot")
            return False

        # 处理图像数据 - 只使用指定的相机名称
        if 'handeye' not in observation or 'fixed' not in observation:
            print("Warning: Required cameras 'handeye' and 'fixed' not found in observation")
            print(f"Available keys: {list(observation.keys())}")
            return False

        # 保存模型本次推理实际使用的观测，供 Rerun 可视化。
        self.latest_observation = observation

        img = observation['fixed']    
        wrist_img = observation['handeye']
        # resize to 224x224x3
        
        # 这里也许还能加速优化一些
        img = image_tools.resize_with_pad(img, 224, 224)
        wrist_img = image_tools.resize_with_pad(wrist_img, 224, 224)
        

        # print(img.dtype)
        # print(img.shape)
        # 提取机器人状态
        state_values = []
        for motor_name in observation:
            if ".pos" in motor_name:
                state_values.append(observation[motor_name])
        
        # 如果没有找到位置数据，使用默认状态
        if not state_values:
            state_values = np.ones(8)  # 默认8维状态
        else:
            state_values = np.array(state_values, dtype=np.float32)
        
      

        self.observation_window = {
            # if img is float, convert to uint8
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state_values,
            "prompt": self.instruction,
        }
        return True
    
    # 从远程策略服务器获取动作
    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        # 将观察发送到远程服务器并获取动作
        return self.client.infer(self.observation_window)["actions"]
    
    # 重置观察窗口
    def reset_observation_window(self):
        self.observation_window = None
        print("Successfully reset observation window")
    
    # 断开机器人连接
    def disconnect(self):
        if hasattr(self, 'robot'):
            self.robot.disconnect()
            print("Robot disconnected")
    

# 推理线程函数
def inference_thread_func(remote_client):
    print("推理线程启动")
    try:
        while remote_client.running:
          
            # 如果不允许推理，等待
            if not remote_client.trigger_inference:
                # print("暂不允许推理")
                time.sleep(0.01)  # 短暂休眠以减少CPU使用
                continue
                
            # print("开始推理")
            # 记录推理开始时间
            inference_start_time = time.perf_counter()
            
            # 使用真实机器人观测更新观察窗口
            if not remote_client.update_observation_window_from_robot():
                print("Failed to get robot observation, skipping...")
                time.sleep(0.1)
                continue
                
            # 从远程服务器获取动作
            action = remote_client.get_action()
            
            # 获取策略推理时间
            policy_time = time.perf_counter() - inference_start_time
            # print(f"Policy inference time: {policy_time:.4f}s")
            
            # 保存推理时间供主线程使用
            remote_client.last_inference_time = policy_time
            
            # 取前X个结果
            # action = action[:50]
            
            # 将新动作放入队列（如果队列已满，会先清空旧动作）
            if remote_client.action_queue.full():
                try:
                    remote_client.action_queue.get_nowait()
                except queue.Empty:
                    pass
            remote_client.action_queue.put(action)
            

            remote_client.trigger_inference = False

            
            
    except Exception as e:
        print(f"推理线程错误: {e}")
    print("推理线程结束")

@parser.wrap()
def test_openpi_client(cfg: ClientConfig):
    init_logging()

    # 仅在显式启用时启动 Rerun，不影响原有推理和控制流程。
    if cfg.display_data:
        _init_rerun(session_name="openpi_policy_test")

    # 创建一个连接到远程策略服务器的客户端
    remote_client = RemotePI0Client(cfg)
    
    print(f"运行策略，目标FPS: {cfg.fps}")
    
    # 启动推理线程
    remote_client.inference_thread = threading.Thread(
        target=inference_thread_func, 
        args=(remote_client,),
        daemon=True
    )
    remote_client.inference_thread.start()
    
    # 等待第一个动作生成
    print("等待第一个动作生成...")
    while remote_client.action_queue.empty():
        time.sleep(0.1)
    
    first_loop = True
    # 主线程执行动作
    try:
        while True:
            start_time = time.perf_counter()
            loop_inferece = False
            
            # 从队列获取最新动作
            try:
                action = remote_client.action_queue.get(timeout=1.0)
            except queue.Empty:
                print("警告：动作队列为空，等待新动作...")
                continue
            
            # 记录获取动作后的时间
            action_time = time.perf_counter()
            
            # 记录下一次应该触发推理的时间点
            next_inference_time = action_time + remote_client.inference_interval
            
            # 执行动作序列
            for i in range(len(action)):
                # 记录每个动作开始的时间
                action_start_time = time.perf_counter()
                
                # 检查是否到达下一次推理时间点，且推理未在进行中
                if (action_start_time >= next_inference_time ) and (not loop_inferece):
                    # print(f"触发下一次推理，当前动作索引: {i}/{len(action)}")
                    # 更新上次推理时间以触发子线程工作
                    remote_client.trigger_inference = True
                    loop_inferece = True
                    
                
                qpos = action[i]
                
                # 将原始数组转换为字典格式
                action_dict = {}
                motor_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

                # 确保qpos长度不超过motor_names长度
                for j, motor_name in enumerate(motor_names):
                    if j < len(qpos):
                        action_dict[f"{motor_name}.pos"] = float(qpos[j])

                # 每个动作块只记录一次模型观测和首个实际执行的动作，避免重复写入图像拖慢控制循环。
                if (
                    cfg.display_data
                    and i == min(6, len(action) - 1)
                    and remote_client.latest_observation is not None
                ):
                    log_rerun_data(
                        remote_client.latest_observation,
                        action_dict,
                    )
                # 发送动作字典到机器人
                # 跳过一些步骤，因为observation的滞后性，另外，可以让机械臂始终停不下来，保持顺畅
                # 理论上越大越好
                if i <= 5 or i > 50:
                    # print(f"即将跳过:{i}")
                    continue
                else:
                    remote_client.robot.send_action(action_dict)
                
                # 计算当前执行时间
                dt_s = time.perf_counter() - action_start_time
                # 计算需要等待的时间以维持目标FPS
                remaining_time = 1 / cfg.fps - dt_s
                
                # 等待以维持FPS
                busy_wait(remaining_time)
            
            # 计算整体循环时间和FPS
            real_dt_s = time.perf_counter() - start_time
            real_fps = 1 / real_dt_s
            
            # 计算动作执行部分的时间
            action_dt_s = time.perf_counter() - action_time
            action_fps = len(action) / action_dt_s if action_dt_s > 0 else float('inf')
            loop_inferece = False
            first_loop = False
            
            # 动态更新推理间隔 = 整体循环时间 - 推理执行时间
            # 确保推理间隔不会太小，至少保留0.5秒的缓冲
            new_inference_interval = max(0.5, real_dt_s - remote_client.last_inference_time) - 0.01
            remote_client.inference_interval = new_inference_interval
            
            print(f"Loop duration: {real_dt_s:.4f}s, Real FPS: {real_fps:.2f}, Action FPS: {action_fps:.2f}, Action count: {len(action)}, Action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
            # print(f"推理时间: {remote_client.last_inference_time:.4f}s, 新的推理间隔: {new_inference_interval:.4f}s")

    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 停止推理线程
        remote_client.running = False
        if remote_client.inference_thread and remote_client.inference_thread.is_alive():
            remote_client.inference_thread.join(timeout=1.0)
        if cfg.display_data:
            rr.rerun_shutdown()
        # 断开机器人连接
        remote_client.disconnect()


# 示例用法
if __name__ == '__main__':
    test_openpi_client()

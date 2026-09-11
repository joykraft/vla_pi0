#!/usr/bin/env python

"""Script to convert a LeRobot dataset with angles in radians to degrees format.

This script loads an existing LeRobot dataset where angles are stored in radians (0-2π),
creates a new dataset with the same structure, and converts all angle values to degrees (0-360)
while preserving all other data.

The conversion is done by multiplying each angle value by 180/π, which transforms:
- Input range: 0 to 2π radians
- Output range: 0 to 360 degrees

特殊处理：
- 对于第7个关节（夹爪），当use_radian=True时，其值被映射到[0,1]范围而不是真正的弧度值
- 脚本会将夹爪值从[0,1]范围转换回角度范围[20, 110]度

Example usage: 
python convert_radians_to_degrees.py --source_repo_id=enpeicv/move_fruit_radians --target_repo_id=enpeicv/move_fruit_degrees --source_dataset_root=/path/to/source/dataset --push_to_hub=False
"""

import math
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
import tyro
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset, LeRobotDatasetMetadata
from lerobot.teleoperators.enpei_leader.enpei_leader import EnpeiLeader


def radians_to_degrees(radians):
    """
    Convert angles from radians to degrees.
    
    Args:
        radians: Tensor or array of angles in radians
        
    Returns:
        Tensor or array of angles in degrees
    """
    return radians * (180.0 / math.pi)


def convert_dataset(source_repo_id: str, target_repo_id: str, *, push_to_hub: bool = False, max_episodes: int = None, output_path: str = None, source_dataset_root: str = None):
    """Convert a dataset from radians to degrees.
    
    Args:
        source_repo_id: The repository ID of the source dataset (with angles in radians)
        target_repo_id: The repository ID for the new dataset (with angles in degrees)
        push_to_hub: Whether to push the new dataset to the Hugging Face Hub
        max_episodes: Maximum number of episodes to convert. If None, all episodes are converted.
        output_path: Custom output path for the dataset. If None, uses HF_LEROBOT_HOME/target_repo_id.
        source_dataset_root: Custom root path for the source dataset. If None, uses HF_LEROBOT_HOME/source_repo_id.
    
    注意：
        对于第7个关节（夹爪），当use_radian=True时，其值被映射到[0,1]范围而不是真正的弧度值。
        本函数会将夹爪值从[0,1]范围转换回角度范围[20, 110]度。
    """
    # Clean up any existing dataset in the output directory
    if output_path is None:
        dataset_path = HF_LEROBOT_HOME / target_repo_id
    else:
        dataset_path = Path(output_path)
    
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Load metadata from the source dataset
    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_dataset_root)
    print(f"Loaded metadata from {source_repo_id}")
    print(f"Total episodes: {source_meta.total_episodes}")
    print(f"Total frames: {source_meta.total_frames}")
    
    # Load the source dataset
    source_dataset = LeRobotDataset(source_repo_id, root=source_dataset_root)
    print(f"Loaded source dataset with {source_dataset.num_frames} frames")
    
    # Create a new dataset with the same structure
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        robot_type=source_meta.robot_type,
        fps=source_meta.fps,
        features=source_meta.features,
        use_videos=True,  # Preserve video format if present
        image_writer_threads=10,
        image_writer_processes=5,
        root=dataset_path if output_path is not None else None,
    )
    print(f"Created new dataset at {dataset_path}")
    
    # Determine how many episodes to process
    num_episodes = min(source_meta.total_episodes, max_episodes) if max_episodes else source_meta.total_episodes
    print(f"Converting {num_episodes} episodes out of {source_meta.total_episodes}")
    
    # Process each episode
    for episode_idx in tqdm.tqdm(range(num_episodes)):
        # Get the frame indices for this episode
        from_idx = source_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = source_dataset.episode_data_index["to"][episode_idx].item()
        
        # Get the task for this episode
        episode_info = source_dataset.meta.episodes[episode_idx]
        task = episode_info["tasks"][0] if "tasks" in episode_info and episode_info["tasks"] else None
        
        # Process each frame in the episode
        for frame_idx in range(from_idx, to_idx):
            # Get the frame data
            frame = source_dataset[frame_idx]
            
            # Create a new frame with converted angles
            new_frame = {}
            
            # Copy only the feature data from the original frame, excluding metadata fields
            # that are automatically handled by add_frame (timestamp, frame_index, episode_index, etc.)
            for key, value in frame.items():
                # Skip metadata fields that are handled by add_frame
                if key in ["timestamp", "frame_index", "episode_index", "index", "task_index", "task"]:
                    continue
                    
                if key == "observation.state" or key == "action":
                    # 创建一个新的张量/数组来存储转换后的值
                    if isinstance(value, torch.Tensor):
                        converted_value = value.clone()
                    else:
                        converted_value = value.copy()
                    
                    # 对前6个关节角度（如果存在）进行弧度到角度的转换
                    if len(value) >= 6:
                        converted_value[:6] = radians_to_degrees(value[:6])
                    
                    # 对第7个关节（夹爪）进行特殊处理
                    # 夹爪在use_radian=True时是映射到[0,1]范围，而不是真正的弧度值
                    # 需要将其转回到角度范围[min_gripper_angle, max_gripper_angle]
                    if len(value) >= 7:
                        # 使用EnpeiLeader类中定义的夹爪角度范围，保持一致性
                        # 与enpei_follower.py中的转换逻辑相同
                        
                        # 将[0,1]范围的值转换回角度范围
                        converted_value[6] = value[6] * (EnpeiLeader.max_gripper_angle - EnpeiLeader.min_gripper_angle) + EnpeiLeader.min_gripper_angle
                    
                    new_frame[key] = converted_value
                # process observation.images.handeye and observation.images.fixed
                elif key.startswith("observation.images."):
                    # Convert images from (3, 480, 640) to (480, 640, 3) format
                    new_frame[key] = np.transpose(value, (1, 2, 0))
                else:
                    # Keep other data unchanged
                    new_frame[key] = value
            
            # Add the frame to the new dataset
            # The timestamp will be automatically calculated based on frame_index and fps
            target_dataset.add_frame(new_frame, task=task)
        
        # Save the episode
        target_dataset.save_episode()
    
    # Dataset conversion complete
    print(f"Converted dataset saved to {output_path}")
    
    # Push to the Hugging Face Hub if requested
    if push_to_hub:
        print(f"Pushing dataset to the Hugging Face Hub as {target_repo_id}")
        target_dataset.push_to_hub(
            tags=["lerobot", "enpei", "degrees"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Dataset pushed successfully!")


def main(source_repo_id: str = "enpeicv/move_fruit_radians", 
         target_repo_id: str = "enpeicv/move_fruit_degrees", 
         *, push_to_hub: bool = False,
         max_episodes: int = None,
         output_path: str = None,
         source_dataset_root: str = None):
    """Main function to convert a dataset from radians to degrees.
    
    Args:
        source_repo_id: The repository ID of the source dataset (with angles in radians)
        target_repo_id: The repository ID for the new dataset (with angles in degrees)
        push_to_hub: Whether to push the new dataset to the Hugging Face Hub
        max_episodes: Maximum number of episodes to convert. If None, all episodes are converted.
        output_path: Custom output path for the dataset. If None, uses HF_LEROBOT_HOME/target_repo_id.
        source_dataset_root: Custom root path for the source dataset. If None, uses HF_LEROBOT_HOME/source_repo_id.
    
    注意：
        对于第7个关节（夹爪），当use_radian=True时，其值被映射到[0,1]范围而不是真正的弧度值。
        本函数会将夹爪值从[0,1]范围转换回角度范围[20, 110]度。
    """
    convert_dataset(source_repo_id, target_repo_id, push_to_hub=push_to_hub, max_episodes=max_episodes, output_path=output_path, source_dataset_root=source_dataset_root)


if __name__ == "__main__":
    tyro.cli(main)


# python enpei_dataset_meta/convert_radians_to_degrees.py \
#     --source_repo_id=enpeicv/move_fruit_radians \
#     --target_repo_id=enpeicv/move_fruit_degrees \
#     --output_path=/media/enpei/娱乐/ubuntu_lerobot_dataset \
#     --source_dataset_root=/path/to/source/dataset \
#     --max_episodes=1
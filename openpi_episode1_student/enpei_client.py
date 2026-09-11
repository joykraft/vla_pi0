#!/usr/bin/env python3
# -- coding: UTF-8

import time
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

class RemotePI0Client:
    def __init__(self, host="localhost", port=6006, instruction="place the starfruit on plate."):
        # Initialize the websocket client to connect to the remote policy server
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        print(f"Connected to remote policy server at {host}:{port}")
        print(f"Server metadata: {self.client.get_server_metadata()}")
        
        # Set default image size and instruction
        self.img_size = (640, 480)
        self.instruction = instruction
        self.observation_window = None
        print(f"Using instruction: {self.instruction}")
    
    # Set image size for resizing
    def set_img_size(self, img_size):
        self.img_size = img_size
        print(f"Image size set to {img_size}")
    
    # Set language instruction
    def set_instruction(self, instruction):
        self.instruction = instruction
        print(f"Instruction set to: {instruction}")
    
    # Update the observation window buffer with camera images and robot state
    def update_observation_window(self, img_arr, state):
        # Process images from the camera array
        img_hand = img_arr['images']['handeye']
        img_wrist = img_arr['images']['fixed']
        
        # Resize images to reduce bandwidth and convert to uint8
        img_hand = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img_hand, self.img_size[0], self.img_size[1])
        )
        img_wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img_wrist, self.img_size[0], self.img_size[1])
        )
        
        # Transpose images to match the expected format (C, H, W)
        img_hand = np.transpose(img_hand, (2, 0, 1))
        img_wrist = np.transpose(img_wrist, (2, 0, 1))
        
        # Create the observation dictionary
        self.observation_window = {
            "state": state,
            "images": {
                "fixed": img_hand,
                "handeye": img_wrist,
            },
            "prompt": self.instruction,
        }
    
    # Get action from the remote policy server
    def get_action(self):
        assert (self.observation_window is not None), "update observation_window first!"
        # Send the observation to the remote server and get the action
        return self.client.infer(self.observation_window)["actions"]
    
    # Reset observation window
    def reset_observation_window(self):
        self.observation_window = None
        print("Successfully reset observation window")
    
    # Create a random example for testing
    def make_aloha_example(self) -> dict:
        """Creates a random input example for the Aloha policy."""
        # Create random images and state
        self.observation_window = {
            "state": np.ones((8,)),
            "images": {
                "fixed": np.random.randint(256, size=(3, self.img_size[0], self.img_size[1]), dtype=np.uint8),
                "handeye": np.random.randint(256, size=(3, self.img_size[0], self.img_size[1]), dtype=np.uint8),
            },
            "prompt": self.instruction,
        }
        return self.observation_window

# Example usage
if __name__ == '__main__':
    # Create a client that connects to the remote policy server
    # Change host and port as needed to match your server configuration
    remote_client = RemotePI0Client(host="localhost", port=6006)
    
    # Run a continuous loop to test the connection
    try:
        while True:
            time_start = time.perf_counter()
            
            # Create a random observation example
            obs = remote_client.make_aloha_example()
            
            # Get action from the remote server
            action = remote_client.get_action()
            act = action[:, :7]  # Extract the first 7 dimensions of the action
            
            # Calculate time cost and fps
            time_end = time.perf_counter()
            time_cost = time_end - time_start
            fps = 1 / time_cost
            print(f"time cost: {time_cost:.4f}s, fps: {fps:.2f}, action shape: {act.shape}")
            
            # Add a small delay to avoid flooding the server
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Error: {e}")

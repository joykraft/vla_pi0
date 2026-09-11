from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=6006)


for step in range(10):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    img = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    wrist_img = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    state = np.random.rand(10)
    task_instruction = "Pick up the red block"

    print(img.dtype)
    print(img.shape)
    
    

    # to sum up, the final img must be int8, 224x224, otherwise the speed will be very slow
    # because the image will be resized to 224x224 in the server side

    observation = {
        # if img is float, convert to uint8
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    # print(action_chunk)

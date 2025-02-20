import torch
import os
import cv2
import numpy as np
import json
import pandas as pd
from pathlib import Path

dataset_name = "Ityl/so100_recording1"
dino_ds_folder = "dataset_dino"

H, W = 224, 224

user_dir = str(Path.home())
lerobot_ds_folder = user_dir + "/.cache/huggingface/lerobot/" + dataset_name # dataset_hugging

def video_to_tensor(video_path):
    """Function to load the video and convert it into a tensor"""
    # Read the video with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame from BGR (OpenCV format) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to desired dimensions (H, W)
        frame = cv2.resize(frame, (H, W))
        
        frames.append(frame)
    
    cap.release()
    
    # Convert the frame list into a tensor
    video_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    return video_tensor


def save_video_as_pth(video_folder, save_folder):
    """Function to save videos as .pth files"""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for video_name in os.listdir(video_folder):
        print(video_name)
        video_path = os.path.join(video_folder, video_name)
        
        if os.path.isfile(video_path) and video_name.endswith(".mp4"):
            print(f"Processing {video_name}...")
            video_tensor = video_to_tensor(video_path)
            
            # Saving the tensor in a .pth file
            save_path = os.path.join(save_folder, f"{os.path.splitext(video_name)[0]}.pth")
            torch.save(video_tensor, save_path)
            print(f"Saved {save_path}")

# Path to the folder with .mp4 videos
video_folder = lerobot_ds_folder + '/videos/chunk-000/observation.images.realsense_side'
# Path where you want to save videos as .pth files
save_folder = dino_ds_folder + '/obses'

# save_video_as_pth(video_folder, save_folder)

lengths = []
with open(lerobot_ds_folder + "/meta/episodes.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)  # Convertir la ligne en dictionnaire
        lengths.append(data["length"])  # Extraire "length"

with open(lerobot_ds_folder + "/meta/info.json", "r") as f:
    data = json.load(f)  # Convertir le fichier en dictionnaire
    action_shape = data["features"]["action"]["shape"]
    state_shape = data["features"]["observation.state"]["shape"]

# Convertir en tenseur PyTorch
tensor_lengths = torch.tensor(lengths)
# Sauvegarder dans un fichier .pth
torch.save(tensor_lengths, dino_ds_folder + "/seq_length.pth")

nb_episodes = len(tensor_lengths)
max_length = tensor_lengths.max().item()

tensors_states = []
tensors_actions = []

for episode in range(nb_episodes):

    df = pd.read_parquet(f"{lerobot_ds_folder}/data/chunk-000/episode_{episode:06d}.parquet", engine="pyarrow")
    
    tensor_states = torch.tensor(np.array(df["observation.state"].to_list()))  # Convertir en tenseur PyTorch
    tensor_states = torch.cat((tensor_states, torch.zeros((max_length-tensor_lengths[episode], *state_shape))), dim=0) # pad zeros to max tensor length
    tensors_states.append(tensor_states)

    tensor_actions = torch.tensor(np.array(df["action"].to_list()))  # Convertir en tenseur PyTorch
    tensor_actions = torch.cat((tensor_actions, torch.zeros((max_length-tensor_lengths[episode], *action_shape))), dim=0) # pad zeros to max tensor length
    tensors_actions.append(tensor_actions)

# Sauvegarder en fichier .pth dans un autre dossier
tensors_states = torch.stack(tensors_states)
torch.save(tensors_states, dino_ds_folder + "/states.pth")

# Sauvegarder en fichier .pth dans un autre dossier
tensors_actions = torch.stack(tensors_actions)
torch.save(tensor_actions, dino_ds_folder + "/actions.pth")

print(tensors_states.shape)
print(tensors_actions.shape)
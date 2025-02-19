import torch
import os
import cv2
import numpy as np
import datasets
import json


# Función para cargar el video y convertirlo en tensor
def video_to_tensor(video_path):
    # Leer el video con OpenCV
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir el frame de BGR (formato OpenCV) a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar a las dimensiones deseadas (H, W)
        frame = cv2.resize(frame, (224, 224))  # Cambia el tamaño según sea necesario
        
        frames.append(frame)
    
    cap.release()
    
    # Convertir la lista de frames en un tensor
    video_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    return video_tensor

# Función para guardar los videos como archivos .pth
def save_video_as_pth(video_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    for video_name in os.listdir(video_folder):
        print(video_name)
        video_path = os.path.join(video_folder, video_name)
        
        if os.path.isfile(video_path) and video_name.endswith(".mp4"):
            print(f"Procesando {video_name}...")
            video_tensor = video_to_tensor(video_path)
            
            # Guardar el tensor en un archivo .pth
            save_path = os.path.join(save_folder, f"{os.path.splitext(video_name)[0]}.pth")
            torch.save(video_tensor, save_path)
            print(f"Guardado {save_path}")

# Ruta a la carpeta con los videos .mp4
video_folder = 'dataset_hugging/videos/chunk-000/observation.images.realsense_side'
# Ruta donde quieres guardar los archivos .pth
save_folder = 'dataset_dino/obses'

save_video_as_pth(video_folder, save_folder)

import pandas as pd

df = pd.read_parquet("dataset_hugging/chunk-000/episode_000000.parquet", engine="pyarrow")  # O usa "fastparquet"
print(df.info())

tensor_states = torch.tensor(np.array(df["observation.state"].to_list()))  # Convertir en tenseur PyTorch

# Sauvegarder en fichier .pth dans un autre dossier
torch.save(tensor_states, "dataset_dino/states.pth")

tensor_actions = torch.tensor(np.array(df["action"].to_list()))  # Convertir en tenseur PyTorch

# Sauvegarder en fichier .pth dans un autre dossier
torch.save(tensor_actions, "dataset_dino/actions.pth")

lengths = []
with open("dataset_hugging/meta/episodes.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)  # Convertir la ligne en dictionnaire
        lengths.append(data["length"])  # Extraire "length"

# Convertir en tenseur PyTorch
tensor_lengths = torch.tensor(lengths)

# Sauvegarder dans un fichier .pth
torch.save(tensor_lengths, "dataset_dino/seq_length.pth")



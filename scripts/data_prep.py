"""
This Script processes the original dataset for our model
This includes:
 - turning them into 256x256
 - turn them into LAB
 - extract A and B, quantize (These will be the labels)
 - use L as grayscale image for training and testing
"""

import os
import numpy as np
import cv2
from PIL import ImageFile
from skimage import io, color, transform
from lab_quantization import *
from torchvision import transforms
import torch
from torchvision.transforms.functional import to_tensor
from concurrent.futures import ThreadPoolExecutor


ImageFile.LOAD_TRUNCATED_IMAGES = True  # needed because otherwise it gives an error

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data")

print(data_dir)

input_dir_train = os.path.join(data_dir, f"original_data{os.sep}train")
input_dir_test = os.path.join(data_dir, f"original_data{os.sep}test")
input_dir_val = os.path.join(data_dir, f"original_data{os.sep}val")

output_dir_L_train = os.path.join(data_dir, f"preprocessed_data{os.sep}L_channel{os.sep}train")
output_dir_L_test = os.path.join(data_dir, f"preprocessed_data{os.sep}L_channel{os.sep}test")
output_dir_L_val = os.path.join(data_dir, f"preprocessed_data{os.sep}L_channel{os.sep}val")

output_dir_AB_train = os.path.join(data_dir, f"preprocessed_data{os.sep}AB_channel{os.sep}train")
output_dir_AB_test = os.path.join(data_dir, f"preprocessed_data{os.sep}AB_channel{os.sep}test")
output_dir_AB_val = os.path.join(data_dir, f"preprocessed_data{os.sep}AB_channel{os.sep}val")


def process_image(img_path, ab_grid, target_size=(256, 256)):

    try:
        # load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # Resize and convert lab
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

        # normalize L to [0:1]
        L_channel = lab[:, :, 0] / 255.0  

        # Ab-Channel to [-127:128]
        AB_channel = lab[:, :, 1:] - 128  

        # make tensor
        L_tensor = torch.tensor(L_channel, dtype=torch.float32, device="cuda")
        AB_tensor = torch.tensor(AB_channel, dtype=torch.float32, device="cuda")

        # quantize
        labels = quantize_ab(AB_tensor.cpu().numpy(), ab_grid)

        return L_tensor.cpu().numpy(), labels

    except Exception as e:
        print(f"Fehler bei {img_path}: {e}")
        return None, None

def process_and_save_images(input_dir, output_dir_L, output_dir_AB, ab_grid, num_workers=8):

    os.makedirs(output_dir_L, exist_ok=True)
    os.makedirs(output_dir_AB, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png',".JPEG"))]
    
    # Multi-Threading
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for file_name in image_files:
            img_path = os.path.join(input_dir, file_name)
            futures[file_name] = executor.submit(process_image, img_path, ab_grid)

        for idx, (file_name, future) in enumerate(futures.items()):
            L_norm, labels = future.result()
            if L_norm is not None and labels is not None:
                base_name = os.path.splitext(file_name)[0]
                np.save(os.path.join(output_dir_L, f"{base_name}_L.npy"), L_norm)
                np.save(os.path.join(output_dir_AB, f"{base_name}_AB.npy"), labels)
                print(f"Verarbeitet & gespeichert: {file_name} ({idx + 1}/{len(image_files)})")


ab_grid = lab_bins()

# train images
process_and_save_images(input_dir_train, output_dir_L_train, output_dir_AB_train, ab_grid)
# test images
process_and_save_images(input_dir_test, output_dir_L_test, output_dir_AB_test, ab_grid)
# val images
process_and_save_images(input_dir_val, output_dir_L_val, output_dir_AB_val, ab_grid)

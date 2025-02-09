"""
In this script, I will prepare the dataset for the usage in the CNN
This includes:
 - turning them into 256x256
 - turn them into LAB
 - extract A and B, turn into classes (These will be the labels)
 - use L as grayscale image for training and testing
"""

import os
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFile
from skimage import io, color, transform
from lab_quantization import *

ImageFile.LOAD_TRUNCATED_IMAGES = True #needed because otherwise it gives an error

data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data")


def process_and_save_images(input_dir, output_dir_L, output_dir_AB, ab_grid, target_size=(256,256)):
    """
    reads all images of input_dir, which should be the original images and transforms them as follows:
    - resize
    - RGB to LAB transformation
    - Split L Channel from AB Channels
    - Quantize AB-values
    - save L channel under output_dir_L
    - save quantized AB Channels under output_dir_AB
    """
    # chech whether dirs exist
    os.makedirs(output_dir_L, exist_ok=True)
    os.makedirs(output_dir_AB, exist_ok=True)

    # List of Image-Data
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    for idx, file_name in enumerate(image_files):
        img_path = os.path.join(input_dir, file_name)
        try:
            # load image
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error with loading of file: {file_name}: {e}")
            continue

        # resize
        img = img.resize(target_size)

        # transform into numpy array
        img_np = np.array(img)

        # transform into lab
        lab = color.rgb2lab(img_np)

        # split L and AB
        L_channel = lab[:,:,0]
        AB_channel = lab[:,:,1:]

        # normalize L from 0-100 to 0-1
        L_norm = L_channel / 100.0

        # quantize AB, sort to bin centers
        labels = quantize_ab(AB_channel, ab_grid)

        # save as npy
        base_name = os.path.splitext(file_name)[0]
        np.save(os.path.join(output_dir_L, f"{base_name}_L.npy"), L_norm)
        np.save(os.path.join(output_dir_AB, f"{base_name}_AB.npy"), labels)
        print(f"Transformed and saved: {file_name} ({idx+1}/{len(image_files)})")

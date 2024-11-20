"""
In this script, i will prepare the dataset for the usage in the CNN
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
ImageFile.LOAD_TRUNCATED_IMAGES = True #needed because otherwise it gives an error

data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data")

def turn_image_lab(img):
    img = img.convert("RGB")
    img_np = np.array(img)
    lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)

    return lab_img


# resizes all images in datadir to width*height
# turns all images in datadir into Lab colorspace
def transform_images(datadir, width, height):
    # skimming through every image in datadir with .jpg ending
    for root, dirs, files in os.walk(datadir):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)

                with Image.open(file_path) as img:
                    # resizing image
                    if img.size != (width, height):
                        img = img.resize((width, height))
                        print(f"Bild {file} in {root} auf {width}*{height} transformiert.")

                    # turn image to Lab colorspace
                    lab_img = turn_image_lab(img)
                    np.save(file_path, lab_img)
print(data_dir)
transform_images(data_dir, 256,256)

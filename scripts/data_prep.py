"""
In this script, i will prepare the dataset for the usage in the CNN
This includes:
 - turning them into 256x256
 - turn them into LAB
 - extract A and B, turn into classes (These will be the labels)
 - use L as grayscale image for training and testing
"""

import os
from PIL import Image


data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "dataset")


def turn_image_lab():
    pass


def extract_labels():
    # take ab values and classify them into one of the bins.
    # save the labels
    pass


def extract_l(labimg):
    pass


# resizes all images in datadir to width*height
# turns all images in datadir into Lab colorspace
def transform_images(datadir, width, height):
    # skimming through every image in datadir with .jpg ending
    for root, _, files in os.walk(datadir):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)

                with Image.open(file_path) as img:
                    # resizing image
                    if img.size != (width, height):
                        img_resized = img.resize((width, height))
                        img_resized.save(file_path)
                        print(f"Bild {file} in {root} auf {width}*{height} transformiert.")

                    # turn image to Lab colorspace




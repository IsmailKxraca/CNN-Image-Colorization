"""
This script was used to structure the 8.000 Car images for the prototype.
Train, Test, Validation split as follows:
Train 70%, Test 15%, Val 15%
"""

import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

data_dir = "data"
output_train_dir = "dataset/train"
output_val_dir = "dataset/val"
output_test_dir = "dataset/test"

# create directories
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# find all image-paths
image_paths = glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)

# Train, test, val split
train_paths, temp_paths = train_test_split(image_paths, test_size=0.3, random_state=42)
val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)


# Helper-function for copying images
def copy_images(paths, target_dir):
    for path in paths:
        filename = os.path.basename(path)
        shutil.copy(path, os.path.join(target_dir, filename))


# copy images
copy_images(train_paths, output_train_dir)
copy_images(val_paths, output_val_dir)
copy_images(test_paths, output_test_dir)

print("Success")

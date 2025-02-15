"""
This script counts all labels in the data and saves the classfrequncies
"""

import numpy as np
import os
from collections import Counter

# folder with labels
label_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"data{os.sep}preprocessed_data{os.sep}AB_Channel{os.sep}train")

# list with all label files
label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".npy")]

# Counter for classfrequency
class_counts = Counter()

# parse through all label files and count classfrquencies
for file in label_files:
    labels = np.load(file)
    print(labels)
    class_counts.update(labels.flatten())

# calculate total pixel amount
total_pixels = sum(class_counts.values())

# save classfrquency as array
num_classes = 484
class_frequencies = np.array([class_counts[i] / total_pixels if i in class_counts else 0 for i in range(num_classes)])

# save
np.save("class_frequencies.npy", class_frequencies)
print("finished")

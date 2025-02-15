import numpy as np
import os
from collections import Counter

# Ordner mit deinen Label-Dateien
label_dir = r"/workspace/CNN-Image-Colorization/data/preprocessed_data/AB_channel/train"

# Alle Label-Dateien abrufen
label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".npy")]

# Counter für die Häufigkeiten der Klassen
class_counts = Counter()
i = 0

# Alle Label-Dateien einlesen und Klassen zählen
for file in label_files:
    labels = np.load(file)
    print(labels)
    class_counts.update(labels.flatten())
    i += 1
    print(i)

# calculate total pixel amount
total_pixels = sum(class_counts.values())

# save count as array
num_classes = 484
class_frequencies = np.array([class_counts[i] / total_pixels if i in class_counts else 0 for i in range(num_classes)])

# save
np.save("class_frequencies.npy", class_frequencies)
print("finished")

"""
In this script i will code a dataloader generator for loading the data in the training process
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PreprocessedColorizationDataset(Dataset):
    def __init__(self, folder_L, folder_AB):
        # creating list of base_names without _L or _Ab at the end
        self.base_names = [os.path.splitext(f)[0].replace("_L", "")
                           for f in os.listdir(folder_L) if f.endswith("_L.npy")]
        self.folder_L = folder_L
        self.folder_AB = folder_AB

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        base = self.base_names[idx]
        L = np.load(os.path.join(self.folder_L, f"{base}_L.npy"))
        AB = np.load(os.path.join(self.folder_AB, f"{base}_AB.npy"))
        # transform into tensor
        L_tensor = torch.tensor(L, dtype=torch.float32).unsqueeze(0)
        AB_tensor = torch.tensor(AB, dtype=torch.long)
        return L_tensor, AB_tensor

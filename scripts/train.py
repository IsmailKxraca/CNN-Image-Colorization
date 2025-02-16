"""
This is the train script, where the training of the model happens.
If a Cuda-Gpu is availabe mixed precision will be used

This script includes tensorboard apllicated, the files are saved in "runs" folder
to see the tensorboard you have to use this comand: tensorboard --logdir="path to folder"

In this version Crossentropyloss with Classrebalancing (dont know whether it works) is used, with a former calculated class_frequency file. (caculated with rebalancing.py)
The Optimizer is Adam

Checkpionts of every epoch is being saved in checkpints folder (empty in this version, because files are too big for github)
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from cnn import *
from dataset import *

batch_size = 32
num_workers = 8
epochs = 50
training_name = "with_rebalancing"

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

# folder paths
folder_l = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"CNN-Image-Colorization{os.sep}data{os.sep}preprocessed_data{os.sep}L_channel{os.sep}train")
folder_ab = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"CNN-Image-Colorization{os.sep}data{os.sep}preprocessed_data{os.sep}AB_channel{os.sep}train")
folder_checkpoints = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), r"checkpoints")

# device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32

# dataset loader
train_dataset = PreprocessedColorizationDataset(folder_l, folder_ab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=4)

model = ColorizationCNN().to(device)

# compile for better performance
model = torch.compile(model)

# tensorboard initialization 
writer = SummaryWriter(f"runs/{training_name}")

# class rebalancing
class_frequencies = np.load(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), r"class_frequencies.npy"))

class_weights = 1 - class_frequencies

# min max normalization
class_weights = class_weights / class_weights.sum()

# tensor
class_weights = torch.tensor(class_weights, dtype=dtype).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

scaler = torch.amp.GradScaler("cuda")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (l_channel, ab_classes) in enumerate(loop):
        l_channel = l_channel.to(device)
        ab_classes = ab_classes.to(device)

        optimizer.zero_grad()

        # mixed precision
        with torch.amp.autocast("cuda", dtype=dtype):
            preds = model(l_channel)
            loss = criterion(preds, ab_classes)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

        writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(train_loader) + batch_idx)

    avg_epoch_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)

    print(f"Epoch {epoch + 1}: Average Loss: {avg_epoch_loss:.4f}")

    checkpoint_path = os.path.join(folder_checkpoints, f"checkpoint_{training_name}_epoch_{epoch + 1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")


# save end model
torch.save(model.state_dict(), f"colorization_model_{training_name}.pth")
print("Training finished!")


# test-phase
print("Starting test")

folder_L_test = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"data{os.sep}preprocessed_data{os.sep}L_channel{os.sep}test")
folder_ab_test = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"data{os.sep}preprocessed_data{os.sep}AB_channel{os.sep}train")

test_dataset = PreprocessedColorizationDataset(folder_L_test, folder_ab_test)
test_loader=DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
model.eval()

test_loss = 0.0

with torch.no_grad():
    loop = tqdm(test_loader, desc="Testphase")
    for l_channel, ab_classes in loop:
        l_channel, ab_classes = l_channel.to(device), ab_classes.to(device)

        with torch.cuda.amp.autocast():
            preds = model(l_channel)
            loss = criterion(preds, ab_classes)

        test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    writer.add_scalar("Loss/test", avg_test_loss)
    
print(f"Average Test-Loss: {avg_test_loss:.4f}")

writer.close()

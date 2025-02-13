import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from cnn import *
from dataset import *

batch_size = 8
num_workers = 0
epochs = 10

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

# folder paths
folder_l = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"data{os.sep}preprocessed_data{os.sep}L_channel{os.sep}train")
folder_ab = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), f"data{os.sep}preprocessed_data{os.sep}AB_channel{os.sep}train")
folder_checkpoints = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), r"checkpoints")

# device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

# dataset
train_dataset = PreprocessedColorizationDataset(folder_l, folder_ab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=4)

model = ColorizationCNN().to(device, dtype=dtype)
model = torch.compile(model)

# tensorboard initialization + CNN visualization
writer = SummaryWriter("runs/colorization_experiment")

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler("cuda")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (l_channel, ab_classes) in enumerate(loop):
        l_channel = l_channel.to(device, dtype=dtype)
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

    checkpoint_path = os.path.join(folder_checkpoints, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")


# save end model
torch.save(model.state_dict(), "colorization_model.pth")
print("Training finished!")


# **Nach dem Training: Testen**
print("Starte Testphase...")

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
    
print(f"Durchschnittlicher Test-Loss: {avg_test_loss:.4f}")

writer.close()

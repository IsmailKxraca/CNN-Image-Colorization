import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cnn import *
from dataset import *

# folder paths
folder_l = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), r"data\preprocessed_data\L_channel\train")
folder_ab = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), r"data\preprocessed_data\AB_channel"
                                                                                r"\train")
folder_checkpoints = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), r"checkpoints")

# device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# dataset
train_dataset = PreprocessedColorizationDataset(folder_l, folder_ab)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4)

model = ColorizationCNN().to(device, dtype=dtype)
model = torch.compile(model)

# tensorboard initialization + CNN visualization
writer = SummaryWriter("runs/colorization_experiment")
dummy_input = torch.randn(1, 1, 256, 256).to(device, dtype=dtype)
writer.add_graph(model, dummy_input)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 10
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (l_channel, ab_classes) in enumerate(loop):
        l_channel = l_channel.to(device, dtype=dtype)
        ab_classes = ab_classes.to(device)

        optimizer.zero_grad()

        # mixed precision
        with torch.cuda.amp.autocast(dtype=dtype):
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

writer.close()

# save end model
torch.save(model.state_dict(), "colorization_model.pth")
print("Training finished!")

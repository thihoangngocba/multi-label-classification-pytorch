import models
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from engine import train, validate
from dataset import MultiAttribDataset
from torch.utils.data import DataLoader
matplotlib.style.use('ggplot')

IMAGE_SIZE=(224,224)

# Initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Intialize the model
model = models.resnet50(pretrained=True, requires_grad=False, num_classes=3).to(device)

# Learning parameters
lr = 0.0001
epochs = 1
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# Train dataset & dataloader
train_csv = pd.read_csv('./dataset/train.csv')
train_data = MultiAttribDataset(
    train_csv, size=IMAGE_SIZE
)
train_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True
)

# Validation dataset & dataloader
val_csv = pd.read_csv('./dataset/val.csv')
valid_data = MultiAttribDataset(
    val_csv, size=IMAGE_SIZE
)
valid_loader = DataLoader(
    valid_data, 
    batch_size=batch_size,
    shuffle=False
)

# Start the training and validation
train_loss = []
valid_loss = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device
    )
    train_loss.append(train_epoch_loss)

    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_data, device
    )
    valid_loss.append(valid_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

# Save the trained model
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, './outputs/model.pth')

# Plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./outputs/loss.png')
plt.show()

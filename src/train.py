import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_dataloaders
from model import build_model
import matplotlib.pyplot as plt

DATA_DIR = "data/raw"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
model = build_model(len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item()

    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}: train loss {train_losses[-1]:.4f}, val loss {val_losses[-1]:.4f}")

torch.save(model.state_dict(), "results/model.pth")

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.savefig("results/training_curves.png")


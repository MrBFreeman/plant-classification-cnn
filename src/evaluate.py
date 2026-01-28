import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_dataloaders
from model import build_model

DATA_DIR = "data/raw"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

model = build_model(len(classes))
model.load_state_dict(torch.load("results/model.pth"))
model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, 1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.savefig("results/confusion_matrix.png")

report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv("results/metrics.csv")

print(df)


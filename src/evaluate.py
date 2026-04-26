import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ================= CONFIG =================

MODEL_PATH = "../models/best_model.pth"
DATA_DIR = "../data"

BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

# ================= TRANSFORMS =================

transform = transforms.Compose([
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ================= DATA =================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

classes = dataset.classes
print("Classes:", classes)

# mesmo split do treino (fixo/reprodutível)
generator = torch.Generator().manual_seed(42)

train_size = int(
    0.8 * len(dataset)
)

test_size = len(dataset) - train_size

_, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)

loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= MODEL =================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# ================= EVALUATION =================

all_preds = []
all_labels = []

with torch.no_grad():

    for x,y in loader:

        x=x.to(device)
        y=y.to(device)

        out=model(x)

        preds=torch.argmax(
            out,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            y.cpu().numpy()
        )


# ================= RESULTS =================

acc = accuracy_score(
    all_labels,
    all_preds
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nAccuracy:")
print(f"{acc*100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        digits=4
    )
)
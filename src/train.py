import os
import multiprocessing
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def get_auto_config():
    cfg = {}

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9

        if vram < 4:
            batch = 16
        elif vram < 8:
            batch = 32
        elif vram < 16:
            batch = 64
        else:
            batch = 128

        cfg["device"] = torch.device("cuda")
        cfg["batch"] = batch
        cfg["use_amp"] = True
        cfg["pin_memory"] = True

        torch.backends.cudnn.benchmark = True

        print("GPU:", props.name)
        print(f"VRAM: {vram:.1f} GB")

    else:
        cfg["device"] = torch.device("cpu")
        cfg["batch"] = 8
        cfg["use_amp"] = False
        cfg["pin_memory"] = False
        print("Rodando em CPU")

    # Windows-safe
    cfg["workers"] = 0

    return cfg


def main():

    cfg = get_auto_config()

    device = cfg["device"]
    BATCH_SIZE = cfg["batch"]
    NUM_WORKERS = cfg["workers"]
    USE_AMP = cfg["use_amp"]
    PIN_MEMORY = cfg["pin_memory"]

    LR = 3e-4

    CHECKPOINT_PATH = "../models/checkpoint.pth"
    BEST_MODEL_PATH = "../models/best_model.pth"

    os.makedirs("../models", exist_ok=True)

    # ================= TRANSFORMS =================

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    # ================= DATA =================

    dataset = datasets.ImageFolder(
        "../data",
        transform=train_transform
    )

    train_size = int(0.8*len(dataset))
    test_size = len(dataset)-train_size

    train_data,test_data = random_split(
        dataset,
        [train_size,test_size]
    )

    test_data.dataset.transform = test_transform

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("Classes:", dataset.classes)
    print("Batch:", BATCH_SIZE)

    # ================= MODEL =================

    model = models.resnet18(
        weights="DEFAULT"
    )

    for p in model.parameters():
        p.requires_grad=False

    model.fc = nn.Linear(
        model.fc.in_features,
        len(dataset.classes)
    ) 

    model=model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LR
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # ================= RESUME =================

    epoch=0
    best_val_loss=float("inf")

    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(
                CHECKPOINT_PATH,
                map_location=device
            )

            if isinstance(ckpt,dict):
                model.load_state_dict(
                    ckpt["model_state"]
                )

                optimizer.load_state_dict(
                    ckpt["optimizer_state"]
                )

                scaler.load_state_dict(
                    ckpt["scaler_state"]
                )

                epoch = ckpt["epoch"]
                best_val_loss = ckpt[
                    "best_val_loss"
                ]

                print(
                    f"Retomando epoch {epoch}"
                )
        except:
            print("Checkpoint ignorado")


    # ================= TREINO INFINITO =================

    while True:

        epoch +=1

        # -------- TRAIN --------

        model.train()
        total_loss=0

        loop=tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for x,y in loop:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=USE_AMP
            ):
                out=model(x)
                loss=criterion(out,y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            loop.set_postfix(
                loss=loss.item()
            )

        train_loss = total_loss / len(train_loader)

        print(
            f"Train Loss: {train_loss:.6f}"
        )


        # -------- FINE TUNING --------

        if epoch==2:
            print(
                "Descongelando layer4"
            )

            for name,p in model.named_parameters():
                if (
                    "layer4" in name
                    or "fc" in name
                ):
                    p.requires_grad=True

            optimizer = torch.optim.Adam(
                filter(
                    lambda p:
                    p.requires_grad,
                    model.parameters()
                ),
                lr=LR*0.5
            )


        # -------- VALIDATION --------

        model.eval()

        correct=0
        total=0
        val_loss=0

        with torch.no_grad():

            for x,y in test_loader:

                x=x.to(device)
                y=y.to(device)

                out=model(x)

                loss=criterion(out,y)
                val_loss += loss.item()

                preds=torch.argmax(
                    out,
                    dim=1
                )

                correct += (
                    preds==y
                ).sum().item()

                total += y.size(0)

        avg_val_loss = (
            val_loss / len(test_loader)
        )

        acc = 100*correct/total

        print(
            f"Val Loss: {avg_val_loss:.6f}"
        )

        print(
            f"Val Acc: {acc:.2f}%"
        )


        # -------- SAVE BEST --------

        if avg_val_loss < best_val_loss:
            best_val_loss=avg_val_loss

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                "Melhor modelo salvo"
            )


        # -------- CHECKPOINT --------

        torch.save(
            {
                "model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "scaler_state":scaler.state_dict(),
                "epoch":epoch,
                "best_val_loss":best_val_loss
            },
            CHECKPOINT_PATH
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
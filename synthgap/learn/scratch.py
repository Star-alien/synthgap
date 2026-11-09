"""From-scratch ResNet control (Sec. 3.3, Sec. 6.3).

The question this control answers is *not* "how accurate can synthetic data get"
-- with ~30 images per class from scratch, absolute accuracy is low. It is whether
the from-scratch learner ranks training sets the same way the frozen probe does.
The pipeline reports the Spearman correlation between the two over shared sets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def train_from_scratch(
    train_items: Sequence[tuple[str, int]],
    test_items: Sequence[tuple[str, int]],
    n_classes: int,
    arch: str = "resnet18",
    epochs: int = 60,
    lr: float = 0.1,
    batch_size: int = 64,
    resolution: int = 128,
    weight_decay: float = 5e-4,
    seed: int = 0,
    device: str = "cuda",
    log=None,
) -> dict:
    """Train a randomly initialised ResNet and return real-test accuracy."""
    if not torch_available():  # pragma: no cover
        raise ImportError("the from-scratch control needs torch: pip install -r requirements-gpu.txt")

    import torch
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"

    class Items(Dataset):
        def __init__(self, items, train: bool):
            self.items = list(items)
            from torchvision import transforms

            if train:
                self.tf = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(resolution, scale=(0.6, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )
            else:
                self.tf = transforms.Compose(
                    [
                        transforms.Resize(int(resolution * 1.14)),
                        transforms.CenterCrop(resolution),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]
                )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, label = self.items[index]
            return self.tf(Image.open(path).convert("RGB")), label

    model = getattr(torchvision.models, arch)(num_classes=n_classes).to(device)
    train_loader = DataLoader(Items(train_items, True), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    test_loader = DataLoader(Items(test_items, False), batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if log and (epoch + 1) % 20 == 0:
            log.info(f"    scratch epoch {epoch + 1}/{epochs}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += int(labels.numel())
    return {"accuracy": correct / max(1, total), "n_train": len(train_items), "epochs": epochs, "arch": arch}

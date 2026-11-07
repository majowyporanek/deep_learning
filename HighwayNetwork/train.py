"""Training loop for the Highway Network on CIFAR10.

This is the standard PyTorch loop (Lecture 2, p.19):
    forward -> loss -> backward -> step -> zero grad
plus per-epoch validation and a saved checkpoint of the best model.
"""

import torch
import torch.nn as nn


@torch.no_grad()                      # no gradients needed when just measuring
def evaluate(model, loader, device, criterion):
    """Return (average loss, accuracy) over a loader. Used for val and test."""
    model.eval()                      # eval mode (affects dropout/batchnorm if present)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()   # top-1 predictions
        total += x.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, loader, device, criterion, optimizer):
    """Run one pass over the training data. Returns (avg loss, accuracy)."""
    model.train()                     # train mode
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)             # 1. forward
        loss = criterion(logits, y)   # 2. loss  (softmax + cross-entropy inside)

        optimizer.zero_grad()         # 3. clear old gradients (easy to forget!)
        loss.backward()               # 4. backward  -> fills .grad on every weight
        optimizer.step()              # 5. step      -> optimizer nudges the weights

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, device, epochs=40, lr=0.1,
        weight_decay=5e-4, momentum=0.9, ckpt_path="best_model.pt"):
    """Full training run. Trains for `epochs`, validates each epoch, and saves
    the best model (by val accuracy) to `ckpt_path`. Returns a history dict."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    # Cosine schedule: lr smoothly decays to ~0 over training -> better final accuracy.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        # Save the best model so a Colab disconnect never loses your best run.
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)

        print(f"epoch {epoch:3d}/{epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}"
              f"{'  <- best' if va_acc == best_val_acc else ''}")

    print(f"\nbest val accuracy: {best_val_acc:.3f}  (saved to {ckpt_path})")
    return history


if __name__ == "__main__":
    # Smoke test: 1 epoch on a TINY slice, CPU. Just proves the loop runs and
    # the loss is a sane number (~ln(10) = 2.30 at the start).
    from torch.utils.data import DataLoader, Subset
    from model import HighwayConvNet
    from data import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, _, _ = get_dataloaders(batch_size=128, num_workers=0)
    # shrink to a few batches so the CPU smoke test is fast
    tiny_train = DataLoader(Subset(train_loader.dataset, list(range(256))), batch_size=128)
    tiny_val = DataLoader(Subset(val_loader.dataset, list(range(256))), batch_size=128)

    net = HighwayConvNet(mode="highway", blocks_per_stage=3)
    fit(net, tiny_train, tiny_val, device, epochs=1, ckpt_path="smoke_model.pt")
    print("OK: training loop runs.")

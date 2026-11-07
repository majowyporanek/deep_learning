"""Small helpers: reproducible seeding and plotting training curves.

The plots are what goes into your report (Lecture 1, pp.19-22): reading the
train vs val curves tells you overfitting vs underfitting.
"""

import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Make a run reproducible: same seed -> same split, same init, same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """Number of trainable weights -- handy for the plain-vs-highway comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(history, title="Highway Network", save_path=None):
    """Plot train/val loss and accuracy side by side for a single run."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"saved plot to {save_path}")
    return fig


def plot_comparison(histories, metric="val_acc", title="Plain vs Highway", save_path=None):
    """Overlay one metric from several runs on one axis.

    histories: dict like {"highway-10": hist1, "plain-10": hist2, ...}
    This is the Day-4 figure: plain degrades with depth, highway stays stable.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, hist in histories.items():
        epochs = range(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], label=label)
    ax.set_xlabel("epoch"); ax.set_ylabel(metric)
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"saved plot to {save_path}")
    return fig


if __name__ == "__main__":
    # Smoke test with fake history (no training needed).
    set_seed(42)
    fake = {
        "train_loss": [2.3, 1.8, 1.4, 1.1, 0.9],
        "val_loss":   [2.3, 1.9, 1.6, 1.5, 1.5],
        "train_acc":  [0.1, 0.35, 0.5, 0.62, 0.70],
        "val_acc":    [0.1, 0.33, 0.45, 0.50, 0.52],
    }
    plot_history(fake, save_path="curves_demo.png")
    plot_comparison({"highway": fake, "plain": fake}, save_path="compare_demo.png")
    print("OK: utils run and plots saved.")

"""CIFAR10 data pipeline: train/val/test split, train-only normalization,
and train-only augmentation.

Key correctness point (data leakage): the normalization mean/std are computed
on the TRAIN split ONLY. If you computed them over the whole dataset, info from
val/test would leak into training. Augmentation (random crop/flip) is applied to
the train split only -- val/test must be evaluated on clean images.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_ROOT = "./data"


def _compute_mean_std(dataset, indices, batch_size=512):
    """Per-channel mean/std over a subset, computed in a streaming way
    (no need to hold all images in memory)."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0
    for x, _ in loader:                       # x: [B, 3, 32, 32], values in [0, 1]
        channel_sum += x.sum(dim=[0, 2, 3])   # sum over batch, height, width -> per channel
        channel_sq_sum += (x ** 2).sum(dim=[0, 2, 3])
        n_pixels += x.shape[0] * x.shape[2] * x.shape[3]
    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean ** 2).sqrt()
    return mean, std


def get_dataloaders(root=CIFAR10_ROOT, batch_size=128, val_size=5000,
                    num_workers=2, seed=42):
    """Returns (train_loader, val_loader, test_loader, (mean, std)).

    50,000 train images are split into train (45,000) + val (5,000).
    The 10,000 test images are kept separate and touched only once, at the end.
    """
    # 1. Base train set with ToTensor only -- used to compute stats and to split.
    base = datasets.CIFAR10(root, train=True, download=True,
                            transform=transforms.ToTensor())

    # 2. Reproducible split into train / val indices.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(base), generator=g)
    val_idx = perm[:val_size].tolist()
    train_idx = perm[val_size:].tolist()

    # 3. Normalization stats on the TRAIN split ONLY (this is the no-leakage step).
    mean, std = _compute_mean_std(base, train_idx)

    # 4. Transforms. Train gets augmentation; val/test get clean + normalize only.
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # shift the image a little
        transforms.RandomHorizontalFlip(),      # mirror left-right
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 5. Two views of the SAME train data: one augmented (train), one clean (val).
    train_full = datasets.CIFAR10(root, train=True, download=False, transform=train_tf)
    val_full = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader, (mean, std)


if __name__ == "__main__":
    train_loader, val_loader, test_loader, (mean, std) = get_dataloaders(num_workers=0)
    print(f"train batches: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}")
    print(f"train-only mean: {mean.tolist()}")
    print(f"train-only std:  {std.tolist()}")
    x, y = next(iter(train_loader))
    print(f"one train batch: x {tuple(x.shape)}  y {tuple(y.shape)}")
    print(f"labels range: {y.min().item()}..{y.max().item()}  (expect 0..9)")

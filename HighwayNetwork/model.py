"""Highway Network for CIFAR10 (Srivastava et al., 2015, arXiv:1505.00387).

A highway block mixes a transformed signal H(x) with the untouched input x,
controlled by a learned gate T(x):

    y = H(x) * T(x) + x * (1 - T(x))

- T near 0  -> y ~= x        (carry: pass the input through unchanged)
- T near 1  -> y ~= H(x)     (transform: behave like a normal layer)

The gate's bias starts negative so every block begins in "carry" mode, giving
the gradient a clean highway back through a very deep stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayBlock(nn.Module):
    """One convolutional highway block. Input and output have the SAME shape,
    so the carry path (x * (1 - T)) is valid. Channels stay constant; padding=1
    keeps the 32x32 / 16x16 / 8x8 spatial size unchanged."""

    def __init__(self, channels, mode="highway", gate_bias=-2.0):
        super().__init__()
        self.mode = mode

        # H: the "processing" transform -> conv + ReLU (applied in forward)
        self.H = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        if mode == "highway":
            # T: the gate -> same conv shape, but squashed to 0..1 by sigmoid
            self.T = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            # Start the gate closed (T ~= 0) so the block carries x at first.
            nn.init.constant_(self.T.bias, gate_bias)

    def forward(self, x):
        h = F.relu(self.H(x))  # processed version of x
        if self.mode == "plain":
            return h  # plain layer: no gate, no carry
        t = torch.sigmoid(self.T(x))  # gate values in (0, 1), one per feature
        return h * t + x * (1.0 - t)  # weighted mix: processed vs original


class HighwayConvNet(nn.Module):
    """CIFAR10 classifier: 3 stages of highway blocks at 32/64/128 channels,
    with plain conv layers in between to change channels and halve the spatial
    size. `mode` toggles highway vs plain."""

    def __init__(
        self, mode="highway", blocks_per_stage=3, gate_bias=-2.0, num_classes=10
    ):
        super().__init__()

        # Entry: a PLAIN conv that sets the channel count (3 -> 32). Size stays 32x32.
        self.entry = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Stage 1: highway blocks at 32 channels, 32x32
        self.stage1 = self._make_stage(32, blocks_per_stage, mode, gate_bias)
        # Downsample (plain): 32 -> 64 channels, 32x32 -> 16x16 (stride 2)
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Stage 2: highway blocks at 64 channels, 16x16
        self.stage2 = self._make_stage(64, blocks_per_stage, mode, gate_bias)
        # Downsample (plain): 64 -> 128 channels, 16x16 -> 8x8
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Stage 3: highway blocks at 128 channels, 8x8
        self.stage3 = self._make_stage(128, blocks_per_stage, mode, gate_bias)

        # Head: squeeze each 8x8 feature map to one number, then classify.
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, 128, 1, 1]
        self.fc = nn.Linear(128, num_classes)  # -> [B, 10] logits

    def _make_stage(self, channels, n, mode, gate_bias):
        return nn.Sequential(
            *[HighwayBlock(channels, mode, gate_bias) for _ in range(n)]
        )

    def forward(self, x):
        x = F.relu(self.entry(x))  # [B, 32, 32, 32]
        x = self.stage1(x)  # [B, 32, 32, 32]
        x = F.relu(self.down1(x))  # [B, 64, 16, 16]
        x = self.stage2(x)  # [B, 64, 16, 16]
        x = F.relu(self.down2(x))  # [B, 128, 8, 8]
        x = self.stage3(x)  # [B, 128, 8, 8]
        x = self.pool(x)  # [B, 128, 1, 1]
        x = torch.flatten(x, 1)  # [B, 128]
        return self.fc(x)  # [B, 10]  (raw logits; softmax lives in the loss)


if __name__ == "__main__":
    # Shape sanity check (runs on CPU, no GPU needed).
    for mode in ("highway", "plain"):
        net = HighwayConvNet(mode=mode, blocks_per_stage=3)
        dummy = torch.randn(4, 3, 32, 32)  # a fake batch of 4 CIFAR10 images
        out = net(dummy)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"{mode:8s} -> output {tuple(out.shape)} | params {n_params:,}")
    assert out.shape == (4, 10)
    print("OK: forward pass produces [4, 10] logits.")

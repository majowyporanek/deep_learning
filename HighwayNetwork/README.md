# Highway Network on CIFAR10

Assignment #1 — implement a convolutional Highway Network (Srivastava et al., 2015,
https://arxiv.org/abs/1505.00387) and train it on CIFAR10 in PyTorch (Google Colab, GPU).

## The core idea (one block)

A highway block mixes a transformed signal with the untouched input, controlled by a
learned gate:

    y = H(x) · T(x) + x · (1 − T(x))

- `H(x)` — conv + ReLU (the "processing"), the transform
- `T(x)` — conv + sigmoid (the gate, 0..1), with its **bias initialised negative** so
  blocks start in "carry" mode (y ≈ x) and gradient has a clean highway back
- `x · (1 − T)` — the carry path (the "skip road")

For the carry path to work, `x`, `H(x)`, `T(x)` must have the **same shape** → use
conv with `padding=1` and keep channels constant inside a block. Change size only in
dedicated **plain** (non-highway) layers.

## File structure

| File | Holds | Build order |
|---|---|---|
| `model.py` | `HighwayBlock`, `HighwayConvNet` (supports plain vs highway) | 1 |
| `data.py` | CIFAR10 train/val/test loaders, train-only normalization, augmentation | 2 |
| `train.py` | `train_one_epoch`, `evaluate`, `fit` loop, metric logging | 3 |
| `utils.py` | plotting curves, seeding, accuracy helper | 4 |
| `highway_colab.ipynb` | ties it together, runs on GPU, experiments | 5 |

Design note: `HighwayConvNet` takes a `mode="highway" | "plain"` flag so Day-4's
plain-vs-highway comparison uses the *same* architecture (fair comparison, the paper's point).

## Planned architecture (CIFAR10, 32×32×3 → 10 classes)

    input 3×32×32
    │
    ├─ plain conv  3→32, ReLU                 (entry; sets channel count)
    ├─ K highway blocks @ 32ch, 32×32          (depth lives here)
    ├─ plain conv 32→64 + downsample → 16×16   (size change = plain layer)
    ├─ K highway blocks @ 64ch, 16×16
    ├─ plain conv 64→128 + downsample → 8×8
    ├─ K highway blocks @ 128ch, 8×8
    ├─ global average pool → 128-vector
    └─ linear 128→10  (logits; softmax lives inside CrossEntropyLoss)

Start modest (K≈3 → ~10–12 highway layers). Depth is a knob for Day-4 experiments:
the whole point is to show plain degrades with depth while highway doesn't.

## Build order & checkpoints

1. **`model.py`** → `HighwayBlock` forward gives same shape in = out; full net on a
   random `[4,3,32,32]` tensor returns `[4,10]`. ✅
2. **`data.py`** → one train batch is `[B,3,32,32]`, labels in 0..9; mean/std computed
   on **train split only**. ✅
3. **`train.py`** → one epoch runs; loss is a sane number (~ln(10)≈2.3 at start). ✅
4. **`utils.py`** → can plot train/val loss + accuracy curves.
5. **Colab** → enable GPU, train for real, loss drops, val accuracy rises. ✅

## Hyperparameters (starting point)

- optimizer: SGD(momentum=0.9, weight_decay=5e-4) or Adam(lr=1e-3)
- lr: 0.1 (SGD) with decay, or 1e-3 (Adam)
- batch size: 128
- epochs: ~30–50 (watch val curve; stop when it plateaus)
- transform-gate bias init: -2 (paper used -1..-4)
- augmentation (train only): RandomCrop(32, padding=4), RandomHorizontalFlip

## Experiments (Day 4)

- plain vs highway at increasing depth (K = 3, 6, 10 blocks per stage) → reproduce the
  paper's finding (Figure 1): plain gets worse with depth, highway stays stable.
- report test accuracy (touch test set ONCE), plot curves.

## For the oral defense — be ready to explain

- what the carry gate does and why it saves deep nets (gradient highway)
- why the gate uses sigmoid and its bias starts negative
- why softmax + CrossEntropyLoss for 10 classes
- plain vs highway result and what it shows

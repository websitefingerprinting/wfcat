# WFCAT

Reference implementation for **WFCAT: Augmenting Website Fingerprinting With Channel-Wise Attention on Timing Features**.

This release keeps the experiment interface simple: choose a backbone with `--model` and an input representation with `--feature-type`. The default configuration is WFCAT with IAT histogram features.

All commands below assume the current working directory is the WFCAT project root.

## Layout

```text
.
  README.md
  requirements.txt
  pyproject.toml
  scripts/
  src/
    train.py
    trainer.py
    data.py
    features.py
    metrics.py
    model_registry.py
    utils.py
    models/
```

## Installation

Using `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

## Data Format

Trace files are tab-separated `.cell` files with two columns:

```text
timestamp direction_or_size
```

Monitored traces use names like:

```text
0-0.cell
0-1.cell
1-0.cell
...
```

Open-world unmonitored traces use names like:

```text
0.cell
1.cell
...
```

## Main Commands

Closed-world WFCAT:

```bash
python src/train.py \
  --data-path /path/to/cell_files \
  --model wfcat \
  --feature-type iat \
  --mon-classes 100 \
  --mon-inst 100 \
  --epochs 50
```

Open-world WFCAT:

```bash
python src/train.py \
  --data-path /path/to/cell_files \
  --model wfcat \
  --feature-type iat \
  --open-world \
  --unmon-inst 10000
```

One-fold smoke test:

```bash
python src/train.py \
  --data-path /path/to/cell_files \
  --model wfcat \
  --feature-type iat \
  --one-fold \
  --epochs 1 \
  --nosave
```

## Default WFCAT Parameters

```text
model: wfcat
feature-type: iat
seq-length: 1800
time-window: 0.044
iat-bins: 9
num-kernels: 4
epochs: 50
batch-size: 64
lr0: 0.001
weight-decay: 0.0005
```

## Supported Backbones and Features

Recommended WFCAT configuration:

```bash
--model wfcat --feature-type iat
```

Baseline configurations:

```bash
--model rf --feature-type tam
--model df --feature-type direction
--model df --feature-type directional_timing
--model varcnn --feature-type directional_timing
--model ares --feature-type direction
--model tmwf --feature-type direction --seq-length 30720
```

## Scripts

Local scripts:

```bash
bash scripts/train_closed.sh /path/to/cell_files --one-fold --epochs 1 --nosave
bash scripts/train_open.sh /path/to/cell_files --one-fold --epochs 1 --nosave
bash scripts/train_slow_fast.sh /path/to/cell_files --epochs 50
bash scripts/train_all_models.sh /path/to/cell_files --one-fold --nosave
```

SLURM template:

```bash
DATA_PATH=/path/to/cell_files sbatch scripts/train_slurm.sh
```

Override defaults through environment variables:

```bash
DATA_PATH=/path/to/cell_files MODEL=wfcat FEATURE_TYPE=iat EPOCHS=50 sbatch scripts/train_slurm.sh
```

## Output

Closed-world runs print:

```text
tp fp p n
```

Open-world runs print precision and recall pairs for confidence thresholds from 0.01 to 0.99.

## Citation

If you use this code, please cite the WFCAT paper.

```bibtex
@article{GongCLGWC26,
  author       = {Jiajun Gong and
                  Wei Cai and
                  Siyuan Liang and
                  Zhong Guan and
                  Tao Wang and
                  Ee{-}Chien Chang},
  title        = {{WFCAT:} Augmenting Website Fingerprinting With Channel-Wise Attention
                  on Timing Features},
  journal      = {{IEEE} Transactions on Dependable and Secure Computing},
  volume       = {23},
  number       = {1},
  pages        = {149--163},
  year         = {2026},
  url          = {https://doi.org/10.1109/TDSC.2025.3605197},
  doi          = {10.1109/TDSC.2025.3605197},
}
```

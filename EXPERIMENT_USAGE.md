# Unified Training Script Usage Guide

## Overview

`train_experiment.py` is a unified script that replaces all the individual experiment scripts. Instead of creating or selecting different script files, you now run the same script with different command-line arguments.

## Quick Start

### Basic Independent Experiment
```bash
python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM --test-dataset CREMA --experiment-type indep
```

### With Frozen Encoder
```bash
python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM --test-dataset CREMA --experiment-type indep --freeze
```

### Cross-Corpus Experiment
```bash
python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM,RAVDESS,SAVEE,TESS --experiment-type cross
```

## Command-Line Arguments

### Required Arguments
- `--train-datasets`: Comma-separated list of datasets for training
  - Available: CAFE, CREMA, EMOFILM, RAVDESS, SAVEE, TESS
  - Example: `--train-datasets CAFE,CREMA,EMOFILM`

### Experiment Configuration
- `--experiment-type`: Type of experiment
  - Choices: `indep` (independent) or `cross` (cross-corpus)
  - Default: `indep`
  
- `--test-dataset`: Dataset to use for testing
  - If not provided, uses test splits from training datasets
  - Example: `--test-dataset CREMA`

- `--test-split-source`: Dataset to split for test/val (IJNS-style experiments)
  - Example: `--test-split-source RAVDESS`
  
- `--val-split-frac`: Fraction for validation split
  - Default: `0.5`

### Model Configuration
- `--model`: Model checkpoint to use
  - Available: HuB-LL, HuB-XL, HuB-ER, HuB-SID, W2V2-base, W2V2-large, W2V2-robust, DistilHubert
  - Default: `HuB-LL`
  
- `--checkpoint-path`: Custom checkpoint path (overrides --model)
  - Example: `--checkpoint-path facebook/custom-model`

### Training Configuration
- `--freeze`: Freeze feature encoder (flag, no value needed)
  - Use `--freeze` to enable, omit to disable
  - Default: False
  
- `--epochs`: Number of training epochs
  - Default: `20`
  
- `--rondas`: Number of rounds/runs
  - Default: `3`
  
- `--lr`: Learning rate
  - Default: `1e-5`
  
- `--weight-decay`: Weight decay
  - Default: `0.005`
  
- `--use-attention-mask`: Use attention mask in training
  - Default: True (flag is on by default)

### Paths
- `--root-folder`: Root folder for datasets and outputs
  - Default: `/home/x002/PROJECT`
  
- `--dump-folder`: Output folder
  - Default: `{root_folder}/dump/`

### Advanced Options
- `--epoch-ini`: Initial epoch number (for resuming)
  - Default: `1`
  
- `--best-acc`: Initial best accuracy (for resuming)
  - Default: `0.0`
  
- `--experiment-name`: Custom experiment name for output folder

## Common Experiment Patterns

### Pattern 1: Independent SER (like IndependentSER.indeps.2025Q1.HuB-LL.4.cr.uf.py)
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM,RAVDESS \
  --test-dataset CREMA \
  --experiment-type indep \
  --model HuB-LL \
  --freeze  # Add this for .fr files, omit for .uf files
```

### Pattern 2: Cross-Corpus (like IndependentSER.crossc.2025Q1.HuB-LL.py)
```bash
python train_experiment.py \
  --train-datasets CREMA,SAVEE,RAVDESS,TESS \
  --experiment-type cross \
  --model HuB-LL
```

### Pattern 3: IJNS Matrix Experiments (like IJNS_001_matrix.2025Q2.HuB-LL.cr.uf.py)
```bash
# Train on 5 datasets, test on CREMA
python train_experiment.py \
  --train-datasets CAFE,EMOFILM,RAVDESS,SAVEE,TESS \
  --test-dataset CREMA \
  --experiment-type indep \
  --model HuB-LL \
  --freeze  # Add for .fr, omit for .uf
```

### Pattern 4: IJNS Split Experiment (like IJNS_003_matrix.2025Q2.HuB-LL.rv.cr.fr.py)
```bash
# Train on RAVDESS, split RAVDESS for test/val
python train_experiment.py \
  --train-datasets RAVDESS \
  --test-split-source RAVDESS \
  --experiment-type indep \
  --model HuB-LL \
  --freeze
```

### Pattern 5: Custom Configuration
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA \
  --test-dataset CREMA \
  --experiment-type indep \
  --model HuB-ER \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --rondas 5
```

## Mapping Old Scripts to New Command

| Old Script Pattern | New Command Pattern |
|-------------------|---------------------|
| `IndependentSER.indeps.*.cr.uf.py` | `--train-datasets CAFE,CREMA,EMOFILM,RAVDESS --test-dataset CREMA --experiment-type indep` |
| `IndependentSER.indeps.*.cr.fr.py` | Same as above + `--freeze` |
| `IndependentSER.crossc.*.py` | `--experiment-type cross` |
| `IJNS_*_matrix.*.cr.uf.py` | `--train-datasets CAFE,EMOFILM,RAVDESS,SAVEE,TESS --test-dataset CREMA --experiment-type indep` |
| `IJNS_*_matrix.*.cr.fr.py` | Same as above + `--freeze` |
| `IJNS_003_matrix.*.rv.cr.*.py` | `--train-datasets RAVDESS --test-split-source RAVDESS --experiment-type indep` |

### Using MSP Podcast Dataset

After preprocessing MSP Podcast (see `MSP_PODCAST_PREPROCESSING.md`):

```bash
# Train with MSP Podcast only
python train_experiment.py --train-datasets MSP_PODCAST --test-dataset MSP_PODCAST --experiment-type indep

# Include MSP Podcast in multi-dataset training
python train_experiment.py --train-datasets CAFE,CREMA,MSP_PODCAST --test-dataset CREMA --experiment-type indep
```

## Output

The script creates an output folder with:
- Model checkpoints
- Training logs
- Loss and accuracy plots
- Confusion matrices (when using attention mask)
- Training state files for resuming

Output folder structure: `{dump_folder}/experim_{timestamp}.{job_id}.{experiment_type}/`

## Notes

- All scripts use a fixed random seed (19730309) for reproducibility
- The script automatically handles RAVDESS dataset preprocessing (removes gender, converts calm to neutral)
- Default batch size is 2 for training and 1 for validation/testing
- The script supports both `loop_with_att_mask` (default) and `loop` training functions


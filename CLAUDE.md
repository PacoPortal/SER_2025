# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Speech Emotion Recognition (SER) research codebase for training and evaluating emotion recognition models on multiple audio datasets. The project uses transformer-based models (HuBERT, Wav2Vec2) for classifying speech into 7 emotion categories: neutral, happy, sad, angry, fear, disgust, and surprise.

## Core Architecture

### Common Code Module (`IndependentSER_common_code.py`)

Contains all shared functionality used across experiments:

- **Dataset Classes**: `SpeechDataset` (loads audio files using librosa at 16kHz sampling rate)
- **Training Functions**:
  - `train_the_model_att_mask()` - Training with attention masks (default/recommended)
  - `train_the_model()` - Training without attention masks (legacy)
- **Testing Functions**:
  - `testing_att_mask()` - Evaluation with attention masks, returns (accuracy, loss, record)
  - `testing()` - Evaluation without attention masks, returns (accuracy, loss)
- **Loop Functions**:
  - `loop_with_att_mask()` - Main training loop with attention masks (use this by default)
  - `loop()` - Training loop without attention masks (legacy)
- **Checkpoint Management**: `save_checkpoint()`, `load_checkpoint()`
- **Visualization**: `plot_loss_acc()`, `display_confusion_matrix()`, `data_distribution()`
- **Data Splitting**: `data_split()` - Stratified split by emotion class using random seed 19730309

### Unified Training Script (`train_experiment.py`)

**This is the primary script for running experiments.** It replaces all individual experiment files through command-line configuration.

#### Available Datasets
- CAFE, CREMA, EMOFILM, RAVDESS, SAVEE, TESS, MSP_PODCAST

#### Available Models
- HuB-LL: `facebook/hubert-large-ll60k` (default)
- HuB-XL: `facebook/hubert-xlarge-ll60k`
- HuB-ER: `superb/hubert-large-superb-er`
- HuB-SID: `superb/hubert-large-superb-sid`
- W2V2-base: `facebook/wav2vec2-base-960h`
- W2V2-large: `facebook/wav2vec2-large`
- W2V2-robust: `facebook/wav2vec2-large-robust-ft-libri-960h`
- DistilHubert: `ntu-spml/distilhubert`

## Running Experiments

### Basic Commands

Independent experiment (train and test on specified datasets):
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM \
  --test-dataset CREMA \
  --experiment-type indep
```

With frozen encoder (faster, often used in research experiments):
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM \
  --test-dataset CREMA \
  --experiment-type indep \
  --freeze
```

Cross-corpus experiment (trains on multiple datasets, tests across all):
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM,RAVDESS,SAVEE,TESS \
  --experiment-type cross
```

IJNS-style matrix experiment (leave-one-dataset-out):
```bash
# Train on 5 datasets, test on CREMA
python train_experiment.py \
  --train-datasets CAFE,EMOFILM,RAVDESS,SAVEE,TESS \
  --test-dataset CREMA \
  --experiment-type indep \
  --freeze
```

Custom hyperparameters:
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA \
  --test-dataset CREMA \
  --experiment-type indep \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --rondas 5
```

### Key Command-Line Arguments

- `--train-datasets`: Comma-separated list (e.g., `CAFE,CREMA,EMOFILM`)
- `--test-dataset`: Single dataset for testing
- `--test-split-source`: Split this dataset for test/val instead of using separate test set
- `--experiment-type`: `indep` (independent) or `cross` (cross-corpus)
- `--model`: Model name (default: `HuB-LL`)
- `--freeze`: Flag to freeze feature encoder (no value needed)
- `--epochs`: Number of epochs (default: 20)
- `--rondas`: Number of training rounds/runs (default: 3)
- `--lr`: Learning rate (default: 1e-5)
- `--weight-decay`: Weight decay (default: 0.005)
- `--root-folder`: Root folder for datasets/outputs (default: `/home/x002/PROJECT`)

## Dataset Preprocessing

### MSP Podcast Dataset

To add the MSP Podcast dataset to the project:

```bash
python msp_podcast_preprocess.py \
  --annotations /path/to/msp_podcast/annotations.csv \
  --audio-dir /path/to/msp_podcast/audio/
```

This creates `D.MSP_PODCAST_train_DF.p` and `D.MSP_PODCAST_test_DF.p` in the output directory.

After preprocessing, train with MSP Podcast:
```bash
python train_experiment.py \
  --train-datasets MSP_PODCAST \
  --test-dataset MSP_PODCAST \
  --experiment-type indep
```

Or combine with other datasets:
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,MSP_PODCAST \
  --test-dataset CREMA \
  --experiment-type indep
```

See `MSP_PODCAST_PREPROCESSING.md` for detailed preprocessing instructions.

## File Structure and Naming Conventions

### Legacy Experiment Scripts (Being Replaced)

The repository contains many individual experiment scripts with naming patterns like:
- `IndependentSER.indeps.2025Q1.HuB-LL.4.cr.uf.py` - Independent SER, CREMA test, unfrozen
- `IndependentSER.indeps.2025Q1.HuB-LL.4.cr.fr.py` - Same but frozen encoder
- `IndependentSER.crossc.2025Q1.HuB-LL.py` - Cross-corpus experiment
- `IJNS_00X_matrix.2025Q2.HuB-LL.cr.uf.py` - IJNS matrix experiments

**These are legacy files.** Use `train_experiment.py` instead for all new experiments.

### Dataset Files

Expected location: `/home/x002/PROJECT/DFs/`

Format: `D.{DATASET}_train_DF.p` and `D.{DATASET}_test_DF.p`

Each DataFrame contains:
- `path`: Full path to audio file
- `labels`: Numerical emotion label (0-6)
- `emotions`: String emotion name

### Output Structure

Training outputs go to: `{root_folder}/dump/experim_{timestamp}.{job_id}.{experiment_type}/`

Contains:
- Model checkpoints (`.model` files)
- Training logs (`_log.p` pickle files)
- Loss and accuracy plots (`_TrainingLoss_tight.png`, `_TrainingAccuracy_tight.png`)
- Confusion matrices (`_Test_ConfusionMatrix_tight.png`)
- Training state (`Train_State.p`) for resuming

Model checkpoints are stored separately in: `/home/x002/SCRATCH/models/`

## Important Implementation Details

### Random Seed
All experiments use fixed random seed `19730309` for reproducibility. This is set in:
- Training loops
- Data splitting functions
- PyTorch, NumPy, and Python random modules

### RAVDESS Dataset Preprocessing
RAVDESS requires special handling (automatically done in `train_experiment.py`):
- Remove `gender` column if present
- Replace `calm` emotion with `neutral`
- Convert label 7 to 0 (calm → neutral mapping)

### Batch Sizes
- Training: batch_size=2
- Validation/Testing: batch_size=1

### Feature Extraction
Models use `AutoFeatureExtractor` with:
- `return_attention_mask=True` (default for all training)
- 16kHz sampling rate
- Padding enabled

### Training Defaults
- Learning rate: 1e-5
- Weight decay: 0.005
- Epochs: 20
- Optimizer: AdamW
- Attention masks: Enabled by default (recommended)

### Label Mapping
Standard 7-class emotion mapping:
```python
{
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6
}
```

## Troubleshooting

### CUDA Memory Issues
The code sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to handle memory fragmentation.

### Path Issues
Default paths assume HPC environment with `/home/x002/PROJECT` structure. Override with:
- `--root-folder` for custom root directory
- `--dump-folder` for custom output directory

### Missing Datasets
Ensure pickle files exist in `{root_folder}/DFs/` with correct naming: `D.{DATASET}_train_DF.p` and `D.{DATASET}_test_DF.p`

### Resuming Training
To resume interrupted training:
1. Check `Train_State.p` in the experiment folder for (timestamp, next_epoch, best_accuracy)
2. Use `--epoch-ini` and `--best-acc` arguments with the same experiment folder
3. Checkpoint `model.tmp.model` will be loaded automatically

## Key Differences from Legacy Scripts

1. **Unified interface**: All experiments through one script instead of 50+ separate files
2. **Command-line configuration**: No need to edit Python code for different experiments
3. **Consistent defaults**: All experiments use same attention mask and training settings
4. **Automatic dataset handling**: RAVDESS preprocessing applied automatically
5. **Flexible dataset combinations**: Any combination of datasets supported via CLI

## Working with Common Code

When modifying `IndependentSER_common_code.py`:
- Always use `loop_with_att_mask()` for new experiments (not `loop()`)
- Ensure all functions preserve the random seed (19730309)
- Checkpoint functions save to `DUMP_FOLDER_MODELS` (separate from experiment outputs)
- Testing functions return different formats depending on `verbose` parameter:
  - `verbose=True`: Returns record dict for confusion matrices
  - `verbose=False`: Returns (accuracy, loss) or (accuracy, loss, record)

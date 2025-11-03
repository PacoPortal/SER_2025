# Speech Emotion Recognition (SER) 2025

A research codebase for training and evaluating transformer-based Speech Emotion Recognition models on multiple audio datasets.

## Overview

This project implements Speech Emotion Recognition (SER) using state-of-the-art transformer models (HuBERT, Wav2Vec2) to classify speech audio into 7 emotion categories: neutral, happy, sad, angry, fear, disgust, and surprise.

The codebase supports:
- **Multiple datasets**: CAFE, CREMA, EMOFILM, RAVDESS, SAVEE, TESS, MSP_PODCAST
- **Multiple models**: HuBERT (Large/XLarge), Wav2Vec2 (Base/Large/Robust), DistilHubert
- **Flexible experiment types**: Independent, cross-corpus, and IJNS-style matrix experiments
- **Unified training interface**: Single script with command-line configuration

## Key Features

- **Unified Training Script**: Run any experiment configuration through `train_experiment.py` with command-line arguments
- **Reproducible Research**: Fixed random seeds (19730309) for consistent results
- **Comprehensive Evaluation**: Automatic generation of confusion matrices, loss/accuracy plots, and performance metrics
- **Multi-Dataset Support**: Easy integration of new datasets with standardized preprocessing
- **GPU Acceleration**: CUDA support with memory optimization
- **Checkpoint Management**: Automatic model checkpointing and training resumption

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SER_2025.git
cd SER_2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data directories (adjust paths as needed):
```bash
export ROOT_FOLDER=/path/to/your/project
mkdir -p $ROOT_FOLDER/Datasets
mkdir -p $ROOT_FOLDER/DFs
mkdir -p $ROOT_FOLDER/dump
```

## Quick Start

### Basic Training

Train on CAFE, CREMA, and EMOFILM datasets, test on CREMA:
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM \
  --test-dataset CREMA \
  --experiment-type indep
```

### Cross-Corpus Experiment

Train across multiple datasets:
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM,RAVDESS,SAVEE,TESS \
  --experiment-type cross
```

### Frozen Encoder (Faster Training)

Freeze the feature encoder for faster training:
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM \
  --test-dataset CREMA \
  --experiment-type indep \
  --freeze
```

## Available Datasets

| Dataset | Description | Emotions |
|---------|-------------|----------|
| **CAFE** | Canadian French Emotional Speech | 7 classes |
| **CREMA** | Crowd-sourced Emotional Multimodal Actors | 7 classes |
| **EMOFILM** | Emotional speech from films | 7 classes |
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song | 7 classes (calm→neutral) |
| **SAVEE** | Surrey Audio-Visual Expressed Emotion | 7 classes |
| **TESS** | Toronto Emotional Speech Set | 7 classes |
| **MSP_PODCAST** | MSP Podcast Corpus (400+ hours) | 7 classes |

## Available Models

| Model | Checkpoint | Description |
|-------|------------|-------------|
| **HuB-LL** | `facebook/hubert-large-ll60k` | HuBERT Large (default) |
| **HuB-XL** | `facebook/hubert-xlarge-ll60k` | HuBERT XLarge |
| **HuB-ER** | `superb/hubert-large-superb-er` | HuBERT Large (SUPERB ER) |
| **HuB-SID** | `superb/hubert-large-superb-sid` | HuBERT Large (SUPERB SID) |
| **W2V2-base** | `facebook/wav2vec2-base-960h` | Wav2Vec2 Base |
| **W2V2-large** | `facebook/wav2vec2-large` | Wav2Vec2 Large |
| **W2V2-robust** | `facebook/wav2vec2-large-robust-ft-libri-960h` | Wav2Vec2 Large Robust |
| **DistilHubert** | `ntu-spml/distilhubert` | Distilled HuBERT |

## Dataset Preprocessing

### MSP Podcast Dataset

To add MSP Podcast to your experiments:

```bash
python msp_podcast_preprocess.py \
  --annotations /path/to/msp_podcast/annotations.csv \
  --audio-dir /path/to/msp_podcast/audio/
```

See [MSP_PODCAST_PREPROCESSING.md](MSP_PODCAST_PREPROCESSING.md) for detailed instructions.

### Custom Datasets

To add your own dataset:
1. Create train/test DataFrames with columns: `path`, `labels`, `emotions`
2. Save as pickle files: `D.{DATASET}_train_DF.p` and `D.{DATASET}_test_DF.p`
3. Add dataset mapping to `train_experiment.py`

## Training Configuration

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-datasets` | str | Required | Comma-separated list of training datasets |
| `--test-dataset` | str | None | Dataset for testing |
| `--experiment-type` | str | `indep` | Experiment type: `indep` or `cross` |
| `--model` | str | `HuB-LL` | Model checkpoint to use |
| `--freeze` | flag | False | Freeze feature encoder |
| `--epochs` | int | 20 | Number of training epochs |
| `--rondas` | int | 3 | Number of training rounds/runs |
| `--lr` | float | 1e-5 | Learning rate |
| `--weight-decay` | float | 0.005 | Weight decay |
| `--root-folder` | str | `/home/x002/PROJECT` | Root folder for data |

See [EXPERIMENT_USAGE.md](EXPERIMENT_USAGE.md) for more detailed usage examples.

## Project Structure

```
SER_2025/
├── train_experiment.py              # Main unified training script
├── IndependentSER_common_code.py    # Shared functions and utilities
├── msp_podcast_preprocess.py        # MSP Podcast preprocessing
├── CLAUDE.md                        # AI coding assistant guidance
├── EXPERIMENT_USAGE.md              # Detailed experiment usage guide
├── MSP_PODCAST_PREPROCESSING.md     # MSP Podcast preprocessing guide
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── [legacy experiment scripts]      # Individual experiment files (deprecated)
```

## Output Structure

Training outputs are saved to: `{root_folder}/dump/experim_{timestamp}.{job_id}.{experiment_type}/`

Outputs include:
- **Model checkpoints** (`.model` files)
- **Training logs** (`_log.p` pickle files)
- **Loss and accuracy plots** (PNG files)
- **Confusion matrices** (PNG files)
- **Training state** (`Train_State.p`) for resuming

## Reproducibility

All experiments use:
- **Fixed random seed**: 19730309
- **Consistent preprocessing**: Automatic handling of dataset-specific requirements
- **Deterministic training**: PyTorch, NumPy, and Python random seeds set

## Advanced Usage

### Custom Hyperparameters

```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA \
  --test-dataset CREMA \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --rondas 5
```

### Using Different Models

```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM \
  --test-dataset CREMA \
  --model W2V2-large
```

### IJNS Matrix Experiments

Leave-one-dataset-out evaluation:
```bash
# Train on 5 datasets, test on CREMA
python train_experiment.py \
  --train-datasets CAFE,EMOFILM,RAVDESS,SAVEE,TESS \
  --test-dataset CREMA \
  --experiment-type indep \
  --freeze
```

### Resuming Training

If training is interrupted, resume using the saved state:
```bash
python train_experiment.py \
  --train-datasets CAFE,CREMA \
  --test-dataset CREMA \
  --epoch-ini 11 \
  --best-acc 0.75 \
  --dump-folder /path/to/previous/experiment/
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size (currently fixed at 2 for training, 1 for testing)
- Use a smaller model (e.g., W2V2-base instead of HuB-XL)
- Enable `--freeze` to reduce memory usage

### Missing Dataset Files
Ensure pickle files exist at: `{root_folder}/DFs/D.{DATASET}_train_DF.p`

### Path Issues
Override default paths with `--root-folder` and `--dump-folder` arguments.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ser2025,
  title={Speech Emotion Recognition 2025},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/SER_2025}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- HuggingFace Transformers library
- MSP Podcast, RAVDESS, CREMA-D, and other dataset creators
- Research community for emotion recognition benchmarks

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

# Contributing to SER_2025

Thank you for your interest in contributing to the Speech Emotion Recognition (SER) 2025 project! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include details**:
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - System information (OS, Python version, GPU/CPU)
   - Error messages and stack traces
   - Dataset and model being used

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed
3. **Test your changes**:
   - Ensure existing experiments still run
   - Test with at least one dataset
   - Verify reproducibility (check random seed handling)
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")
5. **Submit a pull request**:
   - Describe what changes were made and why
   - Reference any related issues
   - Ensure CI checks pass (if configured)

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Project Structure

- **Core functionality** belongs in `IndependentSER_common_code.py`
- **Dataset preprocessing** should follow the pattern in `msp_podcast_preprocess.py`
- **Experiment configurations** should use `train_experiment.py` CLI arguments
- **Avoid creating new experiment scripts** - extend `train_experiment.py` instead

### Adding New Datasets

To add support for a new dataset:

1. Create a preprocessing script (e.g., `{dataset}_preprocess.py`)
2. Output pickle files: `D.{DATASET}_train_DF.p` and `D.{DATASET}_test_DF.p`
3. Ensure DataFrames have columns: `path`, `labels`, `emotions`
4. Update `DATASET_MAP` in `train_experiment.py`
5. Add dataset documentation to README.md
6. Create a preprocessing guide (e.g., `{DATASET}_PREPROCESSING.md`)

Required DataFrame format:
```python
{
    'path': ['/path/to/audio1.wav', ...],      # Full paths to audio files
    'labels': [0, 1, 2, ...],                   # Numeric labels (0-6)
    'emotions': ['neutral', 'happy', ...]       # String emotion names
}
```

### Adding New Models

To add support for a new model:

1. Add model checkpoint to `AVAILABLE_CHECKPOINTS` in `train_experiment.py`
2. Test with at least one dataset to ensure compatibility
3. Document model requirements (memory, special preprocessing, etc.)
4. Update README.md with model information

### Testing

Before submitting a PR:

1. **Run a small experiment** to verify functionality:
```bash
python train_experiment.py \
  --train-datasets CAFE \
  --test-dataset CAFE \
  --epochs 2 \
  --rondas 1
```

2. **Check reproducibility**: Run same experiment twice, verify consistent results

3. **Test with different configurations**:
   - Frozen vs. unfrozen encoder
   - Different datasets
   - Different models (if applicable)

### Reproducibility

**CRITICAL**: Maintain reproducibility by:
- Using fixed random seed (19730309)
- Setting seeds for PyTorch, NumPy, and Python random
- Avoiding non-deterministic operations
- Documenting any sources of randomness

## What to Contribute

We welcome contributions in these areas:

### High Priority

- **New dataset integrations** with preprocessing scripts
- **Bug fixes** for existing functionality
- **Performance optimizations** (training speed, memory usage)
- **Documentation improvements** (README, guides, code comments)
- **Testing infrastructure** (unit tests, integration tests)

### Medium Priority

- **New model support** (other transformer architectures)
- **Visualization improvements** (better plots, tensorboard integration)
- **Experiment tracking** (W&B, MLflow integration)
- **Data augmentation** techniques
- **Mixed precision training** support

### Nice to Have

- **Web interface** for running experiments
- **Automated hyperparameter tuning**
- **Multi-GPU training** support
- **Streaming dataset** support for large datasets
- **Export to ONNX** or other formats

## Code Review Process

1. Maintainers will review your PR within 1-2 weeks
2. Feedback will be provided as review comments
3. Address review comments by pushing new commits
4. Once approved, maintainers will merge your PR

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Acknowledge contributions
- Follow the code of conduct

## Questions?

If you have questions about contributing:

1. Check the [README.md](README.md) and [CLAUDE.md](CLAUDE.md)
2. Search existing issues and discussions
3. Create a new issue with the "question" label
4. Contact the maintainers

## Recognition

Contributors will be acknowledged in:
- README.md (Contributors section)
- Release notes
- Citation file (for significant contributions)

Thank you for helping improve SER_2025!

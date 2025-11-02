# MSP Podcast Dataset Preprocessing Guide

## Overview

This guide explains how to preprocess the MSP Podcast dataset to make it compatible with the SER (Speech Emotion Recognition) training pipeline used in this project.

## MSP Podcast Dataset Information

The MSP Podcast corpus is a large-scale, naturalistic emotional speech dataset containing:
- **Over 400 hours** of audio from podcast recordings
- **Categorical emotion labels**: neutral, happy, sad, angry, fear, disgust, surprise
- **Dimensional attributes**: valence, arousal, dominance
- **Speaker-independent partitions**: Train, Development, Test1, Test2, Test3
- **Human transcriptions** for all segments

**Dataset Website**: https://lab-msp.com/MSP/MSP-Podcast.html

## Prerequisites

1. **Download the MSP Podcast dataset** (requires academic license agreement)
2. **Install required Python packages**:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Dataset Structure

The MSP Podcast dataset typically comes with:
- Audio files (WAV format) organized in directories
- Annotation CSV files containing emotion labels and metadata
- Optional: Partition information (train/test splits)

Expected structure:
```
msp_podcast_dataset/
├── audio/              # Audio files (WAV format)
│   ├── segment_001.wav
│   ├── segment_002.wav
│   └── ...
├── annotations.csv     # Main annotation file
├── labels_consensus.csv  # Alternative annotation file (if available)
└── ...
```

## Usage

### Basic Usage

```bash
python msp_podcast_preprocess.py \
  --annotations /path/to/msp_podcast/annotations.csv \
  --audio-dir /path/to/msp_podcast/audio/
```

### With Multiple Audio Directories

If audio files are organized in multiple directories:

```bash
python msp_podcast_preprocess.py \
  --annotations /path/to/msp_podcast/annotations.csv \
  --audio-dir /path/to/msp_podcast/audio_part1/ \
  --audio-dir /path/to/msp_podcast/audio_part2/
```

### Custom Output Directory

```bash
python msp_podcast_preprocess.py \
  --annotations /path/to/msp_podcast/annotations.csv \
  --audio-dir /path/to/msp_podcast/audio/ \
  --output-dir /custom/output/path/
```

### With Custom Test Ratio

If your annotation file doesn't include partition information:

```bash
python msp_podcast_preprocess.py \
  --annotations /path/to/msp_podcast/annotations.csv \
  --audio-dir /path/to/msp_podcast/audio/ \
  --test-ratio 0.15
```

## Annotation CSV Format

The preprocessing script is flexible and can handle various CSV column naming conventions:

### Required Columns

1. **File Identifier**: One of these column names
   - `file_name`, `file`, `filename`
   - `segment_id`, `id`, `utterance_id`, `name`

2. **Emotion Label**: One of these column names
   - `emotion`, `Emotion`, `emotion_label`, `EmotionLabel`
   - `categorical`, `Categorical`, `emotion_categorical`
   - `consensus`, `Consensus`, `label`, `Label`

### Optional Columns

- **Partition/Split**: `Split`, `split`, `Partition`, `partition`, `Set`, `set`, `usage`, `Usage`

### Example CSV Format

```csv
file_name,emotion,Split,valence,arousal,dominance
segment_001.wav,happy,Train,0.75,0.65,0.70
segment_002.wav,neutral,Test,0.50,0.30,0.50
segment_003.wav,angry,Dev,0.20,0.85,0.60
...
```

## Emotion Label Mapping

The script automatically maps MSP Podcast emotion labels to the standard 7-class format:

| MSP Podcast Label | Standard Label | ID |
|------------------|----------------|----|
| neutral, calm | neutral | 0 |
| happy, happiness, joy, excitement | happy | 1 |
| sad, sadness | sad | 2 |
| angry, anger | angry | 3 |
| fear, fearful | fear | 4 |
| disgust, disgusted, contempt | disgust | 5 |
| surprise, surprised | surprise | 6 |

Unknown or unrecognized labels are mapped to `neutral` with a warning.

## Output

The script generates two pickle files in the output directory (default: `/home/x002/PROJECT/DFs/`):

1. **`D.MSP_PODCAST_train_DF.p`**: Training set DataFrame
2. **`D.MSP_PODCAST_test_DF.p`**: Test set DataFrame

Each DataFrame contains:
- `path`: Full path to the audio file
- `labels`: Numerical emotion label (0-6)
- `emotions`: String emotion name (neutral, happy, sad, angry, fear, disgust, surprise)

### Output Format Example

```python
import pickle
import pandas as pd

# Load the preprocessed data
train_df = pickle.load(open('D.MSP_PODCAST_train_DF.p', 'rb'))
print(train_df.columns)  # ['path', 'labels', 'emotions']
print(train_df.head())
```

## Using Preprocessed Data in Training

Once preprocessed, you can use MSP Podcast in training experiments:

```bash
# Train with MSP Podcast dataset
python train_experiment.py \
  --train-datasets MSP_PODCAST \
  --test-dataset MSP_PODCAST \
  --experiment-type indep \
  --model HuB-LL

# Train with multiple datasets including MSP Podcast
python train_experiment.py \
  --train-datasets CAFE,CREMA,EMOFILM,MSP_PODCAST \
  --test-dataset CREMA \
  --experiment-type indep
```

## Troubleshooting

### Issue: "Could not find file identifier column"

**Solution**: Check your CSV file and ensure it has a column matching one of the expected names. You can modify the script to add your column name to the search list.

### Issue: "Could not find emotion label column"

**Solution**: Ensure your CSV has an emotion column. If using dimensional attributes, you may need to map them to categorical emotions first.

### Issue: Audio files not found

**Solution**: 
- Verify the audio directory path is correct
- Check if audio files have different extensions (.wav, .WAV, .flac, etc.)
- The script searches recursively, but ensure file IDs in CSV match actual filenames (excluding extension)

### Issue: Missing many audio files

**Solution**: 
- Check if file IDs in CSV include extensions (script removes extensions automatically)
- Verify audio files are in the specified directories
- Check for case sensitivity issues (the script tries multiple variations)

## Notes

- The script preserves speaker independence when partition information is available
- Random splits maintain class distribution using stratified sampling
- All audio files are expected to be readable by `librosa` (WAV, FLAC, MP3, etc.)
- The script does not modify audio files; it only creates file path references

## Integration Checklist

After preprocessing, verify:

- [ ] Pickle files created successfully in output directory
- [ ] Both train and test DataFrames have `path`, `labels`, and `emotions` columns
- [ ] Emotion distribution looks reasonable (no unexpected labels)
- [ ] Paths in DataFrame point to existing audio files
- [ ] Dataset works with training script: `python train_experiment.py --train-datasets MSP_PODCAST --test-dataset MSP_PODCAST`

## References

- MSP Podcast Corpus: https://lab-msp.com/MSP/MSP-Podcast.html
- Dataset Paper: Check the official MSP Podcast documentation for citation information


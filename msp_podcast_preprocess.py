#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSP Podcast Dataset Preprocessing Script

This script preprocesses the MSP Podcast dataset to match the format used by
other datasets in this project (CAFE, CREMA, RAVDESS, SAVEE, TESS).

The MSP Podcast corpus contains over 400 hours of naturalistic emotional speech
from podcast recordings, annotated with categorical emotions and dimensional attributes.

Dataset Information:
- Partitions: Train, Development, Test1, Test2, Test3 (speaker-independent splits)
- Annotations: Categorical emotions + dimensional attributes (valence, arousal, dominance)
- Format: Typically includes audio files and annotation CSV files

Expected Input Structure:
- msp_podcast_dataset/
  - audio/ (or similar folder with WAV files)
  - annotations.csv (or similar annotation file)
  - labels_consensus.csv (or categorical emotion labels file)
  - (other metadata files as provided)

Output:
- D.MSP_PODCAST_train_DF.p: Training set DataFrame
- D.MSP_PODCAST_test_DF.p: Test set DataFrame
"""

import os
import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
import glob

ROOT_FOLDER = '/home/x002/PROJECT'
DATASETS_FOLDER = ROOT_FOLDER + '/Datasets/'
DFs_FOLDER = ROOT_FOLDER + '/DFs/'

# Standard emotion mapping (7 classes as used in other datasets)
EMOTION_MAPPING = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6
}

# Reverse mapping for label to emotion name
ID_TO_EMOTION = {v: k for k, v in EMOTION_MAPPING.items()}

# Common MSP Podcast emotion label variations to standard mapping
MSP_EMOTION_MAPPING = {
    # Direct matches
    'neutral': 'neutral',
    'happiness': 'happy',
    'happy': 'happy',
    'sadness': 'sad',
    'sad': 'sad',
    'anger': 'angry',
    'angry': 'angry',
    'fear': 'fear',
    'disgust': 'disgust',
    'surprise': 'surprise',
    # Common variations
    'joy': 'happy',
    'happiness_happy': 'happy',
    'excitement': 'happy',
    'surprised': 'surprise',
    'angry_anger': 'angry',
    'sad_sadness': 'sad',
    'fearful': 'fear',
    'disgusted': 'disgust',
    # Other possible labels that might need mapping
    'contempt': 'disgust',  # Map contempt to disgust if needed
    'calm': 'neutral',      # Map calm to neutral (like RAVDESS)
}


def find_audio_file(file_id, audio_dirs):
    """
    Search for audio file in multiple possible directories.
    
    Args:
        file_id: File identifier (e.g., filename without extension)
        audio_dirs: List of directories to search
        
    Returns:
        Full path to audio file if found, None otherwise
    """
    for audio_dir in audio_dirs:
        if not os.path.exists(audio_dir):
            continue
        
        # Try different possible extensions
        for ext in ['.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3']:
            # Try exact filename
            path = os.path.join(audio_dir, file_id + ext)
            if os.path.exists(path):
                return path
            
            # Try with different case
            path = os.path.join(audio_dir, file_id.lower() + ext)
            if os.path.exists(path):
                return path
            
            path = os.path.join(audio_dir, file_id.upper() + ext)
            if os.path.exists(path):
                return path
        
        # Try searching with glob
        pattern = os.path.join(audio_dir, f"*{file_id}*")
        matches = glob.glob(pattern)
        if matches:
            # Prefer .wav files
            wav_matches = [m for m in matches if m.lower().endswith('.wav')]
            if wav_matches:
                return wav_matches[0]
            return matches[0]
    
    return None


def normalize_emotion_label(emotion_str):
    """
    Normalize MSP Podcast emotion labels to standard format.
    
    Args:
        emotion_str: Raw emotion label from MSP Podcast
        
    Returns:
        Standardized emotion name (one of: neutral, happy, sad, angry, fear, disgust, surprise)
    """
    if pd.isna(emotion_str) or emotion_str is None:
        return 'neutral'
    
    emotion_str = str(emotion_str).lower().strip()
    
    # Direct mapping
    if emotion_str in MSP_EMOTION_MAPPING:
        return MSP_EMOTION_MAPPING[emotion_str]
    
    # Check if it contains any of the standard emotions
    for msp_label, std_label in MSP_EMOTION_MAPPING.items():
        if msp_label in emotion_str:
            return std_label
    
    # Default to neutral if unknown
    print(f"Warning: Unknown emotion label '{emotion_str}', mapping to 'neutral'")
    return 'neutral'


def load_msp_annotations(annotations_path, audio_dirs):
    """
    Load MSP Podcast annotations and create DataFrame with path, labels, emotions.
    
    Args:
        annotations_path: Path to annotation CSV file
        audio_dirs: List of directories containing audio files
        
    Returns:
        DataFrame with columns: path, labels, emotions
    """
    print(f"Loading annotations from: {annotations_path}")
    
    # Try to read the CSV file
    try:
        df = pd.read_csv(annotations_path)
        print(f"Loaded {len(df)} annotation entries")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        raise ValueError(f"Failed to load annotations file: {e}")
    
    # Identify relevant columns (common MSP Podcast CSV column names)
    # File identifier column
    file_id_col = None
    for col in ['file_name', 'file', 'filename', 'segment_id', 'id', 'utterance_id', 'name']:
        if col in df.columns:
            file_id_col = col
            break
    
    if file_id_col is None:
        raise ValueError(f"Could not find file identifier column. Available columns: {list(df.columns)}")
    
    # Emotion label column (could be categorical or consensus)
    emotion_col = None
    for col in ['emotion', 'Emotion', 'emotion_label', 'EmotionLabel', 
                'categorical', 'Categorical', 'emotion_categorical',
                'consensus', 'Consensus', 'label', 'Label']:
        if col in df.columns:
            emotion_col = col
            break
    
    if emotion_col is None:
        print("Warning: Could not find emotion column. Looking for partition info...")
        # If no emotion column, we might need to use dimensional attributes
        # For now, we'll require it or use a default
        raise ValueError(f"Could not find emotion label column. Available columns: {list(df.columns)}")
    
    # Partition column (for train/test split)
    partition_col = None
    for col in ['Split', 'split', 'Partition', 'partition', 'Set', 'set', 
                'usage', 'Usage', 'train_test']:
        if col in df.columns:
            partition_col = col
            break
    
    # Process each row
    data = []
    missing_audio = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}...")
        
        file_id = str(row[file_id_col]).strip()
        # Remove extension if present
        file_id = os.path.splitext(file_id)[0]
        
        # Find audio file
        audio_path = find_audio_file(file_id, audio_dirs)
        
        if audio_path is None:
            missing_audio.append(file_id)
            continue
        
        # Get emotion label
        emotion_str = row[emotion_col]
        emotion = normalize_emotion_label(emotion_str)
        label = EMOTION_MAPPING[emotion]
        
        # Get partition if available
        partition = None
        if partition_col:
            partition = str(row[partition_col]).lower().strip()
        
        data.append({
            'path': audio_path,
            'labels': label,
            'emotions': emotion,
            'partition': partition,
            'file_id': file_id
        })
    
    if missing_audio:
        print(f"\nWarning: Could not find {len(missing_audio)} audio files (first 10: {missing_audio[:10]})")
    
    result_df = pd.DataFrame(data)
    print(f"\nSuccessfully processed {len(result_df)} samples")
    print(f"Emotion distribution:")
    print(result_df['emotions'].value_counts())
    
    return result_df


def split_train_test(df, partition_col='partition', test_ratio=0.2):
    """
    Split DataFrame into train and test sets.
    
    Args:
        df: DataFrame with annotations
        partition_col: Column name indicating partition (if available)
        test_ratio: Ratio for test set if partition info not available
        
    Returns:
        train_df, test_df
    """
    if partition_col in df.columns and df[partition_col].notna().any():
        # Use provided partitions
        print("Using dataset partitions for train/test split")
        
        # Common partition labels
        train_partitions = ['train', 'training', 'dev', 'development']
        test_partitions = ['test', 'test1', 'test2', 'test3', 'testing']
        
        train_mask = df[partition_col].str.lower().isin([p.lower() for p in train_partitions])
        test_mask = df[partition_col].str.lower().isin([p.lower() for p in test_partitions])
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # If no test partition found, split from train
        if len(test_df) == 0:
            print("No test partition found, splitting from train set")
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                train_df, test_size=test_ratio, random_state=19730309, 
                stratify=train_df['emotions']
            )
    else:
        # Random split maintaining class distribution
        print(f"Using random split with {test_ratio} test ratio")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=19730309, 
            stratify=df['emotions']
        )
    
    # Remove partition column before saving
    if partition_col in train_df.columns:
        train_df = train_df.drop(columns=[partition_col, 'file_id'])
    if partition_col in test_df.columns:
        test_df = test_df.drop(columns=[partition_col, 'file_id'])
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\nSplit summary:")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"\nTraining emotion distribution:")
    print(train_df['emotions'].value_counts())
    print(f"\nTest emotion distribution:")
    print(test_df['emotions'].value_counts())
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MSP Podcast dataset for SER training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with annotation file and audio directory
  python msp_podcast_preprocess.py --annotations path/to/annotations.csv --audio-dir path/to/audio/

  # With multiple audio directories
  python msp_podcast_preprocess.py --annotations path/to/annotations.csv --audio-dir path/to/audio1/ --audio-dir path/to/audio2/

  # Custom output directory
  python msp_podcast_preprocess.py --annotations path/to/annotations.csv --audio-dir path/to/audio/ --output-dir custom/path/

Expected annotation CSV format:
  - Column with file identifier (e.g., 'file_name', 'file', 'segment_id')
  - Column with emotion labels (e.g., 'emotion', 'emotion_label', 'categorical')
  - Optional: Column with partition info (e.g., 'Split', 'partition', 'Set')
        """
    )
    
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotation CSV file')
    parser.add_argument('--audio-dir', type=str, action='append', required=True,
                       help='Directory(ies) containing audio files (can be specified multiple times)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help=f'Output directory for pickle files (default: {DFs_FOLDER})')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio if partition info not available (default: 0.2)')
    parser.add_argument('--root-folder', type=str, default=ROOT_FOLDER,
                       help=f'Root project folder (default: {ROOT_FOLDER})')
    
    args = parser.parse_args()
    
    # Set up paths
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = DFs_FOLDER if args.root_folder == ROOT_FOLDER else args.root_folder + '/DFs/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(args.annotations):
        raise ValueError(f"Annotation file not found: {args.annotations}")
    
    for audio_dir in args.audio_dir:
        if not os.path.exists(audio_dir):
            print(f"Warning: Audio directory not found: {audio_dir}")
    
    # Load and process annotations
    df = load_msp_annotations(args.annotations, args.audio_dir)
    
    # Split into train and test
    train_df, test_df = split_train_test(df, test_ratio=args.test_ratio)
    
    # Save pickle files
    os.chdir(output_dir)
    DATASET_n = 'D.MSP_PODCAST_'
    TRAIN_DF = DATASET_n + 'train_DF.p'
    TEST_DF = DATASET_n + 'test_DF.p'
    
    print(f"\nSaving to:")
    print(f"  {os.path.join(output_dir, TRAIN_DF)}")
    print(f"  {os.path.join(output_dir, TEST_DF)}")
    
    pickle.dump(train_df, open(TRAIN_DF, 'wb'))
    pickle.dump(test_df, open(TEST_DF, 'wb'))
    
    print("\nPreprocessing complete!")
    print(f"\nDataFrame columns: {list(train_df.columns)}")
    print(f"\nSample train entry:")
    print(train_df.head(1).to_dict('records')[0])
    
    print("\nTo use this dataset in training, use:")
    print("  --train-datasets MSP_PODCAST")
    print("(Note: You may need to add 'MSP_PODCAST' to the DATASET_MAP in train_experiment.py)")


if __name__ == '__main__':
    main()


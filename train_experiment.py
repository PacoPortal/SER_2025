#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified training script for SER (Speech Emotion Recognition) experiments.
Supports all variations previously implemented in separate scripts through command-line arguments.

Usage Examples:
--------------
# Independent experiment: train on CAFE+CREMA+EMOFILM, test on CREMA, unfrozen
python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM --test-dataset CREMA --experiment-type indep

# Independent experiment with frozen encoder (--freeze flag)
python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM --test-dataset CREMA --experiment-type indep --freeze

# Cross-corpus experiment: train on all datasets
python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM,RAVDESS,SAVEE,TESS --experiment-type cross

# IJNS-style experiment: train on RAVDESS, split RAVDESS for test/val
python train_experiment.py --train-datasets RAVDESS --test-split-source RAVDESS --experiment-type indep --freeze

# Use different model
python train_experiment.py --train-datasets CAFE,CREMA --test-dataset CREMA --model HuB-ER

# Custom learning rate and epochs
python train_experiment.py --train-datasets CAFE,CREMA --test-dataset CREMA --lr 1e-4 --epochs 30

Available datasets: CAFE, CREMA, EMOFILM, RAVDESS, SAVEE, TESS, MSP_PODCAST
Available models: HuB-LL, HuB-XL, HuB-ER, HuB-SID, W2V2-base, W2V2-large, W2V2-robust, DistilHubert
"""

from IndependentSER_common_code import *
import os
import argparse
import gc
import datetime
import fnmatch
import numpy as np
import pandas as pd
import random
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import librosa
import librosa.display
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import Wav2Vec2FeatureExtractor
from transformers import HubertForSequenceClassification
from sklearn.metrics import accuracy_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import itertools

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_name = torch.cuda.get_device_name(DEVICE)
else:
    DEVICE = torch.device("cpu")
    DEVICE_name = "cpu"

print("We are using the DEVICE %s - %s" % (DEVICE, DEVICE_name))

from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

# Available datasets mapping
DATASET_MAP = {
    'CAFE': {'name': 'CAFE_', 'key': 'CAFE'},
    'CREMA': {'name': 'CREMA_', 'key': 'CREMA'},
    'EMOFILM': {'name': 'EMOFILM_', 'key': 'EMOFILM'},
    'RAVDESS': {'name': 'RAVDESS_', 'key': 'RAVDESS'},
    'SAVEE': {'name': 'SAVEE_', 'key': 'SAVEE'},
    'TESS': {'name': 'TESS_', 'key': 'TESS'},
    'MSP_PODCAST': {'name': 'MSP_PODCAST_', 'key': 'MSP_PODCAST'}
}

# Available model checkpoints
AVAILABLE_CHECKPOINTS = {
    'HuB-LL': 'facebook/hubert-large-ll60k',
    'HuB-XL': 'facebook/hubert-xlarge-ll60k',
    'HuB-ER': 'superb/hubert-large-superb-er',
    'HuB-SID': 'superb/hubert-large-superb-sid',
    'W2V2-base': 'facebook/wav2vec2-base-960h',
    'W2V2-large': 'facebook/wav2vec2-large',
    'W2V2-robust': 'facebook/wav2vec2-large-robust-ft-libri-960h',
    'DistilHubert': 'ntu-spml/distilhubert'
}


def load_datasets(dfs_folder):
    """Load all datasets from pickle files."""
    os.chdir(dfs_folder)
    
    datasets = {}
    for ds_key, ds_info in DATASET_MAP.items():
        DATASET_n = 'D.' + ds_info['name']
        TEST_DF = DATASET_n + 'test_DF.p'
        TRAIN_DF = DATASET_n + 'train_DF.p'
        
        train_df = pickle.load(open(TRAIN_DF, 'rb'))
        test_df = pickle.load(open(TEST_DF, 'rb'))
        full_df = pd.concat([train_df, test_df])
        full_df.reset_index(inplace=True, drop=True)
        
        # Special handling for RAVDESS
        if ds_key == 'RAVDESS':
            train_df = train_df.drop('gender', axis=1) if 'gender' in train_df.columns else train_df
            train_df = train_df.replace({'calm': 'neutral'})
            train_df = train_df.replace({7: 0})
            test_df = test_df.drop('gender', axis=1) if 'gender' in test_df.columns else test_df
            test_df = test_df.replace({'calm': 'neutral'})
            test_df = test_df.replace({7: 0})
            full_df = pd.concat([train_df, test_df])
            full_df.reset_index(inplace=True, drop=True)
        
        # MSP_PODCAST doesn't need special preprocessing (already handled in preprocessing script)
        
        datasets[ds_key] = {
            'train': train_df,
            'test': test_df,
            'full': full_df
        }
    
    return datasets


def prepare_train_test_val_datasets(datasets, train_datasets, test_dataset, val_split_frac=0.5, 
                                     test_split_source=None, experiment_type='indep'):
    """
    Prepare train, validation, and test datasets based on configuration.
    
    Args:
        datasets: Dictionary of loaded datasets
        train_datasets: List of dataset keys to use for training (e.g., ['CAFE', 'CREMA', 'EMOFILM'])
        test_dataset: Dataset key to use for testing (e.g., 'CREMA')
        val_split_frac: Fraction to use for validation when splitting test data
        test_split_source: If not None, split this dataset for test/val instead of using test_dataset
        experiment_type: 'indep' or 'cross'
    """
    # Prepare training data
    train_dfs = []
    train_names = []
    for ds_key in train_datasets:
        if experiment_type == 'indep':
            train_dfs.append(datasets[ds_key]['train'])
        else:  # cross
            train_dfs.append(datasets[ds_key]['train'])
        train_names.append(ds_key)
    
    train_df = pd.concat(train_dfs)
    train_df.reset_index(inplace=True, drop=True)
    DATASET_n = '_'.join([DATASET_MAP[ds]['key'] for ds in train_datasets])
    
    # Prepare test and validation data
    if test_split_source is not None:
        # Split a specific dataset for test/val
        source_test_df = datasets[test_split_source]['test']
        test_df, val_df = data_split(source_test_df, frac=val_split_frac)
        testlabel = f'{test_split_source}_split'
    elif experiment_type == 'cross':
        # For cross-corpus: split test data from all datasets
        test_dfs = []
        val_dfs = []
        for ds_key in train_datasets:
            ds_test = datasets[ds_key]['test']
            t_df, v_df = data_split(ds_test, frac=val_split_frac)
            test_dfs.append(t_df)
            val_dfs.append(v_df)
        test_df = pd.concat(test_dfs)
        test_df.reset_index(inplace=True, drop=True)
        val_df = pd.concat(val_dfs)
        val_df.reset_index(inplace=True, drop=True)
        testlabel = 'CROSS'
    else:
        # Independent: use full test dataset
        if test_dataset in datasets:
            test_df = datasets[test_dataset]['full']
            val_df = datasets[test_dataset]['full']
            testlabel = test_dataset
        else:
            # Use test splits from training datasets
            test_dfs = []
            val_dfs = []
            for ds_key in train_datasets:
                test_dfs.append(datasets[ds_key]['test'])
                val_dfs.append(datasets[ds_key]['test'])
            test_df = pd.concat(test_dfs)
            test_df.reset_index(inplace=True, drop=True)
            val_df = pd.concat(val_dfs)
            val_df.reset_index(inplace=True, drop=True)
            testlabel = '_'.join(train_datasets)
    
    return train_df, test_df, val_df, DATASET_n, testlabel


def parse_dataset_list(dataset_str):
    """Parse comma-separated dataset string into list."""
    return [ds.strip().upper() for ds in dataset_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='Unified SER training script with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Independent experiment: train on CAFE+CREMA+EMOFILM, test on CREMA, unfrozen
  python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM --test-dataset CREMA --experiment-type indep --freeze False

  # Cross-corpus experiment: train on all datasets, freeze encoder
  python train_experiment.py --train-datasets CAFE,CREMA,EMOFILM,RAVDESS,SAVEE,TESS --experiment-type cross --freeze True

  # IJNS-style experiment: train on RAVDESS, split RAVDESS for test/val
  python train_experiment.py --train-datasets RAVDESS --test-split-source RAVDESS --experiment-type indep --freeze True

Available datasets: CAFE, CREMA, EMOFILM, RAVDESS, SAVEE, TESS, MSP_PODCAST
Available models: HuB-LL, HuB-XL, HuB-ER, HuB-SID, W2V2-base, W2V2-large, W2V2-robust, DistilHubert
        """
    )
    
    # Experiment configuration
    parser.add_argument('--experiment-type', type=str, choices=['indep', 'cross'], default='indep',
                       help='Type of experiment: indep (independent) or cross (cross-corpus)')
    
    # Dataset configuration
    parser.add_argument('--train-datasets', type=str, required=True,
                       help='Comma-separated list of datasets for training (e.g., CAFE,CREMA,EMOFILM)')
    parser.add_argument('--test-dataset', type=str, default=None,
                       help='Dataset to use for testing (e.g., CREMA). If not provided, uses test splits from train datasets.')
    parser.add_argument('--test-split-source', type=str, default=None,
                       help='Dataset to split for test/val (for IJNS-style experiments)')
    parser.add_argument('--val-split-frac', type=float, default=0.5,
                       help='Fraction for validation split (default: 0.5)')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='HuB-LL',
                       choices=list(AVAILABLE_CHECKPOINTS.keys()),
                       help='Model checkpoint to use (default: HuB-LL)')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Custom checkpoint path (overrides --model)')
    
    # Training configuration
    parser.add_argument('--freeze', action='store_true',
                       help='Freeze feature encoder (default: False)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--rondas', type=int, default=3,
                       help='Number of rounds/runs (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--weight-decay', type=float, default=0.005,
                       help='Weight decay (default: 0.005)')
    parser.add_argument('--use-attention-mask', action='store_true', default=True,
                       help='Use attention mask in training (default: True)')
    
    # Paths
    parser.add_argument('--root-folder', type=str, default='/home/x002/PROJECT',
                       help='Root folder for datasets and outputs (default: /home/x002/PROJECT)')
    parser.add_argument('--dump-folder', type=str, default=None,
                       help='Output folder (default: {root_folder}/dump/)')
    
    # Advanced options
    parser.add_argument('--epoch-ini', type=int, default=1,
                       help='Initial epoch number (for resuming, default: 1)')
    parser.add_argument('--best-acc', type=float, default=0.0,
                       help='Initial best accuracy (for resuming, default: 0.0)')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name for output folder')
    
    args = parser.parse_args()
    
    # Handle freeze flag
    freeze = args.freeze
    
    # Set up paths
    ROOT_FOLDER = args.root_folder
    DATASETS_FOLDER = ROOT_FOLDER + '/Datasets/'
    DFs_FOLDER = ROOT_FOLDER + '/DFs/'
    DUMP_FOLDER = args.dump_folder if args.dump_folder else ROOT_FOLDER + '/dump/'
    
    # Parse datasets
    train_datasets = parse_dataset_list(args.train_datasets)
    
    # Validate datasets
    for ds in train_datasets:
        if ds not in DATASET_MAP:
            parser.error(f"Invalid dataset: {ds}. Available: {', '.join(DATASET_MAP.keys())}")
    
    if args.test_dataset and args.test_dataset not in DATASET_MAP:
        parser.error(f"Invalid test dataset: {args.test_dataset}. Available: {', '.join(DATASET_MAP.keys())}")
    
    if args.test_split_source and args.test_split_source not in DATASET_MAP:
        parser.error(f"Invalid test split source: {args.test_split_source}. Available: {', '.join(DATASET_MAP.keys())}")
    
    # Get model checkpoint
    if args.checkpoint_path:
        checkpoint = args.checkpoint_path
    else:
        checkpoint = AVAILABLE_CHECKPOINTS[args.model]
    
    print(f"El nombre de este archivo es: {os.path.basename(__file__)}")
    job_id = os.environ.get('SLURM_JOB_ID')
    print(f"Ejecutando el trabajo con ID: {job_id}")
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_datasets(DFs_FOLDER)
    
    # Prepare train/test/val datasets
    train_df, test_df, val_df, DATASET_n, testlabel = prepare_train_test_val_datasets(
        datasets, train_datasets, args.test_dataset, args.val_split_frac,
        args.test_split_source, args.experiment_type
    )
    
    print(f'Dataset: {DATASET_n}, Experiment type: {args.experiment_type}, Test label: {testlabel}')
    print(f'Train samples: {len(train_df)}, Test samples: {len(test_df)}, Val samples: {len(val_df)}')
    
    # Prepare datasets
    train_dataset = SpeechDataset(train_df)
    test_val_dataset = SpeechDataset(val_df)
    test_test_dataset = SpeechDataset(test_df)
    
    # Label configuration
    labeldict = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fear': 4, 'disgust': 5, 'surprise': 6}
    label2id, id2label = dict(), dict()
    for emotion in labeldict:
        label2id[emotion] = str(labeldict[emotion])
        id2label[str(labeldict[emotion])] = emotion
    num_labels = len(id2label)
    
    # Create experiment folder
    tms = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_name = args.experiment_name if args.experiment_name else f'experim_{tms}'
    EXPERIMENT_FOLDER = DUMP_FOLDER + exp_name + '.' + str(job_id) + '.' + args.experiment_type + '/'
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    DUMP_FOLDER = EXPERIMENT_FOLDER
    
    # Training loop
    checkpoints = (checkpoint, None)
    for c in checkpoints:
        if c == None:
            break
        
        acc = 0.0
        checkpoint_tuple = (c, None)
        
        for round_num in range(args.rondas):
            tms = time.strftime("%Y%m%d%H%M%S", time.localtime())
            print(f"Round {round_num + 1}/{args.rondas} - tms: {tms}")
            print(f'LR: {args.lr}, WD: {args.weight_decay}, Freeze: {freeze}')
            
            epoch_ini = args.epoch_ini
            best_acc = args.best_acc
            
            TrainParams = (tms, epoch_ini, best_acc, args.lr, args.weight_decay)
            os.chdir(DUMP_FOLDER)
            
            # Set random seeds
            RANDOM_SEED = 19730309
            torch.manual_seed(RANDOM_SEED)
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            
            # Training
            if args.use_attention_mask:
                val_acc, val_loss, accuracy, test_loss, record, tmp_name, model = loop_with_att_mask(
                    checkpoint_tuple, train_dataset, test_test_dataset, test_val_dataset, DATASET_n,
                    label2id, id2label, num_labels, args.lr, args.weight_decay, tms, DUMP_FOLDER,
                    N_EPOCHS=args.epochs, epoch_ini=epoch_ini, best_acc=best_acc,
                    freeze=freeze, testlabel=testlabel
                )
            else:
                result = loop(
                    checkpoint_tuple, train_dataset, test_test_dataset, test_val_dataset, DATASET_n,
                    label2id, id2label, num_labels, args.lr, args.weight_decay, tms, DUMP_FOLDER,
                    N_EPOCHS=args.epochs, epoch_ini=epoch_ini, best_acc=best_acc,
                    freeze=freeze, testlabel=testlabel
                )
                if len(result) == 4:
                    val_acc, val_loss, accuracy, test_loss = result
                    record = None
                    tmp_name = None
                    model = None
                else:
                    # Fallback if loop returns different format
                    val_acc, val_loss, accuracy, test_loss = result[:4]
                    record = None
                    tmp_name = None
                    model = None
            
            # Save plots and confusion matrices (if using attention mask)
            if args.use_attention_mask and record is not None:
                os.chdir(DUMP_FOLDER)
                plot_loss_acc(val_loss, val_acc, str(round_num))
                
                RECORD = tmp_name + '.record.p'
                pickle.dump(record, open(RECORD, 'wb'))
                
                df_record = incorrect_ones(record, test_df)
                display_confusion_matrix(df_record, str(round_num) + '_Test_', id2label)
                
                # Drop surprise class (label 6) for additional analysis
                df_record2 = df_record[df_record['Model Prediction'] != 6]
                df_record2 = df_record2[df_record2['Ground Truth'] != 6]
                display_confusion_matrix(df_record2, str(round_num) + '_Test_Drop6_', id2label)
            
            acc += accuracy
            if len(checkpoints) == 2:
                print(f'Round {round_num + 1} accuracy: {accuracy:.4f}')
    
    print(f'\nFinal test accuracy (mean over {args.rondas} rounds): {acc/args.rondas:.4f}')
    print('End training')


if __name__ == '__main__':
    main()


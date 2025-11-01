# -*- coding: utf-8 -*-
#CrossCorpus_2024Q4.Final_Model_paper.ipynb

from IndependentSER_common_code import *

import os
'''

CACHE_FOLDER = '/content/drive/MyDrive/Colab Notebooks/TFM_UPM_2024/pickle24/var_cache'
os.environ['TRANSFORMERS_CACHE'] = CACHE_FOLDER
os.environ['HF_HOME'] = CACHE_FOLDER
os.environ['HF_DATASETS_CACHE'] = CACHE_FOLDER
os.environ['TORCH_HOME'] = CACHE_FOLDER

'''
ROOT_FOLDER =   '/home/x002/PROJECT'
DATASETS_FOLDER = ROOT_FOLDER+'/Datasets/'
DFs_FOLDER =     ROOT_FOLDER+'/DFs/'
DUMP_FOLDER = ROOT_FOLDER+'/dump/'

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

#from IPython.display import Audio
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



"""#Trainings

##Settings
"""

############## script ######################################################################

labeldict = ({'neutral':0, 'happy':1, 'sad':2, 'angry':3, 'fear':4, 'disgust':5, 'surprise':6})

label2id, id2label = dict(), dict()
for emotion in labeldict:
    label2id[emotion] = str(labeldict[emotion])
    id2label[str(labeldict[emotion])] = emotion

num_labels = len(id2label)

os.chdir(DFs_FOLDER)

DATASET_n = 'D.CAFE_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'
CAFE_train_df = pickle.load(open(TRAIN_DF, 'rb'))
CAFE_test_df = pickle.load(open(TEST_DF, 'rb'))
CAFE_df = pd.concat([CAFE_train_df,CAFE_test_df])
CAFE_df.reset_index(inplace=True,drop=True)

DATASET_n = 'D.CREMA_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'
CREMA_train_df = pickle.load(open(TRAIN_DF, 'rb'))
CREMA_test_df = pickle.load(open(TEST_DF, 'rb'))
CREMA_df = pd.concat([CREMA_train_df,CREMA_test_df])
CREMA_df.reset_index(inplace=True,drop=True)

DATASET_n = 'D.EMOFILM_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'
EMOFILM_train_df = pickle.load(open(TRAIN_DF, 'rb'))
EMOFILM_test_df = pickle.load(open(TEST_DF, 'rb'))
EMOFILM_df = pd.concat([EMOFILM_train_df,EMOFILM_test_df])
EMOFILM_df.reset_index(inplace=True,drop=True)

DATASET_n = 'D.RAVDESS_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'
RAVDESS_train_df = pickle.load(open(TRAIN_DF, 'rb'))
RAVDESS_test_df = pickle.load(open(TEST_DF, 'rb'))
RAVDESS_train_df = RAVDESS_train_df.drop('gender',axis=1)
RAVDESS_train_df = RAVDESS_train_df.replace({'calm':'neutral'})  # ¿mejor drop calm
RAVDESS_train_df = RAVDESS_train_df.replace({7:0})
RAVDESS_test_df = RAVDESS_test_df.drop('gender',axis=1)
RAVDESS_test_df = RAVDESS_test_df.replace({'calm':'neutral'})  # ¿mejor drop calm
RAVDESS_test_df = RAVDESS_test_df.replace({7:0})
RAVDESS_df = pd.concat([RAVDESS_train_df,RAVDESS_test_df])
RAVDESS_df.reset_index(inplace=True,drop=True)

DATASET_n = 'D.SAVEE_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'
SAVEE_train_df = pickle.load(open(TRAIN_DF, 'rb'))
SAVEE_test_df = pickle.load(open(TEST_DF, 'rb'))
SAVEE_df = pd.concat([SAVEE_train_df,SAVEE_test_df])
SAVEE_df.reset_index(inplace=True,drop=True)

DATASET_n = 'D.TESS_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'
TESS_train_df = pickle.load(open(TRAIN_DF, 'rb'))
TESS_test_df = pickle.load(open(TEST_DF, 'rb'))
TESS_df = pd.concat([TESS_train_df,TESS_test_df])
TESS_df.reset_index(inplace=True,drop=True)

torch.set_warn_always = False

classes = np.unique(RAVDESS_df['labels'].values)

classes_names = np.unique(RAVDESS_df['emotions'].values)

values = [RAVDESS_df['emotions'].values.tolist().count(class_) for class_ in classes_names] # frequency

#print(CAFE_df.emotions.value_counts())


LR_hub = (1e-5 + 1e-4)/2
WEIGHT_DECAY_hub = (0.01 + 0.005)/2

LR2 = 1e-5
WEIGHT_DECAY2 = 0.005

LR3 = 1e-4
WEIGHT_DECAY3 = 0.01

lr_ops = (LR2,LR3)
wd_ops = (WEIGHT_DECAY2,WEIGHT_DECAY3)

lr = LR2
wd = WEIGHT_DECAY2 

job_id = os.environ.get('SLURM_JOB_ID')
#print("Current drive: ", os.getcwd())
print(f"El nombre de este archivo es: {os.path.basename(__file__)}")
print(f"Ejecutando el trabajo con ID: {job_id}")


DATASET_n ="CA_CR_EM_SA_TE"
#TYPE_EXPERIMENT = 'cross'
TYPE_EXPERIMENT = 'indep'
print('Datastet: ',DATASET_n,', y tipo de experimento: ',TYPE_EXPERIMENT,', test on CREMA')
print('TRAIN con dfs train, val con dfs val y test con CREMA completo')
print('003 es ravdess itself y luego test on CREMA')
#train_df = pd.concat([CREMA_train_df, TESS_train_df, RAVDESS_train_df, SAVEE_train_df])#CREMA does not have surprise !!!!!!!!!
train_df = RAVDESS_train_df
val_df, test_df = data_split(RAVDESS_test_df,  frac=0.5)
#data_distribution(train_df) plot of distributions
#val_df = pd.concat([SAVEE_test_df, RAVDESS_test_df, TESS_test_df, EMOFILM_test_df, CAFE_test_df])
#test_df =  CREMA_df#CREMA_train_df #pd.concat([RAVDESS_test_df, SAVEE_test_df])#CREMA does not have surprise !!!!!!!!!
train_dataset = SpeechDataset(train_df)
test_val_dataset  = SpeechDataset(val_df)
test_test_dataset  = SpeechDataset(test_df)
checkpoints=('facebook/hubert-large-ll60k',None)

DUMP_FOLDER = ROOT_FOLDER+'/dump/'+'experim_003_.20250608220823.801801.indep'
os.chdir(DUMP_FOLDER)
T='''
dump/
dump/experim_003_.20250608220823.801801.indep/20250608220823.CA_CR_EM_SA_TE.hubert-large-ll60k_log.p
dump/experim_003_.20250608220823.801801.indep/20250608220823.CA_CR_EM_SA_TE.hubert-large-ll60k.record.p
'''
LOGFILE = '20250608220823.CA_CR_EM_SA_TE.hubert-large-ll60k_log.p'
train_log = pickle.load(open(LOGFILE, 'rb'))
(val_acc, val_loss) = train_log
plot_loss_acc(val_loss, val_acc)#save plot files

RECORD = '20250608220823.CA_CR_EM_SA_TE.hubert-large-ll60k.record.p'
record = pickle.load(open(RECORD, 'rb'))

df_record = incorrect_ones(record,test_df)
print('df_records')
print(df_record.head())
df_record2 = df_record[df_record['Model Prediction'] != 6]
df_record2 = df_record2[df_record['Ground Truth'] != 6]


#print(test_df.emotions.value_counts())
#plot_loss_acc(val_loss, val_acc)#save plot files
display_confusion_matrix(df_record,'TestCR_',id2label)#save plot files
display_confusion_matrix(df_record2,'TestCR_Drop6_',id2label)







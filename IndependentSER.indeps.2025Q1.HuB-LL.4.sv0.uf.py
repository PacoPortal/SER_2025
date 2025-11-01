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

#cd_df, cv_df = data_split(CREMA_test_df, frac=0.5)
#t_df, tv_df = data_split(TESS_test_df, frac=0.5)
#r_df, rv_df = data_split(RAVDESS_test_df, frac=0.5)
#s_df, sv_df = data_split(SAVEE_test_df, frac=0.5)
#test_df = pd.concat([cd_df, tt_df, rt_df, st_df])#CREMA does not have surprise !!!!!!!!!
#test_df.reset_index(inplace=True,drop=True)
#val_df = test_df
#TYPE_EXPERIMENT = 'cross'
TYPE_EXPERIMENT = 'indep'

testdstname= 'SAVEE'
test_df=SAVEE_df
val_df=SAVEE_df

print('Test testing on ',testdstname)
datasets = [[CAFE_df,'CAFE_'],[CREMA_df,'CREMA_'],[EMOFILM_df,'EMOFILM_'],[RAVDESS_df,'RAVDESS_'],[TESS_df,'TESS_']]
ds = (1,2,3,4)


combinaciones = []
#for combinacion in itertools.combinations(ds, 3):
#  combinaciones.append(combinacion)
for combinacion in itertools.combinations(ds, 4):
  combinaciones.append(combinacion)


#for comb in combinaciones:
#4#comb = (0,1,2,3)
#3#comb = (0,1,2,4)
#2#comb = (0,1,3,4)
#1#comb = (0,2,3,4)
comb = (1,2,3,4)

lista=[]
DATASET_n=""
for cm in comb:
    lista.append(datasets[cm][0])
    DATASET_n+=datasets[cm][1]
train_df = pd.concat(lista)#CREMA does not have surprise !!!!!!!!!
train_df.reset_index(inplace=True,drop=True)
print('Datastet: ',DATASET_n,', TEST_DST es ',testdstname,' y tipo de experimento: ',TYPE_EXPERIMENT)
#data_distribution(train_df) plot of distributions
train_dataset = SpeechDataset(train_df)
test_val_dataset  = SpeechDataset(val_df)
test_test_dataset  = SpeechDataset(test_df)
#checkpoints=('facebook/hubert-large-ll60k','superb/hubert-large-superb-er','superb/hubert-large-superb-sid')
#checkpoints=('superb/hubert-large-superb-sid')#
#checkpoints=('superb/hubert-large-superb-er')
checkpoints=('facebook/hubert-large-ll60k',None)
rondas=5
for c in checkpoints:
    if c==None:
        break
    acc=0.0
    checkpoint = (c, None)
    for round in range(rondas):
        tms = time.strftime("%Y%m%d%H%M%S", time.localtime())
        print(f"Ejecutando el trabajo con tms: {tms}")
        print('LR es: ',lr)
        print('WD es: ',wd)
        EXPERIMENT_FOLDER=DUMP_FOLDER+'experim.'+tms+'.'+str(job_id)+'.'+TYPE_EXPERIMENT+'/'
        os.mkdir(EXPERIMENT_FOLDER)
        DUMP_FOLDER=EXPERIMENT_FOLDER
        epoch_ini=1
        best_acc = 0
        TrainParams = (tms,epoch_ini,best_acc,lr,wd)
        os.chdir(DUMP_FOLDER)
        pickle.dump(TrainParams,  open(DUMP_FOLDER+'Train_State.p', 'wb'))
        RANDOM_SEED = 19730309
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        val_acc, val_loss, accuracy, test_loss = loop_with_att_mask(checkpoint, train_dataset, test_test_dataset, test_val_dataset, DATASET_n,
                                                    label2id, id2label, num_labels,lr, wd, tms, DUMP_FOLDER,
                                                    N_EPOCHS = 20,epoch_ini=epoch_ini,best_acc=best_acc,freeze=False, testlabel=testdstname)

        acc+=accuracy
        if len(checkpoints)==2:
            print('Test accuracy, mean: ', acc/rondas)
print('Test accuracy, mean: ', acc/rondas)
print('End training ')




'''

#tms,epoch_ini,best_acc,lr,wd = pickle.load(open(DUMP_FOLDER+'Train_State.p', 'rb'))
os.chdir(DUMP_FOLDER)
val_acc, val_loss = pickle.load(open('20241210155710_h_SAV_RAV_TESbert-large-superb-er_log.p', 'rb'))

plot_loss(val_loss)

plot_acc(val_acc)'''

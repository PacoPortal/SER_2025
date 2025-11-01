#from IndependentSER_common_code import *

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
os.chdir(DFs_FOLDER)
import gc
import datetime
import fnmatch
import numpy as np
import pandas as pd
import random
import pickle
import time
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

CAFE =    "./cafe_ds/"
PROCESS_PATHS = True


DATASET_n = 'D.CAFE_'
TEST_DF = DATASET_n + 'test_DF.p'
TRAIN_DF = DATASET_n + 'train_DF.p'

CAFE_train_df = pickle.load(open(TRAIN_DF, 'rb'))
CAFE_test_df = pickle.load(open(TEST_DF, 'rb'))





CAFE_train_df['path'] = CAFE_train_df['path'].str.replace('Colère', 'Colere', regex=False)
CAFE_train_df['path'] = CAFE_train_df['path'].str.replace('Dégoût', 'Degout', regex=False)
print(CAFE_train_df)

CAFE_test_df['path'] = CAFE_test_df['path'].str.replace('Colère', 'Colere', regex=False)
CAFE_test_df['path'] = CAFE_test_df['path'].str.replace('Dégoût', 'Degout', regex=False)

os.chdir(DFs_FOLDER)
pickle.dump(CAFE_train_df,  open(TRAIN_DF, 'wb'))
pickle.dump(CAFE_test_df,  open(TEST_DF, 'wb'))
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

DATASET_n ="SAV_RAV_TES_EMO_CAF"
#TYPE_EXPERIMENT = 'cross'
TYPE_EXPERIMENT = 'indep'
print('Datastet: ',DATASET_n,', y tipo de experimento: ',TYPE_EXPERIMENT)
#train_df = pd.concat([CREMA_train_df, TESS_train_df, RAVDESS_train_df, SAVEE_train_df])#CREMA does not have surprise !!!!!!!!!
train_df = pd.concat([RAVDESS_df, SAVEE_df, TESS_df, EMOFILM_df,CAFE_df])#CREMA does not have surprise !!!!!!!!!
train_df.reset_index(inplace=True,drop=True)
#data_distribution(train_df) plot of distributions
val_df = CREMA_df#CREMA_test_df
test_df =  CREMA_df#CREMA_train_df #pd.concat([RAVDESS_test_df, SAVEE_test_df])#CREMA does not have surprise !!!!!!!!!
train_dataset = SpeechDataset(train_df)
test_val_dataset  = SpeechDataset(val_df)
test_test_dataset  = SpeechDataset(test_df)
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
                                                  N_EPOCHS = 20,epoch_ini=epoch_ini,best_acc=best_acc,freeze=True, testlabel='CREMA')

      acc+=accuracy
      if len(checkpoints)==2:
        print('Test accuracy, mean: ', acc/rondas)
print('End training ')






tms,epoch_ini,best_acc,lr,wd = pickle.load(open(DUMP_FOLDER+'Train_State.p', 'rb'))
os.chdir(DUMP_FOLDER)
val_acc, val_loss = pickle.load(open('20241210155710_h_SAV_RAV_TESbert-large-superb-er_log.p', 'rb'))

plot_loss(val_loss)

plot_acc(val_acc)



model, epoch, optim, train_loss, train_accuracies = load_checkpoint(device, model, optim, MODEL_NAME, dump_folder=DUMP_FOLDER)

model.to(device)
record = testing(test_dataset, model)
os.chdir(DUMP_FOLDER)
pickle.dump(record, open(RECORD, 'wb'))

Check the result.

df_record = incorrect_ones(record,test_df)

print(test_df.emotions.value_counts())

Display the Confusion Matrix of the result.

display_confusion_matrix(df_record)

# 99: Record audio from your microphone in Colab
"""
Maily comes from "Noé Tits - Numediart (UMONS) - [noetits.com](https://noetits.com)" Noé also leaarned the part to record from microphone from [here](https://colab.research.google.com/gist/ricardodeazambuja/03ac98c31e87caf284f7b06286ebf7fd/microphone-to-numpy-array-from-your-browser-in-colab.ipynb)

# Preparar entorno en google drive. Si se ejecuta en local u en otro entorno
# habría que modificar esta celda
from google.colab import drive
import os
Project_folder = '/content/drive/MyDrive/Colab Notebooks/TFM_2023_24_MuIA/Transformer.1_202402/'
# Si se ejecutase de nuevo todo, sin desconectar el entorno, drive.mount() a
# veces da problemas. Con try/except solo se mota drive la primera vez y no lo
# vuelve a hacer cuando ya se ha montado

try:
  os.chdir(Project_folder)
except:
  drive.mount('/content/drive', force_remount=True)
  os.chdir(Project_folder)
else:
  print("Drive ya está montado")
  print(os.getcwd())

!apt -y -qq install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg  -q -qq

!pip install pyaudio -qq -q

!pip install wave -qq -q

!pip install ffmpeg-python -qq -q
"""
"""
To write this piece of code I took inspiration/code from a lot of places.
It was late night, so I'm not sure how much I created or just copied o.O
Here are some of the possible references:
https://blog.addpipe.com/recording-audio-in-the-browser-using-pure-html5-and-minimal-javascript/
https://stackoverflow.com/a/18650249
https://hacks.mozilla.org/2014/06/easy-audio-capture-with-the-mediarecorder-api/
https://air.ghost.io/recording-to-an-audio-file-using-html5-and-js/
https://stackoverflow.com/a/49019356
"""
######
"""
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg

"""

######
AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
//my_p.appendChild(my_btn);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    //bitsPerSecond: 8000, //chrome seems to ignore, always 48k
    mimeType : 'audio/webm;codecs=opus'
    //mimeType : 'audio/webm;codecs=pcm'
  };
  //recorder = new MediaRecorder(stream, options);
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
      //console.log("Inside FileReader:" + base64data);
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  //console.log("Inside data:" + base64data)
  resolve(base64data.toString())

});

}
});

</script>
"""


######



"""

def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])

  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)

  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  riff = output[:4] + bytes(b) + output[8:]

  sr, audio = wav_read(io.BytesIO(riff))

  return audio, sr

audio, sr = get_audio()

import scipy
scipy.io.wavfile.write('recording.wav', sr, audio)

!pip install pyaudio

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output10.wav"



"""
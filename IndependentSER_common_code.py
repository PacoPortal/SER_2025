# -*- coding: utf-8 -*-
#CrossCorpus_2024Q4.Final_Model_paper.ipynb


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
DUMP_FOLDER_MODELS = '/home/x002/SCRATCH/models/'
TYPE_EXPERIMENT = 'cross'
#TYPE_EXPERIMENT = 'indep'



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
"""##Common functions and definitions"""

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        self.tsr = 16e3

    def __getitem__(self, idx):
        speech = self.file_to_array( self.data.path.values[idx] , self.tsr)
        return speech, self.data.labels.values[idx]

    def file_to_array(self, path, sampling_rate):
        array, sr = librosa.load(path, sr= sampling_rate, mono=True)
        return array

    def __len__(self):
        return len(self.data)

def collate_fn_padd(batch , feature_extractor):
    batch = np.array(batch, dtype=object)
    inputs = batch[:, 0]
    labels = batch[:, 1]
    """
    print("\n")
    print(list(inputs))
    print("\n")
    print(list(labels))
    """
    encodings = feature_extractor(list(inputs), sampling_rate=16000, padding=True, return_tensors="pt")
    encodings['labels'] = torch.tensor(list(labels))
    return encodings


def get_data_loader(train_dataset, feature_extractor, train_bs):
    collate_fn = lambda batch: collate_fn_padd(batch, feature_extractor = feature_extractor)
    train_dl = DataLoader(train_dataset , batch_size = train_bs, collate_fn = collate_fn, shuffle=True)
    return train_dl #, val_dl


class EmotionDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels

  def __getitem__(self, idx):
      item = {key: val[idx] for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item

  def __len__(self):
      return len(self.labels)

def predict(outputs):
  probabilities = torch.softmax(outputs["logits"], dim=1)
  predictions = torch.argmax(probabilities, dim=1)
  return predictions

def plot_loss_acc(train_loss, train_accuracies, round=''):
  
  # Plot Iteration vs Training Loss
  plt.plot(train_loss, label="Training Loss")
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.title("Iteration vs Training Loss")
  plt.legend()
  #plt.savefig("TrainingLoss.png", dpi=300)#plt.show()
  plt.savefig(round+"_TrainingLoss_tight.png", dpi=300, bbox_inches='tight')#plt.show()
  plt.close()

  # Plot Epoch vs Training Accuracy
  acc_X = np.arange(len(train_accuracies))+1
  plt.plot(acc_X, train_accuracies,"-", label="Training Accuracy")
  plt.xticks(acc_X)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("Epoch vs Training Accuracy")
  plt.legend()
  #plt.savefig("TrainingAccuracy.png", dpi=300)#plt.show()
  plt.savefig(round+"_TrainingAccuracy_tight.png", dpi=300, bbox_inches='tight')#plt.show()
  plt.close()

def data_split(dataset,
               #name_dump ,
               frac=0.2):
  test_df = dataset.groupby("emotions").sample(frac=frac,random_state=19730309)
  train_df = dataset.drop(test_df.index)
  #os.chdir(DUMP_FOLDER)
  #pickle.dump(test_df,  open(name_dump, 'wb'))
  #os.chdir(DATASETS_FOLDER)
  return train_df, test_df


def train_the_model_att_mask(model, model_name, optim, train_dataset, test_dataset, val_dataset, 
                    tms,
                    feature_extractor=None,
                    epoch_ini=1,#the first one to be done, =1 begins in 1-1=0, so =6 begins in 6-1 =5 eq 6
                    epochs=3, #total to be done
                    batch_size=2, dump_folder=None,
                    train_loss = list(),
                    train_accuracies = list(),
                    val_loss = list(),
                    val_acc = list(),
                    best_acc = 0):

  val_loss = []
  val_acc = []
  #tms = time.strftime("%Y%m%d%H%M%S", time.localtime())
  #os.chdir(DATASETS_FOLDER)
  print('Datetime : ',datetime.datetime.now())
  model.train()
  epoch_ini=epoch_ini-1
  epochs=epochs-epoch_ini # total minus the already done
  RANDOM_SEED = 19730309
  torch.manual_seed(RANDOM_SEED)
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  for epoch_i in range(epochs):
    print('Epoch %s/%s' % (epoch_i + 1 + epoch_ini, epochs+epoch_ini))
    os.chdir(DATASETS_FOLDER)
    if feature_extractor:
      train_loader = get_data_loader(train_dataset , feature_extractor, train_bs=batch_size)
      test_loader = get_data_loader(test_dataset, feature_extractor, train_bs=1 )
      val_loader = get_data_loader(val_dataset, feature_extractor, train_bs=1 )
    else:
      print('¿¿ POR QUÉ PASA POR AQUÍ ?? Buscar esto en el código ')
      train_loader = DataLoader(train_dataset,    batch_size,        shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
      val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    correct = 0
    count = 0
    epoch_loss = list()
    pbar = tqdm(train_loader)
    for batch in pbar:
      optim.zero_grad()
      input_ids = batch['input_values'].to(DEVICE)
      attention_mask = batch['attention_mask'].to(DEVICE)
      labels = batch['labels'].to(DEVICE)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs['loss']
      loss.backward()
      optim.step()

      predictions = predict(outputs)

      correct += predictions.eq(labels).sum().item()
      count += len(labels)
      baccuracy = correct * 1.0 / count

      pbar.set_postfix({
          'Loss': '{:.3f}'.format(loss.item()),
          'Batch Accuracy': '{:.3f}'.format(baccuracy)
      })

      epoch_loss.append(loss.item())
      #sys.stdout.flush()
      #sys.stderr.flush()


    train_loss += epoch_loss
    train_accuracies.append(baccuracy) #batch acc, with bs = 2 be careful


    # Save checkpoint
    print("Validation after epoch...")
    val_acc_item, val_loss_item, _ = testing_att_mask(val_loader, model, verbose=False) # _ is for record, not used
    print('Val acc: ', val_acc_item, ' . Val loss: ',val_loss_item)
    val_loss.append(val_loss_item)
    val_acc.append(val_acc_item)
    # pickle save loss accuracy
    train_log = (val_acc, val_loss)
    os.chdir(dump_folder)
    #pickle.dump(train_log,  open(model_name+'_log.p', 'wb'))
    #pickle.dump(train_log,  open(tms+'_h_'+model_name[:-2]+'_log.p', 'wb'))

    pickle.dump(train_log,  open(tms+'.'+model_name+'_log.p', 'wb'))
    if val_acc_item > best_acc:
      best_acc = val_acc_item
      tmp_name = tms+'.'+model_name
      print("Storing epoch: "+str(epoch_i+1+epoch_ini)+" with val_accuracy: ",val_acc_item )
      save_checkpoint(epoch_i+epoch_ini, model, optim, train_loss, train_accuracies, tmp_name, DUMP_FOLDER_MODELS)#dump_folder)
      #shutil.copy(tmp_name, "model.tmp")
    #else:
    save_checkpoint(epoch_i+epoch_ini, model, optim, train_loss, train_accuracies, "model.tmp", DUMP_FOLDER_MODELS)#dump_folder)
    TrainParams = (tms,epoch_i + 2 + epoch_ini, best_acc)
    pickle.dump(TrainParams,  open(dump_folder+'Train_State.p', 'wb'))
    #os.chdir(DATASETS_FOLDER)
    pbar.close()
    '''if val_acc_item > 0.9999:
      print('Break!')
      break'''
  return val_acc, val_loss, tmp_name

def train_the_model(model, model_name, optim, train_dataset, test_dataset, val_dataset, 
                    tms,
                    feature_extractor=None,
                    epoch_ini=1,#the first one to be done, =1 begins in 1-1=0, so =6 begins in 6-1 =5 eq 6
                    epochs=3, #total to be done
                    batch_size=2, dump_folder=None,
                    train_loss = list(),
                    train_accuracies = list(),
                    val_loss = list(),
                    val_acc = list(),
                    best_acc = 0):
  

  print('Datetime: ', datetime.datetime.now())
  model.train()
  epoch_ini=epoch_ini-1
  epochs=epochs-epoch_ini # total minus the already done
  RANDOM_SEED = 19730309
  torch.manual_seed(RANDOM_SEED)
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  for epoch_i in range(epochs):
    print('Epoch %s/%s' % (epoch_i + 1 + epoch_ini, epochs+epoch_ini))
    os.chdir(DATASETS_FOLDER)
    if feature_extractor:
      train_loader = get_data_loader(train_dataset , feature_extractor, train_bs=batch_size)
      test_loader = get_data_loader(test_dataset, feature_extractor, train_bs=1 )
      val_loader = get_data_loader(val_dataset, feature_extractor, train_bs=1 )
    else:
      train_loader = DataLoader(train_dataset,    batch_size,        shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
      val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    correct = 0; count = 0
    epoch_loss = list()
    pbar = tqdm(train_loader)
    for batch in pbar:
      optim.zero_grad()
      input_ids = batch['input_values'].to(DEVICE)
      #attention_mask = batch['attention_mask'].to(DEVICE)
      labels = batch['labels'].to(DEVICE)
      outputs = model(input_ids,  labels=labels)
      loss = outputs['loss']
      loss.backward()
      optim.step()

      predictions = predict(outputs)

      correct += predictions.eq(labels).sum().item()
      count += len(labels)
      baccuracy = correct * 1.0 / count

      pbar.set_postfix({
          'Loss': '{:.3f}'.format(loss.item()),
          'Batch Accuracy': '{:.3f}'.format(baccuracy)
      })

      epoch_loss.append(loss.item())

    train_loss += epoch_loss
    train_accuracies.append(baccuracy)

    # Save checkpoint
    print("Validation after epoch...")
    val_acc_item, val_loss_item = testing(val_loader, model, verbose=False)
    print('Val acc: ', val_acc_item, ' . Val loss: ',val_loss_item)
    val_loss.append(val_loss_item)
    val_acc.append(val_acc_item)
    # pickle save loss accuracy
    train_log = (val_acc, val_loss)
    os.chdir(dump_folder)
    #pickle.dump(train_log,  open(model_name+'_log.p', 'wb'))
    #pickle.dump(train_log,  open(tms+'_w_'+model_name[:-2]+'_log.p', 'wb'))
    pickle.dump(train_log,  open(tms+'.'+model_name+'_log.p', 'wb'))
    #os.chdir(DATASETS_FOLDER)
    if val_acc_item > best_acc:
      best_acc = val_acc_item
      tmp_name = tms+'.'+model_name
      print("Storing epoch: "+str(epoch_i+1+epoch_ini)+" with val_accuracy: ",val_acc_item )
      save_checkpoint(epoch_i+epoch_ini, model, optim, train_loss, train_accuracies, tmp_name, DUMP_FOLDER_MODELS)#dump_folder)
     #shutil.copy(tmp_name, "model.tmp")
    #else:
    save_checkpoint(epoch_i+epoch_ini, model, optim, train_loss, train_accuracies, "model.tmp", DUMP_FOLDER_MODELS)#dump_folder)
    TrainParams = (tms,epoch_i + 2 + epoch_ini, best_acc)
    pickle.dump(TrainParams,  open(dump_folder+'Train_State.p', 'wb'))
    #os.chdir(DATASETS_FOLDER)
    pbar.close()
    '''if val_acc_item > 0.9999:
      print('Break!')
      break'''
  return val_acc, val_loss, tmp_name



def testing(test_loader, model, verbose = True):
  #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  model.eval()
  os.chdir(DATASETS_FOLDER)
  with torch.no_grad():
      correct = 0.0
      count = 0
      running_loss = 0.0
      record = {"labels":list(), "predictions":list()}
      if verbose:
        pbar = tqdm(test_loader)
      else:
        pbar = test_loader
      for batch in pbar:
          input_ids = batch['input_values'].to(DEVICE)
          #attention_mask = batch['attention_mask'].to(DEVICE)
          labels = batch['labels'].to(DEVICE)
          outputs = model(input_ids, labels=labels)
          loss = outputs['loss']
          predictions = predict(outputs)

          count += len(labels)
          #print('Valor los  por item en validación: ',loss.item()
          running_loss += loss.item()*len(labels)
          val_loss = running_loss / count
          correct += predictions.eq(labels).sum().item()
          val_accuracy = correct / count #correct * 1.0 / count

          if verbose:
            pbar.set_postfix({
              'loss': '{:.3f}'.format(loss.item()),
              'accuracy': '{:.3f}'.format(val_accuracy)
            })

          record["labels"] += labels.cpu().numpy().tolist()
          record["predictions"] += predictions.cpu().numpy().tolist()
      if verbose:
        pbar.close()

  if verbose:
    print("The final accuracy on the test dataset: %s%%" % round(val_accuracy*100,4))
    return record
  else:
    model.train()
    return val_accuracy, val_loss



def testing_att_mask(test_loader, model, verbose = True):
  #test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  model.eval()
  os.chdir(DATASETS_FOLDER)
  with torch.no_grad():
      correct = 0.0
      count = 0
      running_loss = 0.0
      record = {"labels":list(), "predictions":list()}
      if verbose:
        pbar = tqdm(test_loader)
      else:
        pbar = test_loader
      for batch in pbar:
          input_ids = batch['input_values'].to(DEVICE)
          attention_mask = batch['attention_mask'].to(DEVICE)
          labels = batch['labels'].to(DEVICE)
          outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs['loss']
          predictions = predict(outputs)

          count += len(labels)
          running_loss += loss.item()*len(labels)
          val_loss = running_loss / count
          correct += predictions.eq(labels).sum().item()
          val_accuracy = correct / count

          if verbose:
            pbar.set_postfix({
              'loss': '{:.3f}'.format(loss.item()),
              'accuracy': '{:.3f}'.format(val_accuracy)
            })

          record["labels"] += labels.cpu().numpy().tolist()
          record["predictions"] += predictions.cpu().numpy().tolist()
      if verbose:
        pbar.close()

  if verbose:
    print("The final accuracy on the test dataset: %s%%" % round(val_accuracy*100,4))
    return record
  else:
    model.train()
    return val_accuracy, val_loss, record # added record in orther to get confusion matrix. Initially only val_accuracy, val_loss


def load_checkpoint(DEVICE_name, model, optim, model_name, dump_folder):
  #load the model
  os.chdir(dump_folder)
  if DEVICE_name == 'cpu':
    checkpoint = torch.load(model_name+'.model', map_location=torch.device('cpu'))
  else:
    checkpoint = torch.load(model_name+'.model')
  epoch = checkpoint['epoch']
  model.load_state_dict(checkpoint['model_state_dict'])
  optim.load_state_dict(checkpoint['optimizer_state_dict'])
  train_loss = checkpoint['train_loss']
  train_accuracies = checkpoint['train_accuracies']
  #os.chdir(DATASETS_FOLDER)
  return model, epoch, optim, train_loss, train_accuracies


def save_checkpoint(epoch_i, model, optim, train_loss, train_accuracies, model_name, dump_folder):
 # Save checkpoint
    os.chdir(dump_folder)
    torch.save({
      'epoch': epoch_i,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optim.state_dict(),
      'train_loss':train_loss,
      'train_accuracies':train_accuracies
      }, model_name+'.model')
    #os.chdir(DATASETS_FOLDER)# quitar en algun momento y asegurarse de que cuando se usa datasetsfolder se hace su chdir

def incorrect_ones(record,test_df):
  df_record = DataFrame(record)
  df_record.columns = ["Ground Truth","Model Prediction"]
  # Concat test texts and test records
  df = pd.concat([test_df.reset_index(), df_record["Model Prediction"]], axis=1)
  df["emotions"] = df.apply(lambda x: x["emotions"][:3], axis=1)
  # Show test result
  pd.set_option('display.max_rows', None)    # Display all rows
  # Show incorrect predictions
  incorrect = df[df["labels"]!=df["Model Prediction"]]
  #print(incorrect[["emotions","labels","Model Prediction"]])
  return df_record

def incorrect_ones2(record,test_df):
  def get_emotions(labels_id):
    return model.config.id2label[labels_id]
  df_record = DataFrame(record)
  df_record.columns = ["Ground Truth","Model Prediction"]
  df_record["Ground Truth"] = df_record.apply(lambda x: get_emotions(x["Ground Truth"]), axis=1)
  df_record["Model Prediction"] = df_record.apply(lambda x: get_emotions(x["Model Prediction"]), axis=1)
  # Concat test texts and test records
  df = pd.concat([test_df.reset_index(), df_record["Model Prediction"]], axis=1)
  df["emotions"] = df.apply(lambda x: x["emotions"][:3], axis=1)
  # Show test result
  pd.set_option('display.max_rows', None)    # Display all rows
  # Show incorrect predictions
  print(df[df["emotions"]!=df["Model Prediction"]])
  return df_record

def display_confusion_matrix(df_record,secondName='',id2label=None):
  # Display the Confusion Matrix
  if id2label!=None:
    df_record["Model Prediction"] = df_record.apply(lambda x: id2label[str(x["Model Prediction"])][:3], axis=1)
    df_record["Ground Truth"] = df_record.apply(lambda x: id2label[str(x["Ground Truth"])][:3], axis=1)

  print("Confusion matrix of the result: ")
  crosstab = pd.crosstab(df_record["Ground Truth"],df_record["Model Prediction"])
  crosstab = crosstab.astype('float') / crosstab.sum(axis=1)
  sns.heatmap(crosstab, cmap='Oranges', annot=True,   fmt='.1%', linewidths=5, cbar=False) 
  accuracy = df_record["Ground Truth"].eq(df_record["Model Prediction"]).sum() / len(df_record["Ground Truth"])
  plt.gca().xaxis.tick_top()
  print("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100,2))
  #plt.title("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100,2))
  #plt.savefig(secondName+"ConfusionMatrix.png", dpi=300)#plt.show()
  plt.savefig(secondName+"ConfusionMatrix_tight.png", dpi=300, bbox_inches='tight')#plt.show()
  plt.title("Confusion Matrix (Accuracy: %s%%)" % round(accuracy*100,2))
  plt.savefig(secondName+"ConfusionMatrix_tight_title.png", dpi=300, bbox_inches='tight')#plt.show()
  plt.close()
  '''
 # Method 3: Visualizing the confusion matrix with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_true), 
            yticklabels=np.unique(y_true))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Method 4: More detailed visualization with percentages
plt.figure(figsize=(8, 6))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=np.unique(y_true), 
            yticklabels=np.unique(y_true))
plt.title('Confusion Matrix (Percentages)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
'''






def data_distribution(df):
    classes_names = np.unique(df['emotions'].values)
    values = [df['emotions'].values.tolist().count(class_) for class_ in classes_names] # frequency
    plt.figure( figsize=(5 , 3)  , dpi=100 )
    plt.bar(classes_names , values)
    plt.xticks(classes_names , size=9)
    plt.xlabel('Class', size=12)
    plt.ylabel('Frequency', size=12)
    plt.title('Class Distribution of Dataset', size=13)
    plt.savefig("DistributionD.png", dpi=300)#plt.show()
    plt.savefig("DistributionD_tight.png", dpi=300, bbox_inches='tight')#plt.show()
    plt.close()

import itertools
import inspect

def get_nvar(variable):
  """Imprime el nombre de una variable."""
  nombre_variable = list(locals().keys())[list(locals().values()).index(variable)]
  #print(f"El nombre de la variable es: {nombre_variable}")
  return nombre_variable

def combinaciones_sin_repeticion(lista):
  """
  Genera todas las combinaciones sin repetición de los elementos de una lista,
  tomando desde 2 hasta 5 elementos a la vez.

  Args:
    lista: La lista de elementos.

  Returns:
    Una lista de tuplas, donde cada tupla representa una combinación.
  """

  combinaciones = []
  for r in range(2, len(lista)):  # Combinaciones de 2 a n elementos
    for combinacion in itertools.combinations(lista, r):
      combinaciones.append(combinacion)
  return combinaciones


def loop_with_att_mask(checkpoints, train_dataset, test_dataset, val_dataset, DATASET_n,
                       label2id, id2label, num_labels, LR, WEIGHT_DECAY, tms, dump_folder,
                       N_EPOCHS = 20, epoch_ini=1, best_acc = 0, datasets_test=[], freeze=False, testlabel='NotGiven'):
  #print(checkpoints);   print(DATASET_n);   print(num_labels)
  for MODEL_pre in checkpoints:
    if MODEL_pre==None:
      print('Break!')
      break
    print('Checkpoint :',MODEL_pre)
    MODEL_NAME = DATASET_n + '.' + MODEL_pre.rsplit('/', 1)[1] #eliminamos superb, facecbook...
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_pre, return_attention_mask=True)
    print("Feature extractor is: ",feature_extractor.__class__.__name__)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_pre, ignore_mismatched_sizes=True,
                                                            num_labels=num_labels, label2id=label2id, id2label=id2label)
    model.to(DEVICE)
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if epoch_ini>1:
      model, epoch, optim, train_loss, train_accuracies = load_checkpoint(DEVICE, model, optim, "model.tmp", DUMP_FOLDER_MODELS)#dump_folder)
      #model, epoch, optim, train_loss, train_accuracies = load_checkpoint(DEVICE, model, optim, MODEL_NAME, DUMP_FOLDER_MODELS)#dump_folder)
    print('Freeze: ', freeze)
    if freeze==True:
      model.freeze_feature_encoder();     
    model.to(DEVICE) ;     
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    val_acc, val_loss, tmp_name =  train_the_model_att_mask(model, MODEL_NAME, optim, train_dataset, test_dataset, val_dataset, tms,
                                                  feature_extractor=feature_extractor,
                                                  epochs=N_EPOCHS,batch_size=2,dump_folder=dump_folder,epoch_ini=epoch_ini,best_acc = best_acc)

    model, epoch, optim, train_loss, train_accuracies = load_checkpoint(DEVICE, model, optim, tmp_name, DUMP_FOLDER_MODELS)#dump_folder)
    model.to(DEVICE)
    print('Calculating test phase values... ')
    test_loader = get_data_loader(test_dataset, feature_extractor, train_bs=1 )
    accuracy, test_loss, record = testing_att_mask(test_loader, model, verbose=False)
    print("Test accuracy of "+MODEL_pre+" over "+ DATASET_n +"on test set "+ testlabel+" in "+str(N_EPOCHS)+" epochs is: ", accuracy) 
    print()
    for dataset_df in datasets_test:
      valds=SpeechDataset(dataset_df)
      test_loader = get_data_loader(valds, feature_extractor, train_bs=1 )
      accuracy = testing_att_mask(test_loader, model, verbose=True) # shouldnt work, seems not goes though here never
      print("Accuracy of "+MODEL_pre+" over "+ dataset_df.source[0] +" is: ", accuracy); print()

    #if 'model' in globals() or 'model' in locals():
    #  del model; gc.collect(); print("Model memory released")

  return val_acc, val_loss, accuracy, test_loss, record, tmp_name, model 


def loop(checkpoints, train_dataset, test_dataset, val_dataset, DATASET_n,
                       label2id, id2label, num_labels, LR, WEIGHT_DECAY, tms, dump_folder,
                       N_EPOCHS = 20, epoch_ini=1, best_acc = 0, model_name_bis=None, datasets_test=[],freeze=False, testlabel='NotGiven'):

  for MODEL_pre in checkpoints:

    if MODEL_pre==None:
      print('Break!')
      break
    print('Checkpoint :',MODEL_pre)
    MODEL_NAME = DATASET_n + '.' + MODEL_pre.rsplit('/', 1)[1] #eliminamos superb, facecbook...

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_pre)
    print("Feature extractor is: ",feature_extractor.__class__.__name__)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_pre, ignore_mismatched_sizes=True,
                                                            num_labels=num_labels, label2id=label2id, id2label=id2label)
    model.to(DEVICE)
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if epoch_ini>1:
      model, epoch, optim, train_loss, train_accuracies = load_checkpoint(DEVICE, model, optim, "model.tmp", DUMP_FOLDER_MODELS)#dump_folder)
    #model.freeze_feature_encoder() ;     #model.to(DEVICE) ;     #optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print('Freeze: ', freeze)
    if freeze==True:
      model.freeze_feature_encoder();     
    model.to(DEVICE) ;     
    optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    val_acc, val_loss, tmp_name =  train_the_model(model, MODEL_NAME, optim, train_dataset, test_dataset, val_dataset, tms,
                                                  feature_extractor=feature_extractor,
                                                  epochs=N_EPOCHS,batch_size=2,dump_folder=dump_folder,epoch_ini=epoch_ini,best_acc = best_acc)

    model, epoch, optim, train_loss, train_accuracies = load_checkpoint(DEVICE, model, optim, tmp_name,DUMP_FOLDER_MODELS)#dump_folder)
    model.to(DEVICE)
    print('Calculating test phase values... ')
    test_loader = get_data_loader(test_dataset, feature_extractor, train_bs=1 )
    accuracy, test_loss = testing(test_loader, model, verbose=False)
    print("Test accuracy of "+MODEL_pre+" over "+ DATASET_n +"on test set "+ testlabel+" in "+str(N_EPOCHS)+" epochs is: ", accuracy) 
    print()
    for dataset_df in datasets_test:
      valds=SpeechDataset(dataset_df)
      test_loader = get_data_loader(valds, feature_extractor, train_bs=1 )
      accuracy = testing(test_loader, model, verbose=True)
      print("Accuracy of "+MODEL_pre+" over "+ dataset_df.source[0] +" is: ", accuracy); print()

    if 'model' in globals() or 'model' in locals():
      del model
      gc.collect()
      print("Model memory released")

  return val_acc, val_loss, accuracy, test_loss





def plot_loss(train_loss):
  # Plot Iteration vs Training Loss
  plt.plot(train_loss, label="Val Loss")
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.title("Epoch vs Val Loss")
  plt.legend()
  plt.savefig("Loss.png", dpi=300, bbox_inches='tight')#plt.show()
  plt.close()

def plot_acc(train_accuracies):
  # Plot Epoch vs Training Accuracy
  acc_X = np.arange(len(train_accuracies))+1
  plt.plot(acc_X, train_accuracies,"-", label="Val Accuracy")
  plt.xticks(acc_X)
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.title("Epoch vs  Val Accuracy")
  plt.legend()
  plt.savefig("Accuracy.png", dpi=300, bbox_inches='tight')#plt.show()
  plt.close()
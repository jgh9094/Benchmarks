'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

This file will distill the knowledge from teachers into a Multi-task CNN.
'''

# general python imports
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import pandas as pd
import pickle as pk

# keras python inputs
from keras.models import Model
from keras.layers import Input,Embedding,Dropout,Dense,GlobalMaxPooling1D,Conv1D,Activation,Lambda,concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical,plot_model
from keras.losses import categorical_crossentropy as logloss
from keras import backend as K
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy

# global variables
EPOCHS = 100
SPLIT = 0
TEMP = 0
ALPHA = 0.0

# return configuration for the experiment
def GetModelConfig(config):
  # testing configuration
  if config == 0:
    return {
      'learning_rate': 0.01,
      'batch_size': 5,
      'dropout': 0.5,
      'optimizer': 'adam',
      'wv_len': 300,
      'emb_l2': 0.001,
      'in_seq_len': 1500,
      'num_filters': 100,
      'filter_sizes': 3,
      'model_N': 4,
      'alpha': 0.5,
      'temp': 7.0
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# compute
def knowledge_distillation_loss(y_true, y_pred):
  # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
  y_true, y_true_softs = y_true[: , :SPLIT], y_true[: , SPLIT:]
  y_pred, y_pred_softs = y_pred[: , :SPLIT], y_pred[: , SPLIT:]

  diff_alpha = 1 - ALPHA

  loss = ALPHA * logloss(y_true,y_pred) +  diff_alpha * logloss(y_true_softs, y_pred_softs)

  return loss

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(d_dir,t_dir):
  # load data
  rawX = np.load( d_dir + 'train_X.npy' )
  rawY = np.load( d_dir + 'train_Y.npy' )
  rawXT = np.load( d_dir + 'test_X.npy' )
  rawYT = np.load( d_dir + 'test_Y.npy' )

  # raw data descriptions
  print('RAW DATA DIMS')
  print('rawX dim: ', rawX.shape)
  print('rawY dim: ', rawY.shape)
  print('rawXT dim: ', rawXT.shape)
  print('rawYT dim: ', rawYT.shape , end='\n\n')

  if rawY.shape[1] != rawYT.shape[1]:
    print('NUMBER OF TASKS NOT THE SAME BETWEEN DATA SETS')
    exit(-1)

  # create array for each task output
  y = [[] for i in range(rawY.shape[1])]
  yt = [[] for i in range(rawY.shape[1])]
  # load data
  for t in range(rawY.shape[1]):
    y[t] = rawY[:,t]
    yt[t] = rawYT[:,t]

  # make to catagorical data and pack up
  tempY,tempYT = [],[]
  for t in y:
    tempY.append(to_categorical(t))
  for t in yt:
    tempYT.append(to_categorical(t))

  print('Temp Training Output Data')
  i = 0
  for y in tempY:
    print('task', i)
    print('--cases:', len(y))
    print('--classes:',len(y[0]))
    i += 1
  print()

  print('Temp Testing Output Data')
  i = 0
  for y in tempYT:
    print('task', i)
    print('--cases:', len(y))
    print('--classes:',len(y[0]))
    i += 1
  print()

  # number of classes per task
  classes = []
  for y in tempY:
    classes.append(len(y[0]))

  # append teacher softmax outputs to the hard label output
  Y,YT = [],[]
  # number of tasks dictates number of output files expecting
  for i in range(len(classes)):
    # load data
    file = t_dir + 'training-task-' + str(i) + '.npy'
    teach_y = np.load(file)
    file = t_dir + 'testing-task-' + str(i) + '.npy'
    teach_yt = np.load(file)

    # combine and save data
    y,yt = CombineData(tempY[i],tempYT[i],teach_y,teach_yt)
    Y.append(y)
    YT.append(yt)

  print('Training Output Data')
  i = 0
  for y in Y:
    print('task', i)
    print('--cases:', len(y))
    print('--classes:',len(y[0]))
    i += 1
  print()

  print('Testing Output Data')
  i = 0
  for y in YT:
    print('task', i)
    print('--cases:', len(y))
    print('--classes:',len(y[0]))
    i += 1
  print()

  return np.array(rawX),Y,np.array(rawXT),YT,classes


# combine the data output with ground truth and teacher logits
def CombineData(y,yt,ty,tyt):
  Y = []
  for i in range(len(y)):
    Y.append(np.concatenate((y[i],ty[i])))

  YT = []
  for i in range(len(yt)):
    YT.append(np.concatenate((yt[i],tyt[i])))

  return np.array(Y),np.array(YT)


def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str, help='Where is the data located?')
  parser.add_argument('tech_dir',     type=str, help='Where is the teacher data located?')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')
  parser.add_argument('seed',         type=int, help='Random seed for run')

  # Parse all the arguments & set random seed
  args = parser.parse_args()
  print('Seed:', args.seed, end='\n\n')
  np.random.seed(args.seed)

  # check that dump directory exists
  if not os.path.isdir(args.dump_dir):
    print('DUMP DIRECTORY DOES NOT EXIST')
    exit(-1)

  # Step 1: Get experiment configurations
  config = GetModelConfig(args.config)
  print('run parameters:', config, end='\n\n')

  # Step 2: Create training/testing data for models
  X,Y,XT,YT,classes =  GetData(args.data_dir, args.tech_dir)


if __name__ == '__main__':
  main()
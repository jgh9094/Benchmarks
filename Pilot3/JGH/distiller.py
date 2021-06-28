'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH
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

def accuracy(y_true, y_pred):
    y_true = y_true[:, :SPLIT]
    y_pred = y_pred[:, :SPLIT]
    return categorical_accuracy(y_true, y_pred)

def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :SPLIT]
    y_pred = y_pred[:, :SPLIT]
    return top_k_categorical_accuracy(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :SPLIT]
    y_pred = y_pred[:, :SPLIT]
    return logloss(y_true, y_pred)

# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):
    logits = y_true[:, SPLIT:]
    y_soft = K.softmax(logits/TEMP)
    y_pred_soft = y_pred[:, SPLIT:]
    return logloss(y_soft, y_pred_soft)

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
      'const': 0.7
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# compute
def knowledge_distillation_loss(y_true, y_pred, lambda_const):

  # split in
  #    onehot hard true targets
  #    logits from xception
  y_true, logits = y_true[:, :SPLIT], y_true[:, SPLIT:]

  # convert logits to soft targets
  y_soft = K.softmax(logits/TEMP)

  # split in
  #    usual output probabilities
  #    probabilities made softer with temperature
  y_pred, y_pred_soft = y_pred[:, :SPLIT], y_pred[:, SPLIT:]

  print('*******')
  print(logits)
  print(lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft))
  print('(((((((((')

  return lambda_const*logloss(y_true, y_pred, from_logits=False) + logloss(y_soft, y_pred_soft, from_logits=False)

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(dir,N):
  # load data
  trainX = np.load( dir + 'train_X.npy' )
  trainY = np.load( dir + 'train_Y.npy' )[ :, 0 ]
  testX = np.load( dir + 'test_X.npy' )
  testY = np.load( dir + 'test_Y.npy' )[ :, 0 ]

  # find max class number and adjust test/training y
  return np.array(trainX), np.array(to_categorical(trainY)), np.array(testX), np.array(to_categorical(testY))

# combine the data output with ground truth and teacher logits
def CombineData(y,yt,ty,tyt):
  Y = []
  for i in range(len(y)):
    Y.append(np.concatenate((y[i],ty[i])))

  YT = []
  for i in range(len(yt)):
    YT.append(np.concatenate((yt[i],tyt[i])))

  return np.array(Y),np.array(YT)

# create student model
def CreateStudent(x,y,xT,yT,cfg,em_max):
  # word vector lengths
  wv_mat = np.random.randn( em_max + 1, cfg['wv_len'] ).astype( 'float32' ) * 0.1
  # validation data
  validation_data = (xT,yT)
  # stopping criterion
  stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)

  # set input layer, assuming that all input will have same shape as starting case
  input = Input(shape=([x.shape[1]]), name= "Input")
  # embedding lookup
  embed = Embedding(len(wv_mat), cfg['wv_len'], input_length=cfg['in_seq_len'], name="embedding",
                      embeddings_regularizer=l2(cfg['emb_l2']))(input)
  # convolutional layer
  conv = Conv1D(filters=cfg['num_filters'], kernel_size=cfg['filter_sizes'], padding="same",
                  activation="relu", strides=1, name="filter")(embed)
  # max pooling layer
  pooling = GlobalMaxPooling1D()(conv)
  #  drop out layer
  concat_drop = Dropout(cfg['dropout'])(pooling)
  # dense (output) layer
  dense = Dense(y.shape[1]/2, name= "Dense")( concat_drop )

  # hard probabilities
  probabilities = Activation('softmax')(dense)
  # softed probabilities
  logits_T = Lambda(lambda x: x/TEMP)(dense)
  probabilities_T = Activation('softmax')(logits_T)

  # final output layer
  outlayer = concatenate([probabilities, probabilities_T])

  # link, compile, and fit model
  model = Model(inputs=input, outputs = outlayer)

  return model


def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('config',      type=int, help='What kd model config are we using?')
  parser.add_argument('data_dir',    type=str, help='Where is the data located?')
  parser.add_argument('teach_dir',   type=str, help='Where is the student data located?')
  parser.add_argument('modl_dir',    type=str, help='Where are the models located?')
  parser.add_argument('dump_dir',    type=str, help='Where are we dumping the output?')
  parser.add_argument('seed',        type=int, help='Random seed for run')

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

  print('TEACHER STATS')

  # Step 2: Create training/testing data for ensemble model
  xTrain,yTrain,xTest,yTest =  GetData(args.data_dir, config['model_N'])
  global SPLIT
  SPLIT = len(yTrain[0])

  # get the teacher training/testing outputs
  file = open('./Model-1/training_X.pickle', 'rb')
  ttrain_X = pk.load(file)
  file.close
  file = open('./Model-1/test_X.pickle', 'rb')
  ttest_X = pk.load(file)
  file.close

  print(ttrain_X.shape)
  print(ttest_X.shape)
  print(ttrain_X[0])

  yTrain,yTest = CombineData(yTrain,yTest,ttrain_X,ttest_X)

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  # Step 3: Create, compile, train student model
  student = CreateStudent(xTrain, yTrain, xTest, yTest, config, max(np.max(xTrain), np.max(xTest)))
  plot_model(student,to_file= args.dump_dir + 'teacher.png',show_shapes=True, show_layer_names=True)

  student.compile(optimizer= config['optimizer'],
                  loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, config['const']),
                  metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss] )

  # validation data
  validation_data = (xTest,yTest)
  student.fit(xTrain,yTrain, batch_size=config['batch_size'],epochs=EPOCHS, verbose=2, validation_data=validation_data,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)])




if __name__ == '__main__':
  main()
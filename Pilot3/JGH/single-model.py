'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

Python file will focus on creating single models that will later be used
to construct an ensemble network model.
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
from keras.layers import Input, Embedding, Dense, Dropout
from keras.regularizers import l2
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# global variables
EPOCHS = 100

# return the data for training and testing
# will need to modify if other means of data gathering
# USED FOR LOCAL TESTING PURPOSES: EXPECTING DATA TO BE ON LOCAL MACHINE
def GetData(dir,task):
  # load data
  trainX = np.load( dir + 'train_X.npy' )
  trainY = np.load( dir + 'train_Y.npy' )[ :, task ]
  testX = np.load( dir + 'test_X.npy' )
  testY = np.load( dir + 'test_Y.npy' )[ :, task ]

  # find max class number and adjust test/training y
  return np.array(trainX), np.array(to_categorical(trainY)), np.array(testX), np.array(to_categorical(testY))

# transform data into expected format
def TransformData(X, XV, XT, Y, YV, YT):
  return np.array(X), np.array(XV), np.array(XT), np.array(to_categorical(Y)), np.array(to_categorical(YV)), np.array(to_categorical(YT))

# return configuration for the experiment
def GetModelConfig(config):
  # testing configuration
  if config == 0:
    return {
      'learning_rate': 0.01,
      'batch_size': 10,
      'dropout': 0.5,
      'optimizer': 'adam',
      'wv_len': 300,
      'emb_l2': 0.001,
      'in_seq_len': 1500,
      'num_filters': [300,300,300,300,300],
      'filter_sizes': [3,4,4,5,5],
      'task': 0
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# return task we are doing
def GetTask(t):
  if t == 0:
    return'Behavior'
  elif t == 1:
    return 'Histology'
  elif t == 2:
    return 'Laterality'
  elif t == 3:
    return 'Site'
  elif t == 4:
    return 'Subsite'
  else:
    print('UNKNOWN TASK')
    exit(-1)

# @Kevin todo
def loadSingleTask():

  return 0

# Create models, save them and their outputs
# x,y: training input, output
# xV,yV: testing input, output
# cfg: configuration we are using for this experiment
# em_max: embedding layer maximum
def BasicModel(x,y,xV,yV,cfg,cid,em_max):
  print('Using Configuration ID:', cid)

  # word vector lengths
  # wv_mat = np.random.randn( em_max, cfg['wv_len'] ).astype( 'float32' ) * 0.1
  # validation data
  validation_data = ( { 'Input-' + str(cid+1): xV }, {'Dense-' + str(cid+1) : yV})

  # set input layer, assuming that all input will have same shape as starting case
  input = Input(shape=([x.shape[1]]), name= "Input-" + str(cid+1))
  # embedding lookup
  embed = Embedding(em_max, cfg['wv_len'], input_length=cfg['in_seq_len'], name="embedding-"+ str(cid+1),
                      embeddings_regularizer=l2(cfg['emb_l2']))(input)
  # convolutional layer
  conv = Conv1D(filters=cfg['num_filters'][cid], kernel_size=cfg['filter_sizes'][cid], padding="same",
                  activation="relu", strides=1, name=str(cid+1) + "_thfilter")(embed)
  # max pooling layer
  pooling = GlobalMaxPooling1D()(conv)
  #  drop out layer
  concat_drop = Dropout(cfg['dropout'])(pooling)
  # dense (output) layer
  outlayer = Dense(y.shape[1], name= "Dense-"+str(cid+1), activation='softmax')( concat_drop )

  # link, compile, and fit model
  model = Model(inputs=input, outputs = outlayer)
  model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

  history = model.fit(x,y, batch_size=cfg['batch_size'],epochs=EPOCHS, verbose=2, validation_data=validation_data,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)])

  return history, model

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  # parser.add_argument('data_dir',     type=str, help='Where is the data located?')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')
  parser.add_argument('config_id',    type=int, help='What model config id are we using?')
  parser.add_argument('seed',         type=int, help='Random seed for run')
  parser.add_argument('task',         type=int, help='What task are we grabbing')

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

  # Step 2: Get and transform training/testing data for models
  # X,Y,XT,YT =  GetData(args.data_dir,args.task)
  X, XV, XT, Y, YV, YT = loadSingleTask(GetTask(args.task),print_shapes = True)
  X, XV, XT, Y, YV, YT = TransformData(X, XV, XT, Y, YV, YT)

  # quick descriptors of the data
  print('X dim: ', X.shape)
  print('Y dim: ', Y.shape)
  print('XV dim: ', XV.shape)
  print('YV dim: ', YV.shape)
  print('XT dim: ', XT.shape)
  print('YT dim: ', YT.shape , end='\n\n')

  # Step 3: Generate, create, and store  models
  hist, model = BasicModel(X, Y, XV, YV, config, args.config_id, max(np.max(X),np.max(XT)) + 1)

  # create directory to dump all training/testing data predictions
  fdir = args.dump_dir + 'Model-' + str(args.config) +'-' + str(args.config_id) + '/'
  os.mkdir(fdir)

  # saving training, testing, softmax values
  fp = open(fdir + 'training_X.pickle', 'wb')
  pk.dump(model.predict(X), fp)
  fp.close()
  print('Training X Predictions Saved!')

  fp = open(fdir + 'val_X.pickle', 'wb')
  pk.dump(model.predict(XV), fp)
  fp.close()
  print('Validation X Predictions Saved!')

  # save model predictions on testing data
  fp = open(fdir + 'test_X.pickle', 'wb')
  pk.dump(model.predict(XT), fp)
  fp.close()
  print('Testing X Predictions Saved!')

  # save history files
  df = pd.DataFrame(hist.history)
  df.to_csv(path_or_buf= fdir + 'history' + '.csv', index=False)
  print('History Saved!')

  # save model
  filename = fdir + 'model.h5'
  model.save(filename)
  print('Model Saved!')

  # save picture of model created
  plot_model(model, fdir + "model.png", show_shapes=True)
  print('Model Topology Picture Saved!')

if __name__ == '__main__':
  main()
'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

Python file will constuct an ensemble with the logit outputs from
N other smaller models. We then save those ensemble output and logits.
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
      'num_filters': 100,
      'filter_sizes': 3,
      'model_N': 4
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# return the data for training and testing
# data_d: data directory for correct lables
# stud_d: directory where all single models located
# config: configuration used for single models
# con_sz: number of models trained by configuration
def GetData(data_d,stud_d,config,con_sz):
  # Get model logit outputs for ensemble
  trainx = []
  trainxt = []

  for i in range(con_sz):
    # get the teacher training/testing outputs
    dir = stud_d + 'Model-' + str(config) + '-' + str(i) + '/'

    file = open(dir + 'training_X.pickle', 'rb')
    ttrain_X = pk.load(file)
    trainx.append(ttrain_X)
    file.close

    file = open(dir + 'test_X.pickle', 'rb')
    ttest_X = pk.load(file)
    trainxt.append(ttest_X)
    file.close

  X = []
  trainx = np.array(trainx)
  # loop through all data points
  for y in range(trainx.shape[1]):
    # loop through all model outputs
    row = []
    for x in range(trainx.shape[0]):
      row.append(trainx[x][y])
    row = np.concatenate(row)
    X.append(row)

  XT = []
  trainxt = np.array(trainxt)
  # loop through all data points
  for y in range(trainxt.shape[1]):
    # loop through all model outputs
    row = []
    for x in range(trainxt.shape[0]):
      row.append(trainxt[x][y])
    row = np.concatenate(row)
    XT.append(row)

  Y  = np.load( data_d + 'train_Y.npy' )[ :, 0 ]
  YT = np.load( data_d + 'test_Y.npy' )[ :, 0 ]


  # find max class number and adjust test/training y
  return np.array(X), np.array(to_categorical(Y)), np.array(XT), np.array(to_categorical(YT))

# create ensemble learner
def CreateEnsemble(x,y,xt,yt,cfg):

  # set input layer, assuming that all input will have same shape as starting case
  input = Input(shape=([x.shape[1]]), name= "Input")
  hidden = Dense(x.shape[1], activation='relu')(input)
  output = Dense(y.shape[1], activation='softmax')(hidden)

  # link, compile, and fit model
  model = Model(inputs=input, outputs = output)
  model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

  history = model.fit(x,y, batch_size=cfg['batch_size'],epochs=EPOCHS, verbose=2, validation_data=(xt,yt),
                      callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)])

  return history, model

def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for ensemble model.')
  parser.add_argument('data_dir',      type=str, help='Where is the teacher data located?')
  parser.add_argument('stud_dir',      type=str, help='Where is the student data located?')
  parser.add_argument('dump_dir',      type=str, help='Where are we dumping the output?')
  parser.add_argument('config',        type=int, help='Configuration used in single model?')
  parser.add_argument('config_sz',     type=int, help='Number of models that used configuration?')
  parser.add_argument('seed',          type=int, help='Random seed for run')

  # Parse all the arguments & set random seed
  args = parser.parse_args()
  print('Seed:', args.seed, end='\n\n')
  np.random.seed(args.seed)

  # check that data directory exists
  if not os.path.isdir(args.data_dir):
    print('DATA DIRECTORY DOES NOT EXIST')
    exit(-1)
  # check that student directory exists
  if not os.path.isdir(args.stud_dir):
    print('STUDENT DIRECTORY DOES NOT EXIST')
    exit(-1)
  # check that dump directory exists
  if not os.path.isdir(args.dump_dir):
    print('DUMP DIRECTORY DOES NOT EXIST')
    exit(-1)

  # Step 1: Get experiment configurations
  config = GetModelConfig(args.config)
  print('run parameters:', config, end='\n\n')

  # Step 2: Create training/testing data for ensemble model
  xTrain,yTrain,xTest,yTest =  GetData(args.data_dir, args.stud_dir, args.config, args.config_sz)

  # quick descriptors of the data
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  # Step 3: Create ensemble predictions
  hist,ensemble = CreateEnsemble(xTrain,yTrain,xTest,yTest,config)

  # create directory to dump all data related to model
  fdir = args.dump_dir + 'Ensemble-' + str(args.config) + '/'
  os.mkdir(fdir)

  # saving training, testing, softmax values
  fp = open(fdir + 'training_X.pickle', 'wb')
  pk.dump(ensemble.predict(xTrain), fp)
  fp.close()

  fp = open(fdir + 'test_X.pickle', 'wb')
  pk.dump(ensemble.predict(xTest), fp)
  fp.close()

  # convert the history.history dict to a pandas DataFrame:
  hist_df = pd.DataFrame(hist.history)
  hist_df.to_csv(path_or_buf= fdir + 'history.csv', index=False)
  print('History Saved!')
  # save model
  ensemble.save(fdir + 'model.h5')
  print('Model Saved!')
  # save picture of ensemble created
  plot_model(ensemble, fdir + "model.png", show_shapes=True)

if __name__ == '__main__':
  main()
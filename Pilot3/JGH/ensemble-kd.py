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

# keras python inputs
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.regularizers import l2
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers.merge import concatenate

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
      'num_filters': [300,300,300,300],
      'filter_sizes': [3,4,5,6],
      'model_N': 4
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(dir):
  # load data
  train_x = np.load( dir + 'train_X.npy' )
  train_y = np.load( dir + 'train_Y.npy' )[ :, 0 ]
  test_x = np.load( dir + 'test_X.npy' )
  test_y = np.load( dir + 'test_Y.npy' )[ :, 0 ]

  # find max class number and adjust test/training y
  train_y = to_categorical(train_y)
  test_y = to_categorical(test_y)

  return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

# load models from file
def load_models(cfg, dir):
  # collection of models
  models = list()

  # load all models
  for i in range(cfg['model_N']):
    # create file dir for each model
    filename = dir + 'Model-' + str(i+1) + '/model' + '.h5'
    # load model from file
    model = load_model(filename)
    # add to list of members
    models.append(model)
    print('Loaded:', filename)

  # update all layers in all models to not be trainable
  for i in range(len(models)):
    model = models[i]
    for layer in model.layers:
      # make not trainable
      layer.trainable = False
      # rename to avoid 'unique layer name' issue
      layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

  return models

# create ensemble learner
def CreateEnsemble(models,cfg,x,y,xT,yT):
  # stopping criterion
  stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)

  # define multi-headed input
  ensemble_visible = [model.input for model in models]
  # concatenate merge output from each model
  ensemble_outputs = [model.output for model in models]
  # model layers
  merge = concatenate(ensemble_outputs)
  hidden = Dense(y.shape[1] * cfg['model_N'], activation='relu')(merge)
  output = Dense(y.shape[1], activation='softmax')(hidden)
  # ensemble model
  ensembleM = Model(inputs=ensemble_visible, outputs=output)

  # validation data
  validation_data = [[xT for _ in range(len(ensembleM.input))], yT]

  # plot graph of ensemble
  plot_model(ensembleM, show_shapes=True, to_file='ensemble.png')

  # compile & fit
  ensembleM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = ensembleM.fit([x for _ in range(len(ensembleM.input))],y, batch_size=cfg['batch_size'],epochs=EPOCHS, verbose=2, validation_data=validation_data,
                            callbacks=[stopper])



  return history, ensembleM



def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('config',     type=int, help='What kd model config are we using?')
  parser.add_argument('data_dir',   type=str, help='Where is the data located?')
  parser.add_argument('dump_dir',   type=str, help='Where are we dumping the output?')
  parser.add_argument('modl_dir',   type=str, help='Where are the models located?')
  parser.add_argument('model_S',    type=int, help='How many samples per model for training')
  parser.add_argument('model_V',    type=int, help='How many samples per model for testing')
  parser.add_argument('tr_strt',    type=int, help='Where do we start for training index wise?')
  parser.add_argument('ts_strt',    type=int, help='Where do we start for testing index wise?')
  parser.add_argument('seed',       type=int, help='Random seed for run')

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

  # check that our config matches number of models in config
  if len(config['num_filters']) != config['model_N'] or len(config['filter_sizes']) != config['model_N']:
    print('NUMBER OF CONFIG PARAMETERS DOES NOT MATCH NUMBER OF MODELS')
    exit(-1)

  # Step 2: Create training/testing data for models
  xTrain,yTrain,xTest,yTest =  GetData(args.data_dir)

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  # Step 3: Create group ensemble
  models = load_models(config, args.modl_dir)
  # what do the models look like
  for i in range(len(models)):
    print('***********************************************************')
    print(models[i].summary())
  hist,ensemble = CreateEnsemble(models,config,xTrain,yTrain,xTest,yTest)
  print(hist.history)


if __name__ == '__main__':
  main()
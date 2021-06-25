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
      'num_filters': [300,300,300,300],
      'filter_sizes': [3,4,5,6]
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(dir,N):
  # load data
  x = np.load( dir + 'train_X.npy' )
  y = np.load( dir + 'train_Y.npy' )[ :, 0 ]
  xT = np.load( dir + 'test_X.npy' )
  yT = np.load( dir + 'test_Y.npy' )[ :, 0 ]

  # create N copies of the data for each model we are training
  train_x = np.array(x)
  train_y = np.array(y)
  test_x = np.array(xT)
  test_y = np.array(yT)

  for i in range(N-1):
    train_x = np.concatenate((train_x, x))
    train_y = np.concatenate((train_y, y))
    test_x = np.concatenate((test_x, xT))
    test_y = np.concatenate((test_y, yT))

  # find max class number and adjust test/training y
  train_y = to_categorical(train_y)
  test_y = to_categorical(test_y)

  return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

# Create N models and save them
# n: current model being generated
# x,y: training input, output
# xT,yT: testing input, output
# cfg: configuration we are using for this experiment
def BasicModel(n,x,y,xT,yT,cfg,em_max):
  # word vector lengths
  wv_mat = np.random.randn( em_max + 1, cfg['wv_len'] ).astype( 'float32' ) * 0.1
  # validation data
  validation_data = ( { 'Input-' + str(n+1): xT }, {'Dense-' + str(n+1) : yT})
  # stopping criterion
  stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)

  # set input layer, assuming that all input will have same shape as starting case
  input = Input(shape=([x.shape[1]]), name= "Input-" + str(n+1))
  # embedding lookup
  embed = Embedding(len(wv_mat), cfg['wv_len'], input_length=cfg['in_seq_len'], name="embedding-"+ str(n+1),
                      embeddings_regularizer=l2(cfg['emb_l2']))(input)
  # convolutional layer
  conv = Conv1D(filters=cfg['num_filters'][n], kernel_size=cfg['filter_sizes'][n], padding="same",
                  activation="relu", strides=1, name=str(n+1) + "_thfilter")(embed)
  # max pooling layer
  pooling = GlobalMaxPooling1D()(conv)
  #  drop out layer
  concat_drop = Dropout(cfg['dropout'])(pooling)
  # dense (output) layer
  outlayer = Dense(y.shape[1], name= "Dense-"+str(n+1), activation='softmax')( concat_drop )

  # link, compile, and fit model
  model = Model(inputs=input, outputs = outlayer)
  model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

  history = model.fit(x,y, batch_size=cfg['batch_size'],epochs=EPOCHS, verbose=2, validation_data=validation_data, callbacks=[stopper])

  return history, model

# Create N models and save them
# x,y: training input, output
# xT,yT: testing input, output
# N: number of models we are creating
# S: number of samples we take from training per model
# V: number of samples from testing for validation per model
# cfg: configuration we are using for this experiment
def GenerateModels(x,y,xT,yT,N,S,V,cfg,dump,em_max):
  # print the variables (not data) we are working with
  print('# of models:', N)
  print('# of training samples:', S)
  print('# of testing samples:', V, end='\n\n')

  # start and end positions for training/testing ranges
  i,j = 0,S
  u,v = 0,V

  # iterate through the different ranges for training/testing for each model
  for n in range(N):
    print('Working on Model', n + 1)
    print('i,j:', i,j)
    print('u,v:', u,v)

    # send training/testing data for model
    hist, model = BasicModel(n, x[i:j], y[i:j], xT[u:v], yT[u:v], cfg, em_max)

    # create directory to dump all data related to model
    fdir = dump + 'Model-' + str(n+1) + '/'
    os.mkdir(fdir)

    # saving training, testing, softmax values
    fp = open(fdir + 'training_X.pickle', 'wb')
    pk.dump(model.predict(x[i:j]), fp)
    fp.close()

    fp = open(fdir + 'test_X.pickle', 'wb')
    pk.dump(model.predict(xT[u:v]), fp)
    fp.close()

    # save history files
    df = pd.DataFrame({'val_loss': pd.Series(hist.history['val_loss']),'val_acc': pd.Series(hist.history['val_acc']),
                        'loss': pd.Series(hist.history['loss']),'acc': pd.Series(hist.history['acc'])})
    df.to_csv(path_or_buf= fdir + 'history' + '.csv', index=False)

    # save model
    filename = fdir + 'model.h5'
    model.save(filename)

    # save picture of model created
    plot_model(model, fdir + "model.png", show_shapes=True)

    # on to the next range of models
    i += S
    j += S
    u += V
    v += V



def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('config',     type=int, help='What model config are we using?')
  parser.add_argument('data_dir',   type=str, help='Where is the data located?')
  parser.add_argument('dump_dir',   type=str, help='Where are we dumping the output?')
  parser.add_argument('model_N',    type=int, help='How many models are we making?')
  parser.add_argument('model_S',    type=int, help='How many samples per model for training')
  parser.add_argument('model_V',    type=int, help='How many samples per model for testing')
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
  if len(config['num_filters']) != args.model_N or len(config['filter_sizes']) != args.model_N:
    print('NUMBER OF CONFIG PARAMETERS DOES NOT MATCH NUMBER OF MODELS')
    exit(-1)

  # Step 2: Create training/testing data for models
  xTrain,yTrain,xTest,yTest =  GetData(args.data_dir, args.model_N)

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  # Step 3: Generate, create, and store  models
  GenerateModels(xTrain, yTrain, xTest, yTest, args.model_N, args.model_S, args.model_V,
                  config, args.dump_dir, max(np.max(xTrain), np.max(xTest)))


if __name__ == '__main__':
  main()
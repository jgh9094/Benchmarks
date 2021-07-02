'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

Python file will constuct an Multi-Task CNN with any number of tasks.
The number of tasks is calculate during run time from the Y training and testing data.
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
from keras.layers import GlobalMaxPooling1D, Convolution1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers.merge import Concatenate

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
      'num_filters': [3,4,5],
      'filter_sizes': [300,300,300],
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(dir):
  # load data
  rawX = np.load( dir + 'train_X.npy' )
  rawY = np.load( dir + 'train_Y.npy' )
  rawXT = np.load( dir + 'test_X.npy' )
  rawYT = np.load( dir + 'test_Y.npy' )

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
  Y,YT = [],[]
  for t in y:
    Y.append(to_categorical(t))
  for t in yt:
    YT.append(to_categorical(t))

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

  # number of classes per task
  classes = []
  for y in Y:
    classes.append(len(y[0]))

  return np.array(rawX),Y,np.array(rawXT),YT,classes

# will return a mt-cnn with a certain configuration
def CreateMTCnn(num_classes,vocab_size,cfg):
    # define network layers ----------------------------------------------------
    input_shape = tuple([cfg['in_seq_len']])
    model_input = Input(shape=input_shape, name= "Input")
    # embedding lookup
    emb_lookup = Embedding(vocab_size, cfg['wv_len'], input_length=cfg['in_seq_len'],
                           name="embedding", embeddings_regularizer=l2(cfg['emb_l2']))(model_input)

    # convolutional layer and dropout
    conv_blocks = []
    for ith_filter,sz in enumerate(cfg['filter_sizes']):
        conv = Convolution1D(filters=cfg['num_filters'][ ith_filter ], kernel_size=sz, padding="same",
                             activation="relu", strides=1, name=str(ith_filter) + "_thfilter")(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))

    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat_drop = Dropout(cfg['dropout'])(concat)

    # different dense layer per tasks
    FC_models = []
    for i in range(len(num_classes)):
        outlayer = Dense(num_classes[i], name= "Dense"+str(i), activation='softmax')( concat_drop )
        FC_models.append(outlayer)

    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)
    model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

    return model

def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str, help='Where is the data located?')
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
  X,Y,XT,YT,classes =  GetData(args.data_dir)

  # Step 3: Create the mtcnn model
  mtcnn = CreateMTCnn(classes, max(np.max(X),np.max(XT)),config)

  # Step 4: Train mtcnn model

  # create validation data dictionary
  val_dict = {}
  for i in range(len(YT)):
    layer = 'Dense' + str(i)
    val_dict[layer] = YT[i]

  hist = mtcnn.fit(x= X, y= Y, batch_size= config['batch_size'],
          epochs= EPOCHS, verbose= 2, validation_data= ({'Input': XT}, val_dict),
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)]
          )

  # create directory to dump all data related to model
  fdir = args.dump_dir + 'MTModel-' + str(args.config) + '/'
  os.mkdir(fdir)

  pred = mtcnn.predict(X)
  predT = mtcnn.predict(XT)

  print('Saving Training Softmax Output')
  for i in range(len(pred)):
    print('task:',str(i))
    print('--Number of data points: ', len(pred[i]))
    print('--Size of each data point', len(pred[i][0]))

    fname = fdir + 'training-task-' + str(i) + '.npy'
    np.save(fname, pred[i])

  print('Saving Testing Softmax Output')
  for i in range(len(predT)):
    print('task:',str(i))
    print('--Number of data points: ', len(predT[i]))
    print('--Size of each data point', len(predT[i][0]))

    fname = fdir + 'testing-task-' + str(i) + '.npy'
    np.save(fname, predT[i])

  # convert the history.history dict to a pandas DataFrame:
  hist_df = pd.DataFrame(hist.history)
  hist_df.to_csv(path_or_buf= fdir + 'history.csv', index=False)
  print('History Saved!')
  # save model
  mtcnn.save(fdir + 'model.h5')
  print('Model Saved!')
  # save picture of model created
  plot_model(mtcnn, fdir + "model.png", show_shapes=True)
  print('Model Topology Picture Saved!')

if __name__ == '__main__':
  main()
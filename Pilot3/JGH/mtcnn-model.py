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

from loaddata6reg import loadAllTasks
from mpi4py import MPI





# global variables
EPOCHS = 100
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size #Node count. size-1 = max rank.

# return the data for training and testing
# will need to modify if other means of data gathering
# USED FOR LOCAL TESTING PURPOSES: EXPECTING DATA TO BE ON LOCAL MACHINE
'''
def GetData(dir):
  # load data
  rawX = np.load( dir + 'train_X.npy' )
  rawY = np.load( dir + 'train_Y.npy' )
  rawXT = np.load( dir + 'test_X.npy' )
  rawYT = np.load( dir + 'test_Y.npy' )

  # raw data descriptions
  print('RAW DATA DIMS', flush= True)
  print('rawX dim: ', rawX.shape, flush= True)
  print('rawY dim: ', rawY.shape, flush= True)
  print('rawXT dim: ', rawXT.shape, flush= True)
  print('rawYT dim: ', rawYT.shape , end='\n\n', flush= True)

  if rawY.shape[1] != rawYT.shape[1]:
    print('NUMBER OF TASKS NOT THE SAME BETWEEN DATA SETS', flush= True)
    exit(-1)

  # create array for each task output
  y = [[] for i in range(rawY.shape[1])]
  yt = [[] for i in range(rawY.shape[1])]
  # load data
  for t in range(rawY.shape[1]):
    y[t] = rawY[:,t]
    yt[t] = rawYT[:,t]

  # make to catagorical data and pack up
  Y,YT = [],[],[]
  for t in y:
    Y.append(to_categorical(t))
  for t in yt:
    YT.append(to_categorical(t))

  print('Training Output Data', flush= True)
  i = 0
  for y in Y:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  print('Testing Output Data', flush= True)
  i = 0
  for y in YT:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  # number of classes per task
  classes = []
  for y in Y:
    classes.append(len(y[0]))

  return np.array(rawX),Y,np.array(rawXT),YT,classes
'''
# return configuration for the experiment
def GetModelConfig(config):
  # testing configuration
  if config == 0:
    return {
      'learning_rate': 0.01,
      'batch_size': 256,
      'dropout': 0.5,
      'optimizer': 'adam',
      'wv_len': 300,
      'emb_l2': 0.001,
      'in_seq_len': 1500,
      'num_filters': [3,4,5],
      'filter_sizes': [300,300,300],
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST', flush= True)
    exit(-1)

# transform data and return number of classes
def TransformData(rawX, rawXV, rawXT, rawY, rawYV, rawYT):
  # raw data descriptions
  print('RAW DATA DIMS', flush= True)
  print('rawX dim: ', rawX.shape, flush= True)
  print('rawY dim: ', rawY.shape, flush= True)
  print('rawXV dim: ', rawXV.shape, flush= True)
  print('rawYV dim: ', rawYV.shape, flush= True)
  print('rawXT dim: ', rawXT.shape, flush= True)
  print('rawYT dim: ', rawYT.shape , end='\n\n', flush= True)

  # make sure number of tasks between data sets is consistent
  if rawY.shape[1] != rawYT.shape[1] or rawYT.shape[1] != rawYV.shape[1]:
    print('NUMBER OF TASKS NOT THE SAME BETWEEN DATA SETS', flush= True)
    exit(-1)

  # create array for each task output
  y = [[] for i in range(rawY.shape[1])]
  yv = [[] for i in range(rawY.shape[1])]
  yt = [[] for i in range(rawY.shape[1])]

  # load data into appropiate list
  for t in range(rawY.shape[1]):
    y[t] = rawY[:,t]
    yv[t] = rawYV[:,t]
    yt[t] = rawYT[:,t]

  # make to catagorical data and pack up
  Y,YV,YT = [],[],[]
  for t in y:
    Y.append(to_categorical(t))
  for t in yv:
    YV.append(to_categorical(t))
  for t in yt:
    YT.append(to_categorical(t))

  print('Training Output Data', flush= True)
  i = 0
  for y in Y:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  print('Validation Output Data', flush= True)
  i = 0
  for y in YV:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  print('Testing Output Data', flush= True)
  i = 0
  for y in YT:
    print('task', i, flush= True)
    print('--cases:', len(y), flush= True)
    print('--classes:',len(y[0]), flush= True)
    i += 1
  print()

  # number of classes per task
  classes = []
  for y in Y:
    classes.append(len(y[0]))

  return np.array(rawX),np.array(rawXV),np.array(rawXT),Y,YV,YT,classes



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
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  #parser.add_argument('data_dir',     type=str, help='Where is the data located?')
  #parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')
  parser.add_argument('seed',         type=int, help='Random seed for run')

  #It's too complicated to pass the directory as an argument for each model.
  dump_dir = "//gpfs/alpine/med107/proj-shared/kevindeangeli/EnsembleDestilation/joseOutput/" + str(RANK) + "/"
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)


  # Parse all the arguments & set random seed
  args = parser.parse_args()
  print('Seed:', args.seed, end='\n\n', flush= True)
  np.random.seed(args.seed)

  # check that dump directory exists
  # if not os.path.isdir(dump_dir):
  #   print('DUMP DIRECTORY DOES NOT EXIST', flush= True)
  #   exit(-1)

  # Step 1: Get experiment configurations
  config = GetModelConfig(args.config)
  print('run parameters:', config, end='\n\n', flush= True)

  # Step 2: Create training/testing data for models
  # X,Y,XT,YT,classes =  GetData(args.data_dir)
  X, XV, XT, Y, YV, YT= loadAllTasks(print_shapes = False)
  X, XV, XT, Y, YV, YT, classes = TransformData(X, XV, XT, Y, YV, YT)

  # Step 3: Create the mtcnn model
  mtcnn = CreateMTCnn(classes, max(np.max(X),np.max(XT)) + 1,config)

  # Step 4: Train mtcnn model

  # create validation data dictionary
  val_dict = {}
  for i in range(len(YV)):
    layer = 'Dense' + str(i)
    val_dict[layer] = YV[i]

  hist = mtcnn.fit(x= X, y= Y, batch_size= config['batch_size'],
          epochs= EPOCHS, verbose= 1, validation_data= ({'Input': XV}, val_dict),
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)]
          )

  # create directory to dump all data related to model
  fdir = dump_dir + 'MTModel-' + str(args.config) + '-' + args.seed + "_Rank" + str(RANK) +'/'
  os.mkdir(fdir)

  # save predictions from all data inputs

  pred = mtcnn.predict(X)
  predV = mtcnn.predict(XV)
  predT = mtcnn.predict(XT)

  print('Saving Training Softmax Output', flush= True)
  fname = fdir + 'training-task.pickle'
  file = open(fname, 'wb')
  pickle.dump(pred, file)
  file.close()


  # for i in range(len(pred)):
  #   print('task:',str(i))
  #   print('--Number of data points: ', len(pred[i]), flush= True)
  #   print('--Size of each data point', len(pred[i][0]), flush= True)
  #
  #   fname = fdir + 'training-task-' + str(i) + '.npy'
  #   np.save(fname, pred[i])
  # print()

  print('Saving Validation Softmax Output', flush= True)
  fname = fdir + 'validating-task.pickle'
  file = open(fname, 'wb')
  pickle.dump(predV, file)
  file.close()
  # for i in range(len(predV)):
  #   print('task:',str(i), flush= True)
  #   print('--Number of data points: ', len(predV[i]), flush= True)
  #   print('--Size of each data point', len(predV[i][0]), flush= True)
  #
  #   fname = fdir + 'validating-task-' + str(i) + '.npy'
  #   np.save(fname, pred[i])
  # print()

  print('Saving Testing Softmax Output', flush= True)
  fname = fdir + 'testing-task.pickle'
  file = open(fname, 'wb')
  pickle.dump(predT, file)
  file.close()

  # for i in range(len(predT)):
  #   print('task:',str(i))
  #   print('--Number of data points: ', len(predT[i]), flush= True)
  #   print('--Size of each data point', len(predT[i][0]), flush= True)
  #
  #   fname = fdir + 'testing-task-' + str(i) + '.npy'
  #   np.save(fname, predT[i])
  # print()

  # convert the history.history dict to a pandas DataFrame:
  hist_df = pd.DataFrame(hist.history)
  hist_df.to_csv(path_or_buf= fdir + 'history.csv', index=False)
  print('History Saved!', flush= True)

  # save model
  mtcnn.save(fdir + 'model.h5')
  print('Model Saved!', flush= True)

  # save picture of model created
  plot_model(mtcnn, fdir + "model.png", show_shapes=True)
  print('Model Topology Picture Saved!', flush= True)

if __name__ == '__main__':
  main()

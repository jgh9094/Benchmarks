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
from keras.layers.merge import Concatenate

# global variables
EPOCHS = 1
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
      'num_filters': [3,4,5],
      'filter_sizes': [150,150,150],
      'model_N': 4,
      'alpha': 0.5,
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# compute
def knowledge_distillation_loss(y_true, y_pred,alpha,split):
  # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
  y_true, y_true_softs = y_true[: , :split], y_true[: , split:]
  y_pred, y_pred_softs = y_pred[: , :split], y_pred[: , split:]

  diff_alpha = 1 - alpha

  loss = alpha * logloss(y_true,y_pred) +  diff_alpha * logloss(y_true_softs, y_pred_softs)

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
        conv = Conv1D(filters=cfg['num_filters'][ ith_filter ], kernel_size=sz, padding="same",
                             activation="relu", strides=1, name=str(ith_filter) + "_thfilter")(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))

    concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat_drop = Dropout(cfg['dropout'])(concat)

    # different dense layer per tasks
    FC_models = []
    for i in range(len(num_classes)):
        dense = Dense(num_classes[i], name= "Dense"+str(i), )( concat_drop )
        act = Activation('softmax', name= "Active"+str(i))(dense)
        FC_models.append(act)

    # the multitsk model
    model = Model(inputs=model_input, outputs = FC_models)
    model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

    return model

def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str, help='Where is the data located?')
  parser.add_argument('tech_dir',     type=str, help='Where is the teacher data located?')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config are we using?')
  parser.add_argument('seed',         type=int, help='Random seed for run')
  parser.add_argument('temp',         type=float, help='What temperature are we running')

  # Parse all the arguments & set random seed
  args = parser.parse_args()
  print('Seed:', args.seed, end='\n\n')
  np.random.seed(args.seed)

  # check that dump directory exists
  if not os.path.isdir(args.data_dir):
    print('DATA DIRECTORY DOES NOT EXIST')
    exit(-1)
  # check that dump directory exists
  if not os.path.isdir(args.tech_dir):
    print('TEACHER DIRECTORY DOES NOT EXIST')
    exit(-1)
  # check that dump directory exists
  if not os.path.isdir(args.dump_dir):
    print('DUMP DIRECTORY DOES NOT EXIST')
    exit(-1)

  # Step 1: Get experiment configurations
  config = GetModelConfig(args.config)
  print('run parameters:', config, end='\n\n')

  # Step 2: Create training/testing data for models
  X,Y,XT,YT,classes =  GetData(args.data_dir, args.tech_dir)
  global TEMP
  TEMP = args.temp

  # Step 3: Create the studen mtcnn model
  mtcnn = CreateMTCnn(classes, max(np.max(X),np.max(XT)),config)

  #Step 4: Create knowledge distilled student topology

  # remove the last activation layers
  for i in range(4):
    mtcnn.layers.pop()

  # add new distlled layers
  new_out = []
  for i in range(len(classes)):
    logits = mtcnn.get_layer('Dense'+str(i)).output
    probs = Activation('softmax', name='Logits'+str(i))(logits)
    # softed probabilities at raised temperature
    logits_T = Lambda(lambda x: x / TEMP)(logits)
    probs_T = Activation('softmax', name='TLogits'+str(i))(logits_T)
    # output layer
    output = concatenate([probs, probs_T], name="Active"+str(i))
    new_out.append(output)

  # mtcnn distillation model ready to go!
  mtcnn = Model(mtcnn.input, new_out)
  mtcnn.summary()

  # For testing use regular output probabilities - without temperature
  def acc(y_true, y_pred, split):
      y_true = y_true[:, :split]
      y_pred = y_pred[:, :split]
      return categorical_accuracy(y_true, y_pred)

  def categorical_crossentropy(y_true, y_pred, split):
    y_true = y_true[:, :split]
    y_pred = y_pred[:, :split]
    return logloss(y_true, y_pred)

  # logloss with only soft probabilities and targets
  def soft_logloss(y_true, y_pred, split):
    logits = y_true[:, split:]
    y_soft = K.softmax(logits/TEMP)
    y_pred_soft = y_pred[:, split:]
    return logloss(y_soft, y_pred_soft)

  # create loss dictionary for each task
  losses = {}
  for i in range(len(classes)):
    l = lambda y_true, y_pred: knowledge_distillation_loss(y_true,y_pred,config['alpha'],classes[i])
    l.__name__ = 'kdl'
    losses['Active'+str(i)] = l
  # create metric dictionary per task
  metrics = {}
  for i in range(len(classes)):
    metrics['Active'+str(i)] = []
    l1 = lambda y_true, y_pred: acc(y_true,y_pred,classes[i])
    l1.__name__ = 'acc'
    metrics['Active'+str(i)].append(l1)
    l2 = lambda y_true, y_pred: categorical_crossentropy(y_true,y_pred,classes[i])
    l2.__name__ = 'cc'
    metrics['Active'+str(i)].append(l2)
    l3 = lambda y_true, y_pred: soft_logloss(y_true,y_pred,classes[i])
    l3.__name__ = 'sl'
    metrics['Active'+str(i)].append(l3)


  mtcnn.compile(optimizer='adam', loss=losses, metrics=metrics)

  hist = mtcnn.fit(X, Y,
            batch_size=256,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(XT, YT),
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)])


  # begin saving everything
  # fdir = args.dump_dir + 'MTEnsemble-' + str(args.config) + '/'
  # os.mkdir(fdir)

  # save picture of model created
  # plot_model(mtcnn, fdir + "model.png", show_shapes=True)
  # plot_model(mtcnn, "after.png", show_shapes=True)
  print('Model Topology Picture Saved!')

if __name__ == '__main__':
  main()
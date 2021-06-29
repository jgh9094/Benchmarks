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
ALPHA = 0.0

def accuracy(y_true, y_pred):
    y_true = y_true[:, :15]
    y_pred = y_pred[:, :15]
    return categorical_accuracy(y_true, y_pred)

def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :15]
    y_pred = y_pred[:, :15]
    return K.sum(top_k_categorical_accuracy(y_true, y_pred))

def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :15]
    y_pred = y_pred[:, :15]
    return logloss(y_true, y_pred)

# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):
  logits = y_true[:, 15:]
  y_soft = K.softmax(logits/TEMP)
  y_pred_soft = y_pred[:, 15:]
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
      'alpha': 0.7,
      'temp': 1.0
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# compute
def knowledge_distillation_loss(y_true, y_pred, alpha):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :SPLIT], y_true[: , SPLIT:]

    y_pred, y_pred_softs = y_pred[: , :SPLIT], y_pred[: , SPLIT:]

    loss = logloss(y_true,y_pred) + logloss(y_true_softs, y_pred_softs)

    return loss

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
    Y.append(np.concatenate((y[i],ty[i%5])))

  YT = []
  for i in range(len(yt)):
    YT.append(np.concatenate((yt[i],tyt[i%5])))

  return np.array(Y),np.array(YT)

# create student model
def CreateStudent(x,y,cfg,em_max):
  # word vector lengths
  wv_mat = np.random.randn( em_max + 1, cfg['wv_len'] ).astype( 'float32' ) * 0.1

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
  dense = Dense(int(int(y.shape[1])/2), name= "Dense1")( concat_drop )

  act = Activation('softmax')(dense)

  # link, compile, and fit model
  model = Model(inputs=input, outputs = act)

  #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  model.summary()

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
  # global SPLIT, ALPHA
  SPLIT = len(yTrain[0])
  ALPHA = config['alpha']
  TEMP = config['temp']


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
  print(yTrain[0])

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  # Step 3: Create, compile, train student model
  student = CreateStudent(xTrain,yTrain,config, max(np.max(xTrain), np.max(xTest)))
  plot_model(student,to_file= args.dump_dir + 'student-0.png',show_shapes=True, show_layer_names=True)

  # Remove the softmax layer from the student network
  student.layers.pop()

  # Now collect the logits from the last layer
  logits = student.layers[-1].output # This is going to be a tensor. And hence it needs to pass through a Activation layer
  probs = Activation('softmax')(logits)

  # softed probabilities at raised temperature
  logits_T = Lambda(lambda x: x / TEMP)(logits)
  probs_T = Activation('softmax')(logits_T)

  output = concatenate([probs, probs_T])

  # This is our new student model
  student = Model(student.input, output)

  student.summary()
  plot_model(student,to_file= args.dump_dir + 'student-1.png',show_shapes=True, show_layer_names=True)

  print(xTrain[0])
  print(xTrain[0].shape)
  print(xTrain[0].reshape(1500,1))
  print(xTrain[0].reshape(1500,1).shape)


  # For testing use regular output probabilities - without temperature
  def acc(y_true, y_pred):
      y_true = y_true[:, :SPLIT]
      y_pred = y_pred[:, :SPLIT]
      return categorical_accuracy(y_true, y_pred)

  student.compile(
      #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
      optimizer='adam',
      loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 0.1),
      #loss='categorical_crossentropy',
      metrics=[acc,categorical_crossentropy,soft_logloss] )

  student.fit(xTrain, yTrain,
            batch_size=256,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(xTest, yTest),
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)])

  print(student.predict(np.array([np.array(xTrain[0])])))

if __name__ == '__main__':
  main()
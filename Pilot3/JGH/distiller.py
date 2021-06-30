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
      'temp': 7.0
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# compute
def knowledge_distillation_loss(y_true, y_pred, alpha):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :SPLIT], y_true[: , SPLIT:]
    y_pred, y_pred_softs = y_pred[: , :SPLIT], y_pred[: , SPLIT:]

    diff_alpha = 1 - alpha

    loss = alpha * logloss(y_true,y_pred) +  diff_alpha * logloss(y_true_softs, y_pred_softs)

    return loss

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(data_d,tech_d,config):
  # load data
  X = np.load( data_d + 'train_X.npy' )
  Y = to_categorical(np.load( data_d + 'train_Y.npy' )[ :, 0 ])
  XT = np.load( data_d + 'test_X.npy' )
  YT = to_categorical(np.load( data_d + 'test_Y.npy' )[ :, 0 ])

  # get teacher logit outputs
  file = open(tech_d + 'Ensemble-' + str(config) + '/' + 'training_X.pickle', 'rb')
  teach_x = pk.load(file)
  file.close
  file = open(tech_d + 'Ensemble-' + str(config) + '/' + 'test_X.pickle', 'rb')
  teach_xt = pk.load(file)
  file.close

  # combine hard labels with teacher logits
  Y,YT = CombineData(Y,YT,teach_x,teach_xt)

  return np.array(X), np.array(Y), np.array(XT), np.array(YT)

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
  parser.add_argument('data_dir',      type=str, help='Where is the teacher data located?')
  parser.add_argument('tech_dir',      type=str, help='Where is the student data located?')
  parser.add_argument('dump_dir',      type=str, help='Where are we dumping the output?')
  parser.add_argument('config',        type=int, help='Configuration used in single model?')
  parser.add_argument('seed',          type=int, help='Random seed for run')

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

  # Step 2: Create training/testing data for ensemble model
  xTrain,yTrain,xTest,yTest =  GetData(args.data_dir, args.tech_dir, args.config)
  # global SPLIT, ALPHA
  SPLIT = len(yTrain[0])
  ALPHA = config['alpha']
  TEMP = config['temp']


  # # get the teacher training/testing outputs
  # file = open('./Model-1/training_X.pickle', 'rb')
  # ttrain_X = pk.load(file)
  # file.close
  # file = open('./Model-1/test_X.pickle', 'rb')
  # ttest_X = pk.load(file)
  # file.close

  # print(ttrain_X.shape)
  # print(ttest_X.shape)
  # print(ttrain_X[0])

  # yTrain,yTest = CombineData(yTrain,yTest,ttrain_X,ttest_X)
  # print(yTrain[0])

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  exit(-1)

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

  student.compile(
      #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
      optimizer='adam',
      loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 0.5),
      #loss='categorical_crossentropy',
      metrics=[acc,categorical_crossentropy,soft_logloss] )

  student.fit(xTrain, yTrain,
            batch_size=256,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(xTest, yTest),
            callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)])

  print('predict:', student.predict(np.array([np.array(xTrain[0])])))
  print('label:', yTrain[0])

if __name__ == '__main__':
  main()
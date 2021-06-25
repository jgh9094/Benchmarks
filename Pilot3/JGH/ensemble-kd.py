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
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy,KLDivergence

# knowledge distillation files
from distiller import Distiller

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
  # Concatenate merge output from each model
  ensemble_outputs = [model.output for model in models]
  # model layers
  merge = Concatenate(axis=1)(ensemble_outputs)
  hidden = Dense(y.shape[1] * cfg['model_N'], activation='relu')(merge)
  output = Dense(y.shape[1], activation='softmax')(hidden)
  # ensemble model
  ensembleM = Model(inputs=ensemble_visible, outputs=output)

  # validation data
  validation_data = [[xT for _ in range(len(ensembleM.input))], yT]

  # compile & fit
  ensembleM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = ensembleM.fit([x for _ in range(len(ensembleM.input))],y, batch_size=cfg['batch_size'],epochs=EPOCHS, verbose=2, validation_data=validation_data,
                            callbacks=[stopper])

  return history, ensembleM

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
  outlayer = Dense(y.shape[1], name= "Dense", activation='softmax')( concat_drop )

  # link, compile, and fit model
  model = Model(inputs=input, outputs = outlayer)
  model.compile( loss= "categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

  # history = model.fit(x,y, batch_size=cfg['batch_size'],epochs=EPOCHS, verbose=2, validation_data=validation_data, callbacks=[stopper])

  return model



def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('config',      type=int, help='What kd model config are we using?')
  parser.add_argument('teach_dir',   type=str, help='Where is the teacher data located?')
  parser.add_argument('studt_dir',   type=str, help='Where is the student data located?')
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
  xTrain,yTrain,xTest,yTest =  GetData(args.teach_dir, config['model_N'])

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  # Step 3: Create group ensemble
  models = load_models(config, args.modl_dir)
  hist,teacher = CreateEnsemble(models,config,xTrain,yTrain,xTest,yTest)
  # plot graph of teacher
  plot_model(teacher, show_shapes=True, to_file= args.dump_dir + 'teacher.png')
  print(hist.history)

  print('STUDENT STATS')

  # Step 5: Create training/testing data for student model
  xTrain,yTrain,xTest,yTest =  GetData(args.studt_dir, config['model_N'])

  # Step 6: Create student model
  student = CreateStudent(xTrain,yTrain,xTest,yTest,config,max(np.max(xTrain), np.max(xTest)))

  # Step7: Begin distilation
  distiller = Distiller(student=student, teacher=teacher)
  distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
  )

  # Distill teacher to student
  distiller.fit(xTrain, yTrain, epochs=EPOCHS)

  # Evaluate student on test dataset
  distiller.evaluate(xTest, yTest)


if __name__ == '__main__':
  main()
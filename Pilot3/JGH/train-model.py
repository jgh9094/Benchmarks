'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com
Date: 2/11/21

export PATH=$HOME/anaconda3/bin:$PATH
'''

# general python imports
import numpy as np
from matplotlib import pyplot as plt
import argparse

# keras python inputs
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.regularizers import l2
from keras.layers import GlobalMaxPooling1D, Conv1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping




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
      'filter_sizes': 3,
      'filter_sets': 3,
      'num_filters': 100,
      'emb_l2': 0.001,
      'w_l2': 0.01,
      'in_seq_len': 1500,
      'num_filters': [300,300,300],
      'filter_sizes': [3,4,5]
    }

  else:
    print('MODEL CONFIGURATION DOES NOT EXIST')
    exit(-1)

# return the data for training and testing
# will need to modify if other means of data gathering
def GetData(dir):
  train_x = np.load( dir + '/train_X.npy' )
  train_y = np.load( dir + '/train_Y.npy' )
  test_x = np.load( dir + '/test_X.npy' )
  test_y = np.load( dir + '/test_Y.npy' )

  return train_x, train_y[ :, 0 ], test_x, test_y[ :, 0 ]

# Create N models and save them
# x,y: training input, output
# xT,yT: testing input, output
# N: number of models we are creating
# S: number of samples we take from training per model
# V: number of samples from testing for validation per model
# cfg: configuration we are using for this experiment
def GenerateModels(x, y, xT, yT, N, S, V, cfg):
  # print the variables (not data) we are working with
  print('# of models:', N)
  print('# of training samples:', S)
  print('# of testing samples:', V, end='\n\n')

  # word vector lengths
  wv_mat = np.random.randn( max(np.max(x), np.max(xT)) + 1, cfg['wv_len'] ).astype( 'float32' ) * 0.1
  # validation data
  validation_data = ( { 'Input': xT }, {'Dense0': yT})

  X = [i for i in range(5000)]
  Y = [i for i in range(5000)]


  # start and end positions for training/testing ranges
  i,j = 0,S
  u,v = 0,V

  # set input layer, assuming that all input will have same shape as starting case
  input = Input(shape=x[0].shape, name= "Input")

  # embedding lookup
  embed = Embedding(len(wv_mat), cfg['wv_len'], input_length=cfg['in_seq_len'],
                          name="embedding", embeddings_regularizer=l2(cfg['emb_l2']))(input)

  # convolutional layer and dropout
  conv = Conv1D(filters=cfg['num_filters'][0],
                             kernel_size=cfg['filter_sizes'][0],
                             padding="same",
                             activation="relu",
                             strides=1,
                             name=str(0) + "_thfilter")(embed)
  # max pooling layer
  pooling = GlobalMaxPooling1D()(conv)
  #  drop out layer
  concat_drop = Dropout(cfg['dropout'])(pooling)
  # dense (output) layer
  outlayer = Dense(np.max(y) + 1, name= "Dense"+str(0), activation='softmax')( concat_drop )

  # the multitsk model
  model = Model(inputs=input, outputs = outlayer)
  model.compile( loss= "sparse_categorical_crossentropy", optimizer= cfg['optimizer'], metrics=[ "acc" ] )

  plot_model(model, "my_first_model.png", show_shapes=True)

  # stopping criterion
  stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', restore_best_weights=True)

  hist = model.fit(x=np.array(x), y=[np.array(y)], batch_size=cfg['batch_size'],
                    epochs=EPOCHS, verbose=2, validation_data=validation_data, callbacks=[stopper]
                  )


  # iterate through the different ranges for training/testing for each model
  for it in range(N):
    print('i,j:', i,j)
    print('X: ', X[i:j])
    i += S
    j += S

    print('u,v',u,v)
    print('Y:', Y[u:v])
    u += V
    v += V

    print()



  return 0



def main():
  print('\n************************************************************************************', end='\n\n')
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('model_txt',  type=int, help='What model text are we using?')
  parser.add_argument('data_dir',   type=str, help='Where is the data located?')
  parser.add_argument('model_N',    type=int, help='How many models are we making?')
  parser.add_argument('model_S',    type=int, help='How many samples per model for training')
  parser.add_argument('model_V',    type=int, help='How many samples per model for testing')
  parser.add_argument('seed',       type=int, help='Random seed for run')

  # Parse all the arguments & set random seed
  args = parser.parse_args()
  print('Seed:', args.seed, end='\n\n')
  np.random.seed(args.seed)

  # Step 1: Get experiment configurations
  config = GetModelConfig(args.model_txt)
  print('run parameters:', config, end='\n\n')

  # Step 2: Create training/testing data for models
  xTrain,yTrain,xTest,yTest =  GetData(args.data_dir)

  # quick descriptors of the data
  # could also do some fancy tricks to data before we send off to cnn
  print('xTrain dim: ', xTrain.shape)
  print('yTrain dim: ', yTrain.shape)
  print('xTest dim: ', xTest.shape)
  print('yTest dim: ', yTest.shape , end='\n\n')

  for i,j in enumerate([3,5,6]):
    print(i,j)

  # Step 3: Train the cnn models
  GenerateModels(xTrain, yTrain, xTest, yTest, args.model_N, args.model_S, args.model_V, config)



if __name__ == '__main__':
  main()
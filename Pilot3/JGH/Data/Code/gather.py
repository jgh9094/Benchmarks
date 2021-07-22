'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

Python file will aggregate
'''

# general python imports
import numpy as np
import argparse
import pickle as pk
import psutil
import os
import pandas as pd

from sklearn.metrics import f1_score

# global variables for data storing
data = {'seed': [], 'Beh_Mic': [], 'Beh_Mac': [], 'His_Mic': [], 'His_Mac': [], 'Lat_Mic': [],
          'Lat_Mac': [], 'Site_Mic': [], 'Site_Mac': [], 'Subs_Mic': [], 'Subs_Mac': []}
header = ['seed', 'Beh_Mic', 'Beh_Mac', 'His_Mic', 'His_Mac', 'Lat_Mic', 'Lat_Mac', 'Site_Mic',
                              'Site_Mac', 'Subs_Mic', 'Subs_Mac']

def GetModelType(c):
  if c == 0:
    return 'MTModel-0_Rank-'

  else:
    print('UNKNOWN MODEL TYPE')

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where is the data?')
  parser.add_argument('dump_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('model',        type=int,      help='What type of models are we getting')
  parser.add_argument('models',       type=int,      help='How many models')
  parser.add_argument('name',         type=str,      help='Name of file to output')

  # parse all the argument
  args = parser.parse_args()
  print(args)

  # what are the inputs
  print('data_dir:', args.data_dir, flush= True)

  # iterate through all the models and gather the data
  for r in range(args.models):
    # load data
    print(args.data_dir + GetModelType(args.model) + str(r) + '/MicMacTest_R' + str(r) + '.csv')
    file = args.data_dir + GetModelType(args.model) + str(r) + '/MicMacTest_R' + str(r) + '.csv'
    df = pd.read_csv(file, index_col=False)
    # store and update data
    x = df.iloc[1].to_list()
    x[0] = r
    # store data
    for i in range(len(header)):
      data[header[i]].append(x[i])

  print(data)
  pd.DataFrame(data).to_csv(args.dump_dir + args.name + '.csv', index = False)



if __name__ == '__main__':
  main()

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

import os.path
from os import path

import pandas as pd

from sklearn.metrics import f1_score

# global variables for data storing
data = {'seed': [], 'Beh_Mic': [], 'Beh_Mac': [], 'His_Mic': [], 'His_Mac': [], 'Lat_Mic': [],
          'Lat_Mac': [], 'Site_Mic': [], 'Site_Mac': [], 'Subs_Mic': [], 'Subs_Mac': [], 'model': []}
header = ['seed', 'Beh_Mic', 'Beh_Mac', 'His_Mic', 'His_Mac', 'Lat_Mic', 'Lat_Mac', 'Site_Mic',
                              'Site_Mac', 'Subs_Mic', 'Subs_Mac', 'model']

Pvals = ['P-1', 'P-2', 'P-5']

temp = [1,2,5,7,10,13,15,17,20,22,25,30]

def GetModelType(c, n):
  if c == 0:
    return 'MTModel-'+ str(n) +'_Rank-'
  elif c == 1 or c == 2:
    return 'MicMacTest_R.csv'
  elif c == 3:
    return 'MTDistilled-'+ str(n) +'-'
  else:
    print('UNKNOWN MODEL TYPE')

def Get276(args):
  # iterate through all the models and gather the data
  for r in range(args.models):
    # load data
    print(args.data_dir + GetModelType(args.model, args.cfg) + str(r) + '/MicMacTest_R' + str(r) + '.csv')
    file = args.data_dir + GetModelType(args.model, args.cfg) + str(r) + '/MicMacTest_R' + str(r) + '.csv'
    df = pd.read_csv(file, index_col=False)
    # store and update data
    x = df.iloc[1].to_list()
    x[0] = r
    x.append(args.name)
    # store data
    for i in range(len(header)):
      data[header[i]].append(x[i])

  print(data)
  pd.DataFrame(data).to_csv(args.dump_dir + args.name + '.csv', index = False)

def GetP(args):
  # iterate through all the models and gather the data
  for r in range(len(Pvals)):
    # load data
    print(args.data_dir + Pvals[r] + '/' + GetModelType(args.model, args.cfg))
    file = args.data_dir + Pvals[r] + '/' + GetModelType(args.model, args.cfg)
    df = pd.read_csv(file, index_col=False)
    # store and update data
    x = df.iloc[1].to_list()
    x[0] = r
    x.append(Pvals[r])
    # store data
    for i in range(len(header)):
      data[header[i]].append(x[i])

  print(data)
  pd.DataFrame(data).to_csv(args.dump_dir + args.name + '.csv', index = False)

def GetA(args):
  # load data
  print(args.data_dir + GetModelType(args.model, args.cfg))
  file = args.data_dir + GetModelType(args.model, args.cfg)
  df = pd.read_csv(file, index_col=False)
  # store and update data
  x = df.iloc[1].to_list()
  x[0] = 0
  x.append(args.name)
  # store data
  for i in range(len(header)):
    data[header[i]].append(x[i])

  print(data)
  pd.DataFrame(data).to_csv(args.dump_dir + args.name + '.csv', index = False)

def GetDisAgg(args):
  for i in range(args.models):
    file = args.data_dir + GetModelType(args.model, args.cfg) + str(i) + '/' + 'MicMacTest_R'+ str(i) +'.csv'
    print (file +"exists:"+str(path.exists(file)))

    if not path.exists(file):
      continue

    df = pd.read_csv(file, index_col=False)
    # store and update data
    x = df.iloc[1].to_list()
    x[0] = i
    x.append('t-'+str(temp[i]))
    # store data
    for i in range(len(header)):
      data[header[i]].append(x[i])

  print(data)
  pd.DataFrame(data).to_csv(args.dump_dir + args.name + '.csv', index = False)

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where is the data?')
  parser.add_argument('dump_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('model',        type=int,      help='0: 276 models, 1: Partial % models, 2: Distilled ')
  parser.add_argument('models',       type=int,      help='How many models')
  parser.add_argument('name',         type=str,      help='Name of file to output')
  parser.add_argument('cfg',          type=int,      help='Configuration we used')

  # parse all the argument
  args = parser.parse_args()
  print(args)

  # what are the inputs
  print('data_dir:', args.data_dir, flush= True)

  if args.model == 0:
    Get276(args)
  elif args.model == 1:
    GetP(args)
  elif args.model == 2:
    GetA(args)
  elif args.model == 3:
    GetDisAgg(args)
  else:
    print('UNKNOWN')
    exit(-1)



if __name__ == '__main__':
  main()

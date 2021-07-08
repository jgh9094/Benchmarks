'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

Python file will aggregate
'''

# general python imports
import numpy as np
import argparse
import os
import pandas as pd
import pickle as pk

# OLCF imports
# from mpi4py import MPI

# global variables
# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()
# SIZE = COMM.size #Node count. size-1 = max rank.

def GetFolderName(c):
  if c == 0:
    return 'Model-'
  elif c == 1:
    return 'MTModel-'

def GetData(dir,task,mods):
  # store the directories we are lookin in and dimensions of softmax
  dirs = []
  for i in range(mods):
    fdir = dir + '_Rank-' + str(i) + '/'
    dirs.append(fdir)

  print('DIRS EXPLORING:')
  for d in dirs:
    print(d)


  # get training data
  print('Aggregating training data...')
  train = []
  # check that dimenstions are the same
  x,y = [],[]
  for dir in dirs:
    print(dir + '/training-task-' + str(task) + '.npy')
    X = np.load(dir + '/training-task-' + str(task) + '.npy')
    print(X.shape)
    train.append(X)

    # store dimensions
    x.append(X.shape[0])
    y.append(X.shape[1])

  # make sure that dimensions match for all data
  if 1 < len(set(x)) or 1 < len(set(y)):
    print('TRAINING DATA DIMS NOT EQUAL')
    exit(-1)

  for x in train:
    print(x)
    print()






  return 0


def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config was used')
  parser.add_argument('models',       type=int, help='Number of models used')
  parser.add_argument('cnn',          type=int, help='0: Single, 1: MT model')

  # parse all the argument
  args = parser.parse_args()

  # RANK is synonomous with the task task being evaluated
  RANK = 0 # used for example right now
  seed = int(RANK)
  print('RANK:', seed)

  dir = args.data_dir + GetFolderName(args.cnn) + str(args.config)

  print('dir:', dir)
  print()

  GetData(dir, seed, args.models)





if __name__ == '__main__':
  main()

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

def GetData():


  return 0


def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config was used')
  parser.add_argument('models',       type=float, help='Number of models used')
  parser.add_argument('cnn',          type=float, help='0: Single, 1: MT model')

  # parse all the argument
  args = parser.parse_args()

  # RANK is synonomous with the task task being evaluated
  RANK = 0 # used for example right now
  seed = int(RANK)
  print('RANK:', seed)

  fdir = args.data_dir + GetFolderName(args.cnn) + '_Rank-' + str(RANK)

  print(fdir)





if __name__ == '__main__':
  main()

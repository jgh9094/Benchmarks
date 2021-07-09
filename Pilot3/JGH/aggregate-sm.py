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

def GetDataDirs(dir,mods):
  # store the directories we are lookin in and dimensions of softmax
  dirs = [f for f in os.listdir(dir) if f.isdir()]
  # for i in range(mods):
  #   fdir = dir + '_Rank-' + str(i) + '/'
  #   dirs.append(fdir)

  print('DIRS EXPLORING:')
  for d in dirs:
    print(d)
  print()

  return dirs

def AverageData(dirs,task,mods,dump,data):
  # get training data
  print('AVERAGING',data.upper(),'DATA...')

  # check that dimenstions are the same
  x,y = [],[]
  # go through all files and check the dimensions
  print('CHECKING DATA DIMENSIONS...')
  for dir in dirs:
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy', mmap_mode='r')
    # store dimensions
    x.append(X.shape[0])
    y.append(X.shape[1])
    del X

  # make sure that dimensions match for all data
  if 1 < len(set(x)) or 1 < len(set(y)):
    print('TRAINING DATA DIMS NOT EQUAL')
    exit(-1)
  else:
    print('DATA DIMENSIONS MATCH!')

  # matrix that will
  mat = np.zeros(shape=(x[0],y[0]))
  del x,y

  # memory checks
  print('mem1',psutil.virtual_memory())

  for dir in dirs:
    print('processing:', dir + data +'-task-' + str(task) + '.npy')
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy', mmap_mode='r')

    # iteratate through each file and update the matrix
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        mat[i][j] += X[i][j]

    del X

  # divide all elements in matrix by number of models
  mat = np.array([m / float(mods) for m in mat])

  print('mem2',psutil.virtual_memory())

  np.save(dump + data + '-task-' + str(task) +'.npy', mat)
  print('finished saving:', dump + data + '-task-' + str(task) +'.npy')
  print()

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('dump_dir',     type=str, help='Where are we dumping the output?')
  parser.add_argument('config',       type=int, help='What model config was used')
  parser.add_argument('models',       type=int, help='Number of models used')
  parser.add_argument('cnn',          type=int, help='0: Single, 1: MT model')

  # parse all the argument
  args = parser.parse_args()

  # RANK is synonomous with the task task being evaluated
  RANK = 0 # used for example right now
  task = int(RANK)
  print('task:', task)

  dir = args.data_dir + GetFolderName(args.cnn) + str(args.config)

  # Step 1: Get data directories we are exploring
  dirs = GetDataDirs(dir,args.models)

  # Step 2: Average training data
  # AverageData(dirs,task,args.models,args.dump_dir, 'training')

  # Step 3: Average testing data
  # AverageData(dirs,task,args.models,args.dump_dir, 'testing')

  # Step 3: Average testing data
  # AverageData(dirs,task,args.models,args.dump_dir, 'validating')


if __name__ == '__main__':
  main()

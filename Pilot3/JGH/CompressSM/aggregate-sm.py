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
from mpi4py import MPI

# global variables
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size #Node count. size-1 = max rank.

# will look at all directories in data dir and sample a set of them
def GetDataDirs(dir,p):
  # store the directories we are lookin in and dimensions of softmax
  dirs = filter(os.path.isdir, [os.path.join(dir, o) for o in os.listdir(dir)])
  dirs = [dir + '/' for dir in dirs]

  sub = int(p * len(dirs))
  dirs = np.sort(np.random.choice(dirs, sub, replace=False))

  print('DIRS EXPLORING:', flush= True)
  for d in dirs:
    print(d, flush= True)
  print(flush= True)

  return dirs

# will look through all dirs and average out their data (testing, training, validate)
def AverageData(dirs,task,dump,data):
  # get training data
  print('AVERAGING',data.upper(),'DATA...', flush= True)

  # check that dimenstions are the same
  x,y = [],[]
  # go through all files and check the dimensions
  print('CHECKING DATA DIMENSIONS...', flush= True)
  for dir in dirs:
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy', mmap_mode='r')
    # store dimensions
    x.append(X.shape[0])
    y.append(X.shape[1])
    del X

  # make sure that dimensions match for all data
  if 1 < len(set(x)) or 1 < len(set(y)):
    print('TRAINING DATA DIMS NOT EQUAL', flush= True)
    exit(-1)
  else:
    print('DATA DIMENSIONS MATCH!', flush= True)

  # matrix that will
  mat = np.zeros(shape=(x[0],y[0]))
  del x,y

  for dir in dirs:
    print('processing:', dir + data +'-task-' + str(task) + '.npy', flush= True)
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy', mmap_mode='r')

    # iteratate through each file and update the matrix
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        mat[i][j] += X[i][j]

    # del X

  # divide all elements in matrix by number of models
  mat = np.array([m / float(len(dirs)) for m in mat])

  # memory checks
  print('memory:',psutil.virtual_memory(), flush= True)

  np.save(dump + data + '-task-' + str(task) +'.npy', mat)
  print('finished saving:', dump + data + '-task-' + str(task) +'.npy', flush= True)
  print(flush= True)

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('dump_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('proportion',   type=float,    help='What model config was used')
  parser.add_argument('offset',       type=int,      help='Seed offset for rng')

  # RANK is synonomous with the task task being evaluated
  # RANK = 0 # used for example right now
  task = int(RANK)
  print('task:', task, flush= True)

  # parse all the argument
  args = parser.parse_args()

  # set seed for rng
  seed = int(task+args.offset)
  print('RANDOM SEED:', seed, flush= True)
  np.random.seed(seed)

  # Step 1: Get data directories we are exploring
  dirs = GetDataDirs(args.data_dir.strip(),args.proportion)

  # Step 2: Average training data
  AverageData(dirs,task,args.dump_dir, 'training')

  # Step 3: Average testing data
  AverageData(dirs,task,args.dump_dir, 'testing')

  # Step 3: Average testing data
  AverageData(dirs,task,args.dump_dir, 'validating')

if __name__ == '__main__':
  main()

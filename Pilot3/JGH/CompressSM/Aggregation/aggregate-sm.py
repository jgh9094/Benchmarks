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

  X = np.load(file=dirs[0] + data +'-task-' + str(task) + '.npy')

  # matrix that will
  mat = np.zeros(shape=(X.shape[0],X.shape[1]))

  del X

  print('PROCESSING FILE', flush=True)
  for dir in dirs:
    print('processing:', dir + data +'-task-' + str(task) + '.npy', flush= True)
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy')

    # iteratate through each file and update the matrix
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        mat[i][j] += X[i][j]

    del X

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
  parser.add_argument('data_type',    type=int,      help='0: training, 1: testing, 2: validating')

  # RANK is synonomous with the task task being evaluated
  # RANK = 3 # used for example right now
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
  if args.data_type == 0:
    if task == 0 or task == 2:
      print('ALREADY DID TASK', task)
      return 0
    AverageData(dirs,task,args.dump_dir, 'training')

  elif args.data_type == 1:
    if task == 0 or task == 2 or task == 3:
      print('ALREADY DID TASK', task)
      return 0
    AverageData(dirs,task,args.dump_dir, 'testing')

  elif args.data_type == 2:
    if task == 0 or task == 2:
      print('ALREADY DID TASK', task)
      return 0
    AverageData(dirs,task,args.dump_dir, 'validating')

  else:
    print('UNKNOWN DATA TYPE')
    exit(-1)

if __name__ == '__main__':
  main()

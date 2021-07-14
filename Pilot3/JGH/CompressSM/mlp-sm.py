'''
Created by: Jose Guadalupe Hernandez
Email: jgh9094@gmail.com

export PATH=$HOME/anaconda3/bin:$PATH

Python file will pass multiple inputs into a multi-layered perceptron to
calculate the correct weights of all other inputs for output.
'''

# general python imports
import numpy as np
import argparse
import pickle as pk
import psutil
import os

# # OLCF imports
# from mpi4py import MPI

# # global variables
# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()
# SIZE = COMM.size #Node count. size-1 = max rank.
YLAB = '/gpfs/alpine/world-shared/med106/yoonh/storageFolder/HardLabels/'

# will look at all directories in data dir and sample a set of them
def GetDataDirs(dir,p):
  # store the directories we are lookin in and dimensions of softmax
  dirs = filter(os.path.isdir, [os.path.join(dir, o) for o in os.listdir(dir)])
  dirs = [dir + '/' for dir in dirs]

  sub = int(p * len(dirs))
  dirs = np.sort(np.random.choice(dirs, sub, replace=False))

  print('DIRS EXPLORING:')
  for d in dirs:
    print(d)
  print()

  return dirs

# concatenate all data into on sinlge matrix
def AggregateData(dirs,task,data):
  # get training data
  print('COLLECTING',data.upper(),'DATA...', flush= True)

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
  mat = [ np.array([]) for i in range(x[0])]
  print('len(mat)', len(mat))
  print('x', x[0])
  print('y', y[0])

  del x,y

  for dir in dirs:
    # go through all the dirs
    print('processing:', dir + data +'-task-' + str(task) + '.npy', flush= True)
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy')

    # go through all the data points and create a new data matrix
    for i in range(len(X)):
      mat[i] = np.concatenate((mat[i], X[i]), axis=None)

    del X

  print('FINISHED GOING THROUGH ALL DIRS')
  mat = np.array(mat)

  print('mat.shape:',mat.shape)

  # memory checks
  print('memory:',psutil.virtual_memory(), flush= True)
  print()

  return mat

# get a specific row of y labels
def GetYLabs(dir,task,name):
  print('GETTING Y LABELS FOR', str(task).upper())

  file = open(dir + name, 'rb')
  ylab = pk.load(file)
  file.close

  print('type(ylab):',type(ylab))
  print('ylab.shape', ylab.shape)
  print('ylab:', ylab)
  print()

  # for testing purposes [0:20000]
  ylab = ylab[0:20000,task]
  print('ylab.shape', ylab.shape)
  print('ylab:', ylab)

  print()
  return 0

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('dump_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('proportion',   type=float,    help='What model config was used')
  parser.add_argument('offset',       type=int,      help='Seed offset for rng')

  # RANK is synonomous with the task task being evaluated
  RANK = 0 # used for example right now
  task = int(RANK)
  print('task:', task)

  # parse all the argument
  args = parser.parse_args()

  # set seed for rng
  seed = int(task+args.offset)
  print('RANDOM SEED:', seed)
  np.random.seed(seed)

  # Step 1: Get data directories we are exploring
  dirs = GetDataDirs(args.data_dir.strip(),args.proportion)

  # Step 2:  Get all data and transform it into one matrix

  X = AggregateData(dirs,RANK,'training')
  XV = AggregateData(dirs,RANK,'validating')
  Y = GetYLabs(YLAB, RANK, 'train_y.pickle')
  YV = GetYLabs(YLAB, RANK, 'val_y.pickle')





if __name__ == '__main__':
  main()
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

# softmax output transformer
def softmax(x):
    ex = np.exp(x)
    tot = np.sum(ex)
    return ex / tot


def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where are we dumping the output?')

  # parse all the argument
  args = parser.parse_args()

  # what are the inputs
  print('data_dir:', args.data_dir, flush= True)

  # get the softmax values of the our predictions from raw logits
  predT = np.load(args.data_dir + 'testing-task-0.npy')
  for i in range(len(predT)):
    for j in range(len(predT[i])):
      print(predT[i][j])
      predT[i][j] = softmax(predT[i][j])
      print(predT[i][j])
      break
    break


if __name__ == '__main__':
  main()

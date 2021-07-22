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
data = {'Beh_Mic': [], 'Beh_Mac': [], 'His_Mic': [], 'His_Mac': [], 'Lat_Mic': [],
          'Lat_Mac': [], 'Site_Mic': [], 'Site_Mac': [], 'Subs_Mic': [], 'Subs_Mac': []}
header = ['Beh_Mic', 'Beh_Mac', 'His_Mic', 'His_Mac', 'Lat_Mic', 'Lat_Mac', 'Site_Mic',
                              'Site_Mac', 'Subs_Mic', 'Subs_Mac']


def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('file',         type=str,      help='Name of the file we are creating')

  # parse all the argument
  args = parser.parse_args()

  # what are the inputs
  print('data_dir:', args.data_dir, flush= True)

  df = pd.read_csv(args.data_dir + 'MTModel-0_Rank-0/MicMacTest_R0.csv', index_col=False)

  x = df.iloc[1].to_list()
  # x.pop(0)

  print(df)
  print('x:', x.pop(0))
  print(data)



if __name__ == '__main__':
  main()

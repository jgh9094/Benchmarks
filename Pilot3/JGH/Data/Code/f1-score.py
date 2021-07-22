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
from loaddata6reg import loadAllTasks


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

  pred = []
  for t in range(5):
    print(args.data_dir + 'testing-task-' + str(t) + '.npy')
    # get the softmax values of the our predictions from raw logits
    predT = np.load(args.data_dir + 'testing-task-' + str(t) + '.npy')
    p = []
    for i in range(len(predT)):
      p.append(np.argmax(softmax(predT[i])))

    pred.append(np.array(p))

  data_path = args.data_dir + "MicMacTest_R.csv"
  micMac = []
  X, XV, XT, Y, YV, YT= loadAllTasks(print_shapes = False)

  for t in range(5):
    micro = f1_score(YT[:,t], pred[i], average='micro')
    macro = f1_score(YT[:,t], pred[i], average='macro')
    micMac.append(micro)
    micMac.append(macro)

  data = np.zeros(shape=(1, 10))
  data = np.vstack((data, micMac))
  df0 = pd.DataFrame(data,
                     columns=['Beh_Mic', 'Beh_Mac', 'His_Mic', 'His_Mac', 'Lat_Mic', 'Lat_Mac', 'Site_Mic',
                              'Site_Mac', 'Subs_Mic', 'Subs_Mac'])
  df0.to_csv(data_path)



if __name__ == '__main__':
  main()

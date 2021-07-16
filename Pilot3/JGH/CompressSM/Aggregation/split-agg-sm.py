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
# RANK is synonomous with list of splits position
RANK = int(COMM.Get_rank())
SIZE = COMM.size #Node count. size-1 = max rank.

# list of splits
# TESTING
TESTING_T4 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000),
                  (60000, 70000), (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000),
                  (120000, 130000), (130000, 140000), (140000, 150000), (150000, 160000), (160000, 170000), (170000, 180000),
                  (180000, 190000), (190000, 198771)]

TESTING_T1 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000),
                  (60000, 70000), (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000),
                  (120000, 130000), (130000, 140000), (140000, 150000), (150000, 160000), (160000, 170000), (170000, 180000),
                  (180000, 190000), (190000, 198771)]

# VALIDATING
VALID_T1 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000), (60000, 70000),
            (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000), (120000, 130000), (130000, 140000),
            (140000, 150000), (150000, 160000), (160000, 170000), (170000, 178419)]

VALID_T3 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000), (60000, 70000),
            (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000), (120000, 130000), (130000, 140000),
            (140000, 150000), (150000, 160000), (160000, 170000), (170000, 178419)]

VALID_T4 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000), (60000, 70000),
            (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000), (120000, 130000), (130000, 140000),
            (140000, 150000), (150000, 160000), (160000, 170000), (170000, 178419)]

# TRAINING
TRAIN_T1 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000), (60000, 70000),
            (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000), (120000, 130000), (130000, 140000),
            (140000, 150000), (150000, 160000), (160000, 170000), (170000, 180000), (180000, 190000), (190000, 200000), (200000, 210000),
            (210000, 220000), (220000, 230000), (230000, 240000), (240000, 250000), (250000, 260000), (260000, 270000), (270000, 280000),
            (280000, 290000), (290000, 300000), (300000, 310000), (310000, 320000), (320000, 330000), (330000, 340000), (340000, 350000),
            (350000, 360000), (360000, 370000), (370000, 380000), (380000, 390000), (390000, 400000), (400000, 410000), (410000, 420000),
            (420000, 430000), (430000, 440000), (440000, 450000), (450000, 460000), (460000, 470000), (470000, 480000), (480000, 490000),
            (490000, 500000), (500000, 510000), (510000, 520000), (520000, 530000), (530000, 540000), (540000, 550000), (550000, 560000),
            (560000, 570000), (570000, 580000), (580000, 590000), (590000, 600000), (600000, 610000), (610000, 620000), (620000, 630000),
            (630000, 640000), (640000, 650000), (650000, 660000), (660000, 670000), (670000, 680000), (680000, 690000), (690000, 700000),
            (700000, 710000), (710000, 720000), (720000, 730000), (730000, 740000), (740000, 750000), (750000, 760000), (760000, 770000),
            (770000, 780000), (780000, 790000), (790000, 800000), (800000, 810000), (810000, 820000), (820000, 830000), (830000, 840000),
            (840000, 850000), (850000, 860000), (860000, 870000), (870000, 880000), (880000, 890000), (890000, 900000), (900000, 910000),
            (910000, 920000), (920000, 930000), (930000, 940000), (940000, 950000), (950000, 960000), (960000, 970000), (970000, 980000),
            (980000, 990000), (990000, 1000000), (1000000, 1010000), (1010000, 1020000), (1020000, 1030000), (1030000, 1040000), (1040000, 1050000),
            (1050000, 1060000), (1060000, 1070000), (1070000, 1080000), (1080000, 1090000), (1090000, 1100000), (1100000, 1110000), (1110000, 1120000),
            (1120000, 1130000), (1130000, 1140000), (1140000, 1150000), (1150000, 1160000), (1160000, 1170000), (1170000, 1180000), (1180000, 1190000),
            (1190000, 1200000), (1200000, 1210000), (1210000, 1220000), (1220000, 1230000), (1230000, 1240000), (1240000, 1250000), (1250000, 1260000),
            (1260000, 1270000), (1270000, 1280000), (1280000, 1290000), (1290000, 1300000), (1300000, 1310000), (1310000, 1320000), (1320000, 1330000),
            (1330000, 1340000), (1340000, 1350000), (1350000, 1360000), (1360000, 1370000), (1370000, 1380000), (1380000, 1390000), (1390000, 1400000),
            (1400000, 1410000), (1410000, 1420000), (1420000, 1430000), (1430000, 1440000), (1440000, 1450000), (1450000, 1460000), (1460000, 1470000),
            (1470000, 1480000), (1480000, 1490000), (1490000, 1500000), (1500000, 1510000), (1510000, 1520000), (1520000, 1530000), (1530000, 1540000),
            (1540000, 1550000), (1550000, 1560000), (1560000, 1570000), (1570000, 1580000), (1580000, 1590000), (1590000, 1607069)]

TRAIN_T3 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000), (60000, 70000),
            (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000), (120000, 130000), (130000, 140000),
            (140000, 150000), (150000, 160000), (160000, 170000), (170000, 180000), (180000, 190000), (190000, 200000), (200000, 210000),
            (210000, 220000), (220000, 230000), (230000, 240000), (240000, 250000), (250000, 260000), (260000, 270000), (270000, 280000),
            (280000, 290000), (290000, 300000), (300000, 310000), (310000, 320000), (320000, 330000), (330000, 340000), (340000, 350000),
            (350000, 360000), (360000, 370000), (370000, 380000), (380000, 390000), (390000, 400000), (400000, 410000), (410000, 420000),
            (420000, 430000), (430000, 440000), (440000, 450000), (450000, 460000), (460000, 470000), (470000, 480000), (480000, 490000),
            (490000, 500000), (500000, 510000), (510000, 520000), (520000, 530000), (530000, 540000), (540000, 550000), (550000, 560000),
            (560000, 570000), (570000, 580000), (580000, 590000), (590000, 600000), (600000, 610000), (610000, 620000), (620000, 630000),
            (630000, 640000), (640000, 650000), (650000, 660000), (660000, 670000), (670000, 680000), (680000, 690000), (690000, 700000),
            (700000, 710000), (710000, 720000), (720000, 730000), (730000, 740000), (740000, 750000), (750000, 760000), (760000, 770000),
            (770000, 780000), (780000, 790000), (790000, 800000), (800000, 810000), (810000, 820000), (820000, 830000), (830000, 840000),
            (840000, 850000), (850000, 860000), (860000, 870000), (870000, 880000), (880000, 890000), (890000, 900000), (900000, 910000),
            (910000, 920000), (920000, 930000), (930000, 940000), (940000, 950000), (950000, 960000), (960000, 970000), (970000, 980000),
            (980000, 990000), (990000, 1000000), (1000000, 1010000), (1010000, 1020000), (1020000, 1030000), (1030000, 1040000), (1040000, 1050000),
            (1050000, 1060000), (1060000, 1070000), (1070000, 1080000), (1080000, 1090000), (1090000, 1100000), (1100000, 1110000), (1110000, 1120000),
            (1120000, 1130000), (1130000, 1140000), (1140000, 1150000), (1150000, 1160000), (1160000, 1170000), (1170000, 1180000), (1180000, 1190000),
            (1190000, 1200000), (1200000, 1210000), (1210000, 1220000), (1220000, 1230000), (1230000, 1240000), (1240000, 1250000), (1250000, 1260000),
            (1260000, 1270000), (1270000, 1280000), (1280000, 1290000), (1290000, 1300000), (1300000, 1310000), (1310000, 1320000), (1320000, 1330000),
            (1330000, 1340000), (1340000, 1350000), (1350000, 1360000), (1360000, 1370000), (1370000, 1380000), (1380000, 1390000), (1390000, 1400000),
            (1400000, 1410000), (1410000, 1420000), (1420000, 1430000), (1430000, 1440000), (1440000, 1450000), (1450000, 1460000), (1460000, 1470000),
            (1470000, 1480000), (1480000, 1490000), (1490000, 1500000), (1500000, 1510000), (1510000, 1520000), (1520000, 1530000), (1530000, 1540000),
            (1540000, 1550000), (1550000, 1560000), (1560000, 1570000), (1570000, 1580000), (1580000, 1590000), (1590000, 1607069)]

TRAIN_T4 = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, 60000), (60000, 70000),
            (70000, 80000), (80000, 90000), (90000, 100000), (100000, 110000), (110000, 120000), (120000, 130000), (130000, 140000),
            (140000, 150000), (150000, 160000), (160000, 170000), (170000, 180000), (180000, 190000), (190000, 200000), (200000, 210000),
            (210000, 220000), (220000, 230000), (230000, 240000), (240000, 250000), (250000, 260000), (260000, 270000), (270000, 280000),
            (280000, 290000), (290000, 300000), (300000, 310000), (310000, 320000), (320000, 330000), (330000, 340000), (340000, 350000),
            (350000, 360000), (360000, 370000), (370000, 380000), (380000, 390000), (390000, 400000), (400000, 410000), (410000, 420000),
            (420000, 430000), (430000, 440000), (440000, 450000), (450000, 460000), (460000, 470000), (470000, 480000), (480000, 490000),
            (490000, 500000), (500000, 510000), (510000, 520000), (520000, 530000), (530000, 540000), (540000, 550000), (550000, 560000),
            (560000, 570000), (570000, 580000), (580000, 590000), (590000, 600000), (600000, 610000), (610000, 620000), (620000, 630000),
            (630000, 640000), (640000, 650000), (650000, 660000), (660000, 670000), (670000, 680000), (680000, 690000), (690000, 700000),
            (700000, 710000), (710000, 720000), (720000, 730000), (730000, 740000), (740000, 750000), (750000, 760000), (760000, 770000),
            (770000, 780000), (780000, 790000), (790000, 800000), (800000, 810000), (810000, 820000), (820000, 830000), (830000, 840000),
            (840000, 850000), (850000, 860000), (860000, 870000), (870000, 880000), (880000, 890000), (890000, 900000), (900000, 910000),
            (910000, 920000), (920000, 930000), (930000, 940000), (940000, 950000), (950000, 960000), (960000, 970000), (970000, 980000),
            (980000, 990000), (990000, 1000000), (1000000, 1010000), (1010000, 1020000), (1020000, 1030000), (1030000, 1040000), (1040000, 1050000),
            (1050000, 1060000), (1060000, 1070000), (1070000, 1080000), (1080000, 1090000), (1090000, 1100000), (1100000, 1110000), (1110000, 1120000),
            (1120000, 1130000), (1130000, 1140000), (1140000, 1150000), (1150000, 1160000), (1160000, 1170000), (1170000, 1180000), (1180000, 1190000),
            (1190000, 1200000), (1200000, 1210000), (1210000, 1220000), (1220000, 1230000), (1230000, 1240000), (1240000, 1250000), (1250000, 1260000),
            (1260000, 1270000), (1270000, 1280000), (1280000, 1290000), (1290000, 1300000), (1300000, 1310000), (1310000, 1320000), (1320000, 1330000),
            (1330000, 1340000), (1340000, 1350000), (1350000, 1360000), (1360000, 1370000), (1370000, 1380000), (1380000, 1390000), (1390000, 1400000),
            (1400000, 1410000), (1410000, 1420000), (1420000, 1430000), (1430000, 1440000), (1440000, 1450000), (1450000, 1460000), (1460000, 1470000),
            (1470000, 1480000), (1480000, 1490000), (1490000, 1500000), (1500000, 1510000), (1510000, 1520000), (1520000, 1530000), (1530000, 1540000),
            (1540000, 1550000), (1550000, 1560000), (1560000, 1570000), (1570000, 1580000), (1580000, 1590000), (1590000, 1607069)]


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
def AverageData(dirs,task,dump,data,p,R):
  # get training data
  print('AVERAGING',data.upper(),'DATA...', flush= True)
  print('P:', p, flush=True)

  X = np.load(file=dirs[0] + data +'-task-' + str(task) + '.npy')[p[0]:p[1],]

  # matrix that will
  mat = np.zeros(shape=(X.shape[0],X.shape[1]))
  print('mat.shape', mat.shape, flush=True)

  del X

  print('PROCESSING FILE', flush=True)
  for dir in dirs:
    print('processing[',str(R),']:', dir + data +'-task-' + str(task) + '.npy', flush= True)
    X = np.load(file=dir + data +'-task-' + str(task) + '.npy')[p[0]:p[1],]

    # iteratate through each file and update the matrix
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        mat[i][j] += X[i][j]

    del X

  # divide all elements in matrix by number of models
  mat = np.array([m / float(len(dirs)) for m in mat])

  # memory checks
  print('memory:',psutil.virtual_memory(), flush= True)

  np.save(dump + data + '-task-' + str(task) + '-rank-' + str(R) +'.npy', mat)
  print('finished saving:', dump + data + '-task-' + str(task) + '-rank-' + str(R) +'.npy', flush= True)
  print(flush= True)

def main():
  # generate and get arguments
  parser = argparse.ArgumentParser(description='Process arguments for model training.')
  parser.add_argument('data_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('dump_dir',     type=str,      help='Where are we dumping the output?')
  parser.add_argument('proportion',   type=float,    help='What model config was used')
  parser.add_argument('task',       type=int,      help='Seed offset for rng')
  parser.add_argument('data_type',    type=int,      help='0: training, 1: testing, 2: validating')

  # parse all the argument
  args = parser.parse_args()

  # get task
  task = args.task
  print('task:', task, flush= True)
  print('dump_dir:', args.dump_dir)
  print('proportion:', args.proportion)
  print('task:', args.task)
  print('data_type:', args.data_type)


  # set seed for rng
  seed = int(task)
  print('RANDOM SEED:', seed, flush= True)
  np.random.seed(seed)

  # Step 1: Get data directories we are exploring
  dirs = GetDataDirs(args.data_dir.strip(),args.proportion)

  # RANK = 0 # local testing only

  if task == 1:
    # check what data type we are looking for
    if args.data_type == 0:
      AverageData(dirs,task,args.dump_dir, 'training', TRAIN_T1[RANK], RANK)
    elif args.data_type == 1:
      AverageData(dirs,task,args.dump_dir, 'testing', TESTING_T1[RANK],RANK)
    elif args.data_type == 2:
      AverageData(dirs,task,args.dump_dir, 'validating', VALID_T1[RANK], RANK)
    else:
      print('ERROR UNKNOWN DATA TYPE')
      exit(-1)

  elif task == 3:
    # check what data type we are looking for
    if args.data_type == 0:
      AverageData(dirs,task,args.dump_dir, 'training', TRAIN_T3[RANK], RANK)
    elif args.data_type == 1:
      print('ALREADY DID TASK', str(task))
      exit(-1)
    elif args.data_type == 2:
      AverageData(dirs,task,args.dump_dir, 'validating', VALID_T3[RANK], RANK)
    else:
      print('ERROR UNKNOWN DATA TYPE')
      exit(-1)

  elif task == 4:
    # check what data type we are looking for
    if args.data_type == 0:
      AverageData(dirs,task,args.dump_dir, 'training', TRAIN_T4[RANK], RANK)
    elif args.data_type == 1:
      AverageData(dirs,task,args.dump_dir, 'testing', TESTING_T4[RANK],RANK)
    elif args.data_type == 2:
      AverageData(dirs,task,args.dump_dir, 'validating', VALID_T4[RANK], RANK)
    else:
      print('ERROR UNKNOWN DATA TYPE')
      exit(-1)



if __name__ == '__main__':
  main()

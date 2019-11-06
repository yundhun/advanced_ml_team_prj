# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:57:01 2019

@author: John
"""

import pandas as pd
import os
from shutil import copyfile

def calculate_rank(vector):
  a={}
  rank=1
  for num in sorted(vector):
    if num not in a:
      a[num]=rank
      rank=rank+1
  return[a[i] for i in vector]

nof = 1000 #number of files
tr_csv = pd.read_csv('images/boneage-training-dataset.csv')
tr_file_names = [ 'images/rsna-bone-age/boneage-training-dataset/' + str(s) + '.png' for s in list(tr_csv.loc[:nof ,'id']) ]

tr_labels = list( tr_csv.loc[: ,'boneage'] )
tr_rank_labels = calculate_rank(tr_labels)

for i, file_name in enumerate(tr_file_names):
    print(i, file_name)
    directory = 'images/rsna-bone-age/boneage-training-dataset/torchDataset/'+str(tr_rank_labels[i])
    if not os.path.exists(directory):
        os.makedirs(directory)
    copyfile(file_name, directory+'/'+str.split(file_name,'/')[-1])
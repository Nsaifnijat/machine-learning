# -*- coding: utf-8 -*-

from math import sqrt
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data,predict,k=3):
    
    if len(data)>=k:
        warnings.warn('K is set to a value less than total voting groups')
        
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes=[i[1] for i in sorted(distances)[:k]]
    vote_result=Counter(votes).most_common(1)[0][0]
    #to check the nummber of percentage on the points that we are not confident and want to guess
    confidence=Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence    

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
#data has to be float,because the labels are inside quotes
full_data=df.astype(float).values.tolist()
random.shuffle(full_data)

test_size=0.2
train_set={2:[], 4:[]}
test_set={2:[], 4:[]}

train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]
for i in train_data:
    train_set[i[-1]].append(i[:-1])
  
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        vote, confidence=k_nearest_neighbors(train_set,data,k=5)
        if group == vote:
            correct += 1
        #to see the vote on incorrect ones
        else:
            print(confidence)
        total +=1
        
print('Accuracy:', correct/total)














# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np

plot1=[1,3]
plot2=[2,5]


euclidean_distance=sqrt((plot1[0]-plot2[0])**2 +(plot1[1]-plot2[1])**2)

print(euclidean_distance)

#euclidean for multi dimension

euclidean_distance2=np.sqrt(np.sum((np.array(features)-np.array(predict))**2))

#using numpy builtin method
euclidean_distance3=np.linalg.norm(np.array(features)-np.array(predict))












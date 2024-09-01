# -*- coding: utf-8 -*-
'''
kmeans is a flat clustering algorithm which lies under unsupervised
machine learning, where as in supervised learning we had classification
one basic feature of kmeans is that we tell the machine to make n numbers of  cluseters
but in hierarchical (mean shift) algorithms we let the computer decide how many clusters is needed or good
mean shift is a hierarchical clustering machine learning algorithm'''
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
import numpy as np

X=np.array([[1,2],
            [1.5,1.8],
            [5,8],
            [8,8],
            [1,0.6],
            [9,11]])


#plt.scatter(X[:,0],X[:,1],s=150,linewidths=5)
#plt.show()
#n_cluster is the number of cluster you divide to
clf=KMeans(n_clusters=6)
clf.fit(X)
#centroids are the cluster centers that the machine clusters data around it as one cluster, they have labels
centroids=clf.cluster_centers_
labels=clf.labels_
colors=10*['g.','r.','c.','b.','k.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1],colors[labels[i]],markersize=10)
    
plt.scatter(centroids[:,0], centroids[:,1], marker='x',s=150,linewidths=5)

plt.show()
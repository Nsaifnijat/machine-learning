# -*- coding: utf-8 -*-




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



x_train,x_test,y_train,y_test=train_test_split(x,y)


#to make the feature values fit in between 0 and 1

scaler=StandardScaler()
#here we have applied the scaler to the data but we need to transform in order to get actual transformation
x_train_scaleReady=scaler.fit(x_train)
x_test_scaledReady=scaler.fit(x_test)

#standard scaler methods
x_train_scaleReady.mean_
x_train_scaleReady.scale_ #it gives the variance, or signed std
x_train_scaleReady.std_


x_train_scaled=x_train_scaleReady.transform(x_train)
x_test_scaled=x_test_scaleReady.transform(x_test)


x_train_scaled.mean(axis=0)



#using normalization for data scaling


from sklearn import preprocessing
import numpy as np

x=np.array([[0,30],
           [2,40],
           [3,50]])

#option1, by default axis is 1, which means sum of the rows of the normalized array has to become one
#but if we change the axis to 0, then the sum of cols has to become 1
x_norm_L1=preprocessing.normalize(x,norm='l1')
print(x_norm_L1)

#in L2 np.sqrt(row1**2+row2**2) has to become 1 or close to it
x_norm_L2=preprocessing.normalize(x,norm='l2')
print(x_norm_L2)

#options 2, 
transformer=preprocessing.Normalizer(norm='l1')
x_norm_l1=transformer.fit_transform(x)
print(x_norm_L1)

#l2 
transformer=preprocessing.Normalizer(norm='l2')
x_norm_l2=transformer.fit_transform(x)
print(x_norm_L2)



#using scale for preprocessing

x_norm=preprocessing.scale(x)
print(x_norm)

# the following methods are also used
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


x_nom=StandardScaler().fit_transform(x)
print(x_norm)
x_nom=MinMaxScaler().fit_transform(x)
print(x_norm)
x_nom=MaxAbsScaler().fit_transform(x)
print(x_norm)
#the above three methods are sensitve to outliers

x_nom=RobustScaler(quantile_range=(25, 75)).fit_transform(x)
print(x_norm)

x_nom=PowerTransformer(method="yeo-johnson").fit_transform(x)
print(x_norm)

x_nom=PowerTransformer(method="box-cox").fit_transform(x)
print(x_norm)
 
x_nom=QuantileTransformer(output_distribution="uniform").fit_transform(x)
print(x_norm)

x_nom=QuantileTransformer(output_distribution="normal").fit_transform(x)
print(x_norm)

x_nom=Normalizer().fit_transform(x)
print(x_norm)























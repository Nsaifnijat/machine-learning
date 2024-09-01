# -*- coding: utf-8 -*-

#LabelEncoder, or assigning numbers to each Categorical values
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

df=pd.read_csv('car.data')
#to check which column has na vals
print(df.isna().sum())
print(df)
#to exclude na vals from a column in the dataframe
#df=df.loc[df.buying.notna(),['buying','main']]
#the following changes df to numpy array
df=df.iloc[:,:].values
#this type of encoding can bias our data and may change the result,
labl_encoder=LabelEncoder()
#here we give the columns with categorical data,doing it on array
df[:,5]=labl_encoder.fit_transform(df[:,5])
print(df)

#another way to do the label encoding is to do it by hand using mapping whichi is timeconsuming



#option two, onehot encoder which is best for this activity, and it can avoid bias of machine

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#onehot encoding is also called dummy encoding
#remainder, can get two parameters, drop and passthrough, we do ohe on column 5
ct=ColumnTransformer( transformers=[('saifkhan',OneHotEncoder(),[5])],remainder='passthrough')
#now lets apply it to our df array
df=ct.fit_transform(df)
#changing df to dataframe to see it better
df=pd.DataFrame(df)
print(df)

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer

ohe=OneHotEncoder()
#simple imputer fills the na vals with mean of the column by default
imp=SimpleImputer()
#applying both na filler and categorical data transformer to the data, for na filler the columns has to be int or float
ct=make_column_transformer((ohe,['buying','maint','safety']),(imp,['persons']), remainder='passthrough')

transformed_df=ct.fit_transform(df)

print(transformed_df)

#seven ways to use onehot encoder on pandas, and some ways for numpy
from sklearn.compose import make_column_selector

ohe=OneHotEncoder()
#this for pandas only, make column default remainder is drop
ct=make_column_transformer((ohe,['buying','maint']))
#both numpy and pandas, columns 1 and 2
ct=make_column_transformer([ohe,[1,2]])
#column 1 through 3, inclusive of 1 and exclusive of 3
ct=make_column_transformer((ohe,slice(1,3)))
#the columns you want to select need to be true
ct=make_column_transformer((ohe,[False,True,True,False]))
#regex, anyting whic starts capital E AND S
ct=make_column_transformer((ohe,make_column_selector(pattern='E/S')))
#it selects only the object cols or exclude the objects
ct=make_column_transformer((ohe,make_column_selector(dtype_include=object)))
ct=make_column_transformer((ohe,make_column_selector(dtype_exclude=object)))

ct.fit_transform(df)

print(df)

#to show the categories
print(ohe.categories_)













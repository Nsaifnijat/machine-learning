# -*- coding: utf-8 -*-

#Step 1: Import the libraries

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
  
# For data manipulation
import pandas as pd
import numpy as np
  
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
  
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Step 2: Read Stock  data

# Read the csv file using read_csv 
# method of pandas
df = pd.read_csv('RELIANCE.csv')
print(df)

#Step 3: Data Preparation 


# Changes The Date column as index columns
df.index = pd.to_datetime(df['Date'])
print(df)

  
# drop The original date column
df = df.drop(['Date'], axis='columns')
print(df)


#Step 4: Define the explanatory variables

# Create predictor variables
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low
  
# Store all predictor variables in a variable X
X = df[['Open-Close', 'High-Low']]
X.head()

#Step 5: Define the target variable

'''
here we use numpy to create our target which is if price goes up 1 if goes down 0
Syntax:
np.where(condition,value_if_true,value_if_false)
'''
# Target variables
y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


#Step 6: Split the data into train and test

split_percentage = 0.8
split = int(split_percentage*len(df))
  
# Train data set
X_train = X[:split]
y_train = y[:split]
  
# Test data set
X_test = X[split:]
y_test = y[split:]



#Step 7: Support Vector Classifier (SVC)
'''
We will use SVC() function from sklearn.svm.SVC library to create our classifier model using the fit() 
method on the training data set.
'''
# Support vector classifier
cls = SVC().fit(X_train, y_train)



#Step 8: Classifier accuracy
'''
We will compute the accuracy of the algorithm on the train and test the data set 
by comparing the actual values of the signal with the predicted values of the signal.
 The function accuracy_score() will be used to calculate the accuracy.
An accuracy of 50%+ in test data suggests that the classifier model is effective.
'''


#Step 9: Strategy implementation
'''
We will predict the signal (buy or sell) using the cls.predict() function.
'''
df['Predicted_Signal'] = cls.predict(X)


# Calculate daily returns
df['Return'] = df.Close.pct_change()



# Calculate strategy returns
df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)



# Calculate Cumulutive returns
df['Cum_Ret'] = df['Return'].cumsum()
df

# Plot Strategy Cumulative returns 
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
df


#Plot Strategy Returns vs Original Returns
import matplotlib.pyplot as plt
  
plt.plot(df['Cum_Ret'],color='red')
plt.plot(df['Cum_Strategy'],color='blue')















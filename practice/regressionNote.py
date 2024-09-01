# -*- coding: utf-8 -*-
'''


Link of the whole following data.
https://realpython.com/train-test-split-python-data/



Training, Validation, and Test Sets
Splitting your dataset is essential for an unbiased evaluation of prediction performance. In most cases, it’s enough 
to split your dataset randomly into three subsets:

The training set is applied to train, or fit, your model. For example, you use the training set to find the optimal weights,
or coefficients, for linear regression, logistic regression, or neural networks.

The validation set is used for unbiased model evaluation during hyperparameter tuning. For example, when you want to find
the optimal number of neurons in a neural network or the best kernel for
a support vector machine, you experiment with different values. For each considered setting of hyperparameters,
you fit the model with the training set and assess its performance with the validation set.
The test set is needed for an unbiased evaluation of the final model. You shouldn’t use it for fitting or validation.

sklearn.model_selection.train_test_split(*arrays, **options)

Example: x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=4, stratify=y)



in the above train test split 
Arrays are:
In supervised machine learning applications, you’ll typically work with two such sequences:

1-A two-dimensional array with the inputs (x)
2-A one-dimensional array with the outputs (y)

Options are:
    the optional keyword arguments that you can use to get desired behavior:

train_size - is the number that defines the size of the training set. If you provide a float,
 then it must be between 0.0 and 1.0 and will define the share of the dataset used for testing. 
 If you provide an int, then it will represent the total number of the training samples. The default value is None.

test_size - is the number that defines the size of the test set. It’s very similar to train_size.
 You should provide either train_size or test_size. If neither is given, then the default share of the dataset that will 
 be used for testing is 0.25, or 25 percent.

random_state -is the object that controls randomization during splitting.
 It can be either an int or an instance of RandomState. The default value is None.

shuffle - is the Boolean object (True by default) that determines whether to shuffle the dataset before applying the split.

stratify - is an array-like object that, if not None, determines how to use a stratified split.
            the stratigy balance the categories in the y_test


With linear regression, fitting the model means determining the best intercept (model.intercept_) and 
slope (model.coef_) values of the regression line.

Although you can use x_train and y_train to check the goodness of fit, this isn’t a best practice.
 An unbiased estimation of the predictive performance of your model is based on test data:
  
Given two sequences, like x and y here, train_test_split() performs the split and returns four sequences
 (in this case NumPy arrays) in this order:

x_train, x_test, y_train, y_test

Note:
    You probably got different results from what you see here. This is because dataset splitting is random by default.
    The result differs each time you run the function,
    so if you want to make is same in every call then random_state parameter has to be a positive int or a random state
    
The samples of the dataset are shuffled randomly and then split into the training and test sets according to the size you 
defined.

You can see that y has six zeros and six ones. However, the test set has three zeros out of four items. 
If you want to (approximately) keep the proportion of y values through the training and test sets, then pass stratify=y.



you can turn off data shuffling and random split with shuffle=False:
With linear regression, fitting the model means determining the best intercept (model.intercept_) 
and slope (model.coef_) values of the regression line.     
  
     
.score() returns the coefficient of determination, or R², for the data passed. 
Its maximum is 1. The higher the R² value, the better the fit. In this case,
 the training data yields a slightly higher coefficient. However, the R² calculated with test data is an
 unbiased measure of your model’s prediction performance.   
     

Validation data vs. testing data
Not all data scientists rely on both validation data and testing data.
 To some degree, both datasets serve the same purpose: make sure the model works on real data.

However, there are some practical differences between validation data and testing data. 
If you opt to include a separate stage for validation data analysis, this dataset is typically labeled so the data 
scientist can collect metrics that they can use to better train the model. In this sense, validation data occurs as 
part of the model training process. Conversely, the model acts as a black box when you run testing data through it. 
Thus, validation data tunes the model, whereas testing data simply confirms that it works.

The nature of the dependent variables differentiates regression and classification problems.
 Regression problems have continuous and usually unbounded outputs. An example is when you’re estimating 
 the salary as a function of experience and education level. On the other hand, classification problems have discrete 
 and finite outputs called classes or categories. For example, predicting if an employee is going to be promoted or not
 (true or false) is a classification problem.
 
Validation Functionalities
The package sklearn.model_selection offers a lot of functionalities related to model selection and validation, 
including the following:

Cross-validation
Learning curves
Hyperparameter tuning
Cross-validation is a set of techniques that combine the measures of prediction performance to get more accurate model estimations.

One of the widely used cross-validation methods is k-fold cross-validation. In it, you divide your dataset into 
k (often five or ten) subsets, or folds, of equal size and then perform the training and test procedures k times. 
Each time, you use a different fold as the test set and all the remaining folds as the training set. This provides k 
measures of predictive performance, and you can then analyze their mean and standard deviation.

You can implement cross-validation with KFold, StratifiedKFold, LeaveOneOut, and a few other classes and functions from 
sklearn.model_selection.

A learning curve, sometimes called a training curve, shows how the prediction score of training and validation sets 
depends on the number of training samples. You can use learning_curve() to get this dependency, which can help you find 
the optimal size of the training set, choose hyperparameters, compare models, and so on.

Hyperparameter tuning, also called hyperparameter optimization, is the process of determining the best set of
 hyperparameters to define your machine learning model. sklearn.model_selection provides you with several options for 
 this purpose, including GridSearchCV, RandomizedSearchCV, validation_curve(), and others. Splitting your data is also 
 important for hyperparameter tuning.


'''
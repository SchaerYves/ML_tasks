
import numpy as np
import matplotlib.pyplot as plt

##Project files.
#from util import gradient_descent, generate_polynomial_data
#import plot_helpers
#from regressors import LinearRegressor
#from regularizers import Regularizer, L2Regularizer

# Widget and formatting modules
import ipywidgets
from ipywidgets import interact, interactive, interact_manual
import pylab
# If in your browser the figures are not nicely vizualized, change the following line. 
pylab.rcParams['figure.figsize'] = (10, 5)

# Machine Learning library. 

import sklearn
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
# from sklearn import datasets, linear_model


#train_all=pd.read_csv('data/train.csv', sep=',', index_col=0)
##
#X=train_all.iloc[:,1:]
##
#y=train_all.iloc[:,0:1]

with open('data/train.csv') as train:
    trainData= np.genfromtxt(train, delimiter=",")
 
trainData=np.delete(trainData, 0,0)    
trainData=scale(trainData)
X=trainData[:,2:]

y=trainData[:,[1]]
#
i=0
q=0
reg_coef=np.ones((21, 1))
#use single feature
for i in range(0, 5):
    
    use_feature = i
    # Use only one feature
    
    X_feat = X[:, np.newaxis, use_feature]
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training set
    regr.fit(X_feat, y)
    
    # Make predictions on the testing set
    #Y_pred = regr.predict(X_feat)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    reg_coef[q]=regr.coef_
    q+=1
    
for i in range(0, 5):
    
    use_feature = i
    # Use only one feature
    
    X_feat = (X[:, np.newaxis, use_feature])**2
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training set
    regr.fit(X_feat, y)
    
    # Make predictions on the testing set
    #Y_pred = regr.predict(X_feat)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    reg_coef[q]=regr.coef_
    q+=1


for i in range(0, 5):
    
    use_feature = i
    # Use only one feature
    
    X_feat = np.exp(X[:, np.newaxis, use_feature])
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training set
    regr.fit(X_feat, y)
    
    # Make predictions on the testing set
    #Y_pred = regr.predict(X_feat)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    reg_coef[q]=regr.coef_
    q+=1
    
for i in range(0, 5):
    
    use_feature = i
    # Use only one feature
    
    X_feat = np.cos(X[:, np.newaxis, use_feature])
    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training set
    regr.fit(X_feat, y)
    
    # Make predictions on the testing set
    #Y_pred = regr.predict(X_feat)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    reg_coef[q]=regr.coef_
    q+=1


use_feature = i
# Use only one feature

X_feat =np.ones((900,1))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training set
regr.fit(X_feat, y)

# Make predictions on the testing set
#Y_pred = regr.predict(X_feat)

# The coefficients
print('Coefficients: \n', regr.coef_)
reg_coef[q]=regr.coef_



with open('submission1.csv', 'wb') as f:
  np.savetxt(f, reg_coef, fmt='%1.48f')



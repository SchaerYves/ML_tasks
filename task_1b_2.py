
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

#import sklearn
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
#from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
# from sklearn import datasets, linear_model


#open file
with open('data/train.csv') as train:
    trainData= np.genfromtxt(train, delimiter=",")
 
trainData=np.delete(trainData, 0,0)    
#trainData=scale(trainData)
X=trainData[:,2:]
y=trainData[:,[1]]

reg_coef=np.ones((21, 1))

X= np.hstack((X, X**2, np.exp(X), np.cos(X), np.ones((900,1))))
# Create linear regression object
#regr = linear_model.RidgeCV(alphas=(0.1, 1, 10, 100, 1000, 10000, 100000),cv=10, fit_intercept=False)
regr = linear_model.Ridge(alpha=1000, fit_intercept=False)
# Train the model using the training set
regr.fit(X, y)

#print(regr.alpha)

# The coefficients
print('Coefficients: \n', regr.coef_)

with open('submission7.csv', 'wb') as f:
  np.savetxt(f, np.transpose(regr.coef_), fmt='%1.48f')


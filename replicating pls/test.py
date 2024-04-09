# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:50:28 2020

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:13:15 2020

@author: Kashif
"""


import numpy as np
from sklearn import preprocessing
import os, sys, traceback
import logging
import json
import datetime
import pandas as pd
import pickle
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from logging.handlers import RotatingFileHandler
from sklearn.metrics import mean_squared_error, r2_score
import csv
import shutil

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
import seaborn as sns
sns.set()

import statistics
from scipy.stats import f as f_distrib
import scipy.stats
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet


folderName = os.path.dirname(__file__)
df_x_y = pd.read_csv(os.path.join(folderName,'input','test.csv'))
df_x = df_x_y.drop(['Y'], axis=1)
df_y=df_x_y[['Y']]
df_x.apply(pd.to_numeric)
df_y.apply(pd.to_numeric)


X=df_x.copy(deep=True)
X=X.values

Y=df_y.copy(deep=True)
Y=Y.values



X_mah=X

scaler1 = StandardScaler()
scaler1.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
#X = scaler1.transform(X)
    
#------------------------------PLS--------------------------------
n_components=2
pls = PLSRegression(n_components)
pls.fit(X, Y)
X_pls=pls.transform(X,Y)
pred_Y=pls.predict(X)
X_pls=X_pls[0]

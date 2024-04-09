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
from scipy.optimize import minimize

sns.set()

import statistics
from scipy.stats import f as f_distrib
import scipy.stats
import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet


folderName = os.path.dirname(__file__)
df_x_y = pd.read_csv(os.path.join(folderName,'input','input.csv'))
df_x = df_x_y.drop(['Date', 'PDI10801.PV'], axis=1)
df_y=df_x_y[['PDI10801.PV']]
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
n_components=3
pls = PLSRegression(n_components)
pls.fit(X, Y)
X_pls=pls.transform(X,Y)
pred_Y=pls.predict(X)


#----------------------------------------coeffecients calculations and intercept  -------

W=pls.x_weights_
P=pls.x_loadings_
C=pls.y_loadings_
coeff_standardized=np.dot(np.dot(W,(np.linalg.inv(np.dot(P.T,W)))),C.T)
coeff_nonstandardized=coeff_standardized[:,0]*pls.y_std_/pls.x_std_
intercept=pls.y_mean_-np.sum(coeff_nonstandardized*pls.x_mean_,axis=0)

#----------------------------------------coeffecients calculations and intercept  -------

#-----------------optimize-------------------

i=999
def objective(x):
    return (np.dot(x,coeff_nonstandardized)+intercept)[0]

def constaint_SHC(x):
    return x[17]-0.35

def constraint_feed_temp(x):
    return x[14]-440


def constraint_T2(x):
    PCA_red=X_pls[0].copy()
    T2=[]
    k=PCA_red.shape[1]
    n=PCA_red.shape[0]
    mue=np.mean(PCA_red, axis=0)
    F_critical=scipy.stats.f.ppf(0.95,k,n-k)
    
    xbar=np.zeros((n_components,1))
    cov_mat=np.cov(X_pls[0].T)
    cov_inve=np.linalg.inv(cov_mat)
    i=999
    #xbar[:,0]=PCA_red[i]-mue
    
    x=x.reshape(1,18)
    x=pls.transform(x)
    
    
    xbar[:,0]=x-mue
    xbarT=xbar.T
    c=xbarT.dot(cov_inve)
    hotel=c.dot(xbar)*1
    print(f'hotelling T2 value is {hotel[0][0]}')
    lim_T2_99=k*(n**2-1)/(n*(n-k))*scipy.stats.f.ppf(0.99,k,n-k)
    print(f'T lim 99% is {lim_T2_99}')
    print(lim_T2_99-hotel[0][0])
    return lim_T2_99-hotel[0][0]
    
    

   
x0=X[i]
print(constraint_T2(x0))
print(objective(x0))
 

#b=[1,5]


#bnds=[b,b,b,b]
con1={'type':'eq','fun':constaint_SHC}
con2={'type':'eq','fun':constraint_feed_temp}
con3={'type':'ineq','fun':constraint_T2}
cons=[con1,con2,con3]
#sol=minimize(objective, x0,method='SLSQP',bounds=bnds,constraints=cons)
sol=minimize(objective, x0,method='SLSQP',constraints=cons)

#---------- --------------------PLS--------------------------------

value_predicted_y=np.dot(X,coeff_nonstandardized)+intercept


#--------------------------------------------------------Hotelling T2 test-----------------------------------
PCA_red=X_pls[0].copy()
T2=[]
k=PCA_red.shape[1]
n=PCA_red.shape[0]
mue=np.mean(PCA_red, axis=0)
F_critical=scipy.stats.f.ppf(0.95,k,n-k)

xbar=np.zeros((n_components,1))
cov_mat=np.cov(X_pls[0].T)
cov_inve=np.linalg.inv(cov_mat)

for i in range(PCA_red.shape[0]):
    xbar[:,0]=PCA_red[i]-mue
    xbarT=xbar.T
    c=xbarT.dot(cov_inve)
    hotel=c.dot(xbar)*1
    print(hotel[0][0])
    T2.append(hotel[0][0])

T2 = pd.Series(T2)

lim_T2_95=k*(n**2-1)/(n*(n-k))*scipy.stats.f.ppf(0.95,k,n-k)
lim_T2_99=k*(n**2-1)/(n*(n-k))*scipy.stats.f.ppf(0.99,k,n-k)


#--------------------------------------------------------Hotelling T2 test-----------------------------------




#------------------------------------------------------- sq pred error---------------------
TP=X_pls.dot(pls.x_loadings_.T)
Residuals=X-TP
Residuals_squared=Residuals**2
MSE=Residuals_squared.sum(axis=1)
v=statistics.variance(MSE)
m=statistics.mean(MSE)

chi_coeff=2*m**2/v
lim_chi2_mse_95=v/2/m*scipy.stats.chi2.ppf(1-.05, df=chi_coeff)
lim_chi2_mse_99=v/2/m*scipy.stats.chi2.ppf(1-.01, df=chi_coeff)

#------------------------------------------------------- sq pred error---------------------



#--------------------------------------------------Score plotting------------------------------------------
fig_1=plt.figure(1)
chart_1=fig_1.add_subplot(111)
chart_1.scatter(X_pls[:, 0], X_pls[:, 1])
for i, txt in enumerate(df_x.index):
    chart_1.annotate(int(txt)+1,(X_pls[i, 0], X_pls[i, 1]))
chart_1.set_ylim(-6,6)
chart_1.set_xlim(-6,6)
plt.show()
#--------------------------------------------------Score plotting------------------------------------------


#--------------------------------------------------Loadings plotting------------------------------------------
fig_1=plt.figure(2)
chart_1=fig_1.add_subplot(111)
chart_1.scatter(pls.x_loadings_.T[0, :], pls.x_loadings_.T[1, :]*-1)
for i, txt in enumerate(df_x):
    chart_1.annotate(txt,(pls.x_loadings_.T[0, i], pls.x_loadings_.T[1, i]*-1))
plt.show()
#--------------------------------------------------Loadings plotting------------------------------------------



#--------------------------Hotelling T2 plotting--------------------------------------------------------------
fig_1=plt.figure(3)
chart_1=fig_1.add_subplot(111)
chart_1.scatter(df_x.index+1, T2,label="sample points")
low_lim_T2=[]
hi_lim_T2=[]
for i, txt in enumerate(df_x.index):
    chart_1.annotate(int(txt)+1,(i+1, T2[i]))
    low_lim_T2.append(lim_T2_95)
    hi_lim_T2.append(lim_T2_99)
chart_1.plot(df_x.index+1,low_lim_T2,label="95% confidence interval")
chart_1.plot(df_x.index+1,hi_lim_T2,label="99% confidence interval")   
chart_1.set_title('Hotelling T2 Test')
chart_1.set_xlabel('Sample points')
chart_1.set_ylabel('T2') 
chart_1.legend()
plt.show()

#--------------------------Hotelling T2 plotting--------------------------------------------------------------


#---------------------------SPE plotting-------------------------------------------------------------
fig_1=plt.figure(4)
chart_1=fig_1.add_subplot(111)
chart_1.scatter(range(n),MSE,label="Sample points")
hi_lim_chi2_mse_99=[]
low_lim_chi2_mse_95=[]
for i,text in enumerate(df_x.index):
    chart_1.annotate(int(text)+1,(i+1,MSE[i]))
    hi_lim_chi2_mse_99.append(lim_chi2_mse_99)
    low_lim_chi2_mse_95.append(lim_chi2_mse_95)
chart_1.plot(range(n),hi_lim_chi2_mse_99,label="99% confidence interval")
chart_1.plot(range(n),low_lim_chi2_mse_95,label="95% confidence interval")
chart_1.set_title('SPE Test')
chart_1.set_xlabel('Sample points')
chart_1.set_ylabel('SPE') 
chart_1.legend(loc='upper left',)


plt.show()
#---------------------------SPE plotting------------------------------------------------------------


#-------------------------------------Contributions Plotting----------------------------------------

observation_number=51
Contribution_SPE=Residuals_squared[observation_number]*np.sign(Residuals[observation_number])
x_tags_names=df_x.columns

fig_1=plt.figure(5)
chart_1=fig_1.add_subplot(111)
chart_1.bar(x_tags_names,Contribution_SPE)
chart_1.set_title('Contibution Plot of observation number'+str(int(observation_number)+1))
chart_1.set_xlabel('Variables')
chart_1.set_ylabel('Contributions')
plt.xticks(rotation=90)
plt.show()


#-------------------------------------Contributions Plotting----------------------------------------


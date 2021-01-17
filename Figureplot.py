#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 21:02:50 2021

@author: yuxiushao

Figure plot and data analysis
"""

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from numpy import linalg as la
from scipy.optimize import fsolve
from scipy import linalg as scpla
# import seaborn as sb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve,leastsq 
from math import tanh,cosh
import math
import time
# from sympy import *
from scipy.linalg import schur, eigvals
extras_require = {'PLOT':['matplotlib>=1.1.1,<3.0']},


''' loading data '''
data = np.load('2021-01-15-20_54_54_homogeneousR2_savedata.npz')
# Jparameters = np.zeros(4+1)
# Jparameters[0],Jparameters[1],Jparameters[2],Jparameters[3]=JEE,JEI,JIE,JII
# Jparameters[4]=NE
# np.savez(now+'_homogeneous_savedata',Jparameters=Jparameters,eigvecxt_series=eigvecxt_series,sglmxt_series
# =sglmxt_series,sglnxt_series=sglnxt_series,eigvoutlier_series=eigvoutlier_series,simradius_series=simradius_series)

Jparameters=data['Jparameters']
NE = int(Jparameters[4])
print(NE)
P = np.zeros((2,2))
P = np.array([[Jparameters[0],-Jparameters[1]],[Jparameters[2],-Jparameters[3]]])
Am=np.zeros((NE*2,NE*2))
Am[:NE,:NE],Am[:NE,NE:]=P[0,0]/NE,P[0,1]/NE
Am[NE:,:NE],Am[NE:,NE:]=P[1,0]/NE,P[1,1]/NE
eigvorg,eigvecorg=la.eig(Am)
eigvecxt_series=data['eigvecxt_series']
eigvoutlier_series=data['eigvoutlier_series']
ngavg,nrank,nbatch=20,2,50
gaverageseries=np.linspace(0.05,1.0,20)

# calculate lambda_1 and lambda_2
mean_lambda_r2,std_lambda_r2=np.zeros(ngavg),np.zeros(ngavg)
stdreal_lambda_r2,stdimag_lambda_r2=np.zeros(ngavg),np.zeros(ngavg)
for i in range(ngavg):
    dataflat= np.reshape(np.squeeze(eigvoutlier_series[i,:nrank,:]),(2*nbatch,1))
    mean_lambda_r2[i]=np.mean(np.squeeze(eigvoutlier_series[i,0,:]+eigvoutlier_series[i,1,:])/2.0)
    print('one mean:',mean_lambda_r2[i],'; second:',np.mean(dataflat))
    # std_lambda_r2[i]=np.std(dataflat)
    
''' plot the two outliers '''    
fig,ax=plt.subplots(figsize=(9,9))
for i in range(20):
    ax.scatter(gaverageseries[i]*np.ones(50),eigvoutlier_series[i,0,:],c='r',alpha=0.05)
    ax.scatter(gaverageseries[i]*np.ones(50),eigvoutlier_series[i,1,:],c='b',alpha=0.05)
ax.set_aspect('equal')
ax.plot(gaverageseries,eigvorg[0]*np.ones(ngavg),'r')
ax.plot(gaverageseries,eigvorg[1]*np.ones(ngavg),'b')

# calculate \theta overlap
avg_overlapEI_eigvec_m,std_overlapEI_eigvec_m=np.zeros((ngavg,2,nrank)),np.zeros((ngavg,2,nrank))
for j in range(nrank):
    for i in range(ngavg):
        uaE=np.reshape(eigvecorg[:NE,j],(NE,1))
        uaI=np.reshape(eigvecorg[NE:,j],(NE,1))
        eigvecE_current=np.squeeze(eigvecxt_series[:NE,j,:,i])
        eigvecI_current=np.squeeze(eigvecxt_series[NE:,j,:,i])
        # EXC
        overlap_eigvecE_m = np.abs(uaE.T@eigvecE_current)
        avg_overlapEI_eigvec_m[i,0,j]=np.mean(overlap_eigvecE_m)
        std_overlapEI_eigvec_m[i,0,j]=np.std(overlap_eigvecE_m)
        #INH
        overlap_eigvecI_m = np.abs(uaI.T@eigvecI_current)
        avg_overlapEI_eigvec_m[i,1,j]=np.mean(overlap_eigvecI_m)
        std_overlapEI_eigvec_m[i,1,j]=np.std(overlap_eigvecI_m)
        
fig,ax=plt.subplots(figsize=(9,9))
ax.plot(gaverageseries,avg_overlapEI_eigvec_m[:,0,1],'r.',linewidth=1.5)
ax.plot(gaverageseries,avg_overlapEI_eigvec_m[:,1,1],'b.',linewidth=1.5)

ax.plot(gaverageseries,avg_overlapEI_eigvec_m[:,0,1]+std_overlapEI_eigvec_m[:,0,1],'m.',linewidth=1.5)
ax.plot(gaverageseries,avg_overlapEI_eigvec_m[:,1,1]++std_overlapEI_eigvec_m[:,1,1],'c.',linewidth=1.5)
ax.set_aspect('equal')
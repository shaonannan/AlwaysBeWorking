# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import numpy as np
import matplotlib.pylab as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'qt')
from numpy import linalg as la
from scipy.optimize import fsolve
from scipy import linalg as scpla
import seaborn as sb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve,leastsq 
from math import tanh,cosh
from sympy import *
extras_require = {'PLOT':['matplotlib>=1.1.1,<3.0']},


# %%
def FPfunc(x,*data):
    JE,JI,a,b,tfunc=data
    if tfunc=='tanh':
        x0 = float(x[0])
        x1 = float(x[1])
        resp0=np.tanh(x0+x1)+1
        resp1=np.tanh(x0-x1)+1
        return [
            x0-JE*resp0+JI*resp1,
            x1+a*resp0-b*resp1
        ]  
    elif tfunc=='tanhs':
        x0 = float(x[0])
        x1 = float(x[1])
        resp0=2.0*(np.tanh(x0+x1)+1)
        resp1=2.0*(np.tanh(x0-x1)+1)
        return [
            x0-JE*resp0+JI*resp1,
            x1+a*resp0-b*resp1
        ]        
        # return [
        #     (np.tanh(x0-x1)+1)-(JE*x1+a*x0)/(b*JE-a*JI),
        #     (np.tanh(x0+x1)+1)-(JI*x1+b*x0)/(b*JE-a*JI)
        # ]
def FPfuncSVD(x,*data):
    JE,JI,a,b,tfunc=data
    if tfunc=='tanh':
        x0 = float(x[0])
        x1 = float(x[1])
        # calculate SVD
        Jt = np.zeros((2,2))
        Jt[:,0],Jt[:,1]=JE,-JI
        Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-a,Jt[0,1]+b,Jt[1,0]+a,Jt[1,1]-b
        lvec,sv,rvech=la.svd(Jt)
        m,n=lvec,rvech.T
        m[:,0]*=sv[0]
        m[:,1]*=sv[1]
        if n[0,0]>0:
            n[:,0]*=(-1)   
            m[:,0]*=(-1)
        if n[0,1]<0:
            n[:,1]*=(-1)   
            m[:,1]*=(-1)
        resp0=np.tanh(x0*m[0,0]+x1*m[0,1])+1
        resp1=np.tanh(x0*m[1,0]+x1*m[1,1])+1
        return [
            x0-n[0,0]*resp0-n[1,0]*resp1,
            x1-n[0,1]*resp0-n[1,1]*resp1
        ]
    elif tfunc=='tanhs':
        x0 = float(x[0])
        x1 = float(x[1])
        # calculate SVD
        Jt = np.zeros((2,2))
        Jt[:,0],Jt[:,1]=JE,-JI
        Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-a,Jt[0,1]+b,Jt[1,0]+a,Jt[1,1]-b
        lvec,sv,rvech=la.svd(Jt)
        m,n=lvec,rvech.T
        m[:,0]*=sv[0]
        m[:,1]*=sv[1]
        if sv[1]<0:
            n[:,1]*=(-1)   
            m[:,1]*=(-1)
        resp0=2.0*(np.tanh(x0*m[0,0]+x1*m[0,1])+1)
        resp1=2.0*(np.tanh(x0*m[1,0]+x1*m[1,1])+1)
        return [
            x0-n[0,0]*resp0-n[1,0]*resp1,
            x1-n[0,1]*resp0-n[1,1]*resp1
        ]

JE,JI,a,b=1.6,0.8,0.12,0.10
# 二元一次方程
c=b*JE-a*JI
aadd,asub=(a+JE)/(2*c),(a-JE)/(2*c)
badd,bsub=(b+JI)/(2*c),(b-JI)/(2*c)
# # -------------
# # A------method
# # -------------
# x0 = Symbol('kp')
# x1 = Symbol('km')
# solved_value=solve([((1-x1**2)*x1+1)-(aadd*x0+asub*x1),((1-x0**2)*x0+1)-(badd*x0+bsub*x1)], [x0, x1])
# print(solved_value)
# # -------------
# # B------method
# # -------------
# figure = plt.figure()
# ax = Axes3D(figure)
# k1 = np.arange(-10,10,0.5)
# k2 = np.arange(-10,10,0.5)
# X,Y = np.meshgrid(k1,k2)
# R0 = (np.tanh(X-Y)+1)-(JE*Y+a*X)/(b*JE-a*JI)
# R1 = (np.tanh(X+Y)+1)-(JI*Y+b*X)/(b*JE-a*JI)
# Z0=0*X
# ax.plot_surface(X,Y,R0,rstride=1,cstride=1,cmap='rainbow')
# ax.plot_surface(X,Y,R1,rstride=1,cstride=1,cmap='rainbow')
# ax.plot_surface(X,Y,Z0,rstride=1,cstride=1,cmap='gray')
# plt.show()
# ------------
# C-----method
# ------------

null.tpl [markdown]
# We change the parameters $J_E,\ b$, then observe the dynamics under different sets of vectors.

# %%
# bseries= np.logspace(-2,  2.0,num =20,base=2)
JE,JI,a,b=1.6,0.8,0.10,0.10
bseries = np.linspace(-15.0,  15.0,num =500)  
jeseries=np.linspace(0.6,1.6,num=10)
jeseries=np.linspace(1.2,1.3,num=1)
nlen,nje=len(bseries),len(jeseries)
M=np.array([[1,1],[1,-1]])
xFPseries = np.zeros((nje,nlen,2,2))
svdvalues = np.zeros((nje,nlen,2))
svdvec = np.zeros((nje,nlen,2,4))
kappaMN,kappamnSVD=np.zeros((nje,nlen,2)),np.zeros((nje,nlen,2))
for idxje,JE in enumerate(jeseries):
    for idxb,bv in enumerate(bseries):
        bv*=a
        # calculate SVD
        Jt = np.zeros((2,2))
        Jt[:,0],Jt[:,1]=JE,-JI
        Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-a,Jt[0,1]+bv,Jt[1,0]+a,Jt[1,1]-bv
        if (np.min(Jt[:,0])<0.0) or (np.max(Jt[:,1])>0.0):
            xFPseries[idxje,idxb,:,:]=np.nan
            kappaMN[idxje,idxb,:]=np.nan
            kappamnSVD[idxje,idxb,:]=np.nan
            continue   
        lvec,sv,rvech=la.svd(Jt)
        svdvalues[idxje,idxb,:]=sv
        m,n=lvec,rvech.T
              
        m[:,0]*=sv[0]
        m[:,1]*=sv[1]
        if n[0,0]>0:
            n[:,0]*=(-1)   
            m[:,0]*=(-1)
        if n[0,1]<0:
            n[:,1]*=(-1)   
            m[:,1]*=(-1)
        svdvec[idxje,idxb,:,:2]=m
        svdvec[idxje,idxb,:,2:]=n
        # check eig
        eigv,eigvec=la.eig(Jt)
        if eigv[0]>1.0:
            xFPseries[idxje,idxb,:,:]=np.nan
            kappaMN[idxje,idxb,:]=np.nan
            kappamnSVD[idxje,idxb,:]=np.nan
            continue
        N=np.array([[JE,-a],[-JI,bv]])
        data=(JE,JI,a,bv,'tanh')
        x0=[0.10,0.10]
        results = fsolve(FPfunc,x0,data)
        kappaMN[idxje,idxb,:]=results
        xFP= M@np.reshape(results,(2,1))
        xFPseries[idxje,idxb,0,:]=xFP[:,0]
        
        resultSVD= fsolve(FPfuncSVD,x0,data)
        kappamnSVD[idxje,idxb,:]=resultSVD
        xFPSVD= m@np.reshape(resultSVD,(2,1))
        xFPseries[idxje,idxb,1,:]=xFPSVD[:,0]
        # print('Fixed points A:',xFP,' SVD(B):',xFPSVD)
        # overlap under MNT
        diagdphi=np.zeros((2,2))
        for i in range(2):
            diagdphi[i,i]=1/np.cosh(xFPSVD[i])**2
        stabilityMN=N.T@diagdphi@M#
        stabilitymn=n.T@diagdphi@m#
        eigvMN,eigvecMN=la.eig(stabilityMN)
        eigvmn,eigvecmn=la.eig(stabilitymn)
        # print('eigenvalues of overlap MN and mn:',eigvMN,' and ',eigvmn)


# %%
fig = plt.figure()
ax0 = fig.add_subplot(121,projection='3d')
ax1 = fig.add_subplot(122,projection='3d')
for idxje in range(nje):
    ax0.plot3D(bseries,kappaMN[idxje,:,0],kappaMN[idxje,:,1])#,'r',label=r'$\kappa_{M}$ basis')
    ax0.plot3D(bseries,kappamnSVD[idxje,:,0],kappamnSVD[idxje,:,1])#,'b',label=r'$\kappa_{m}$ basis')
    ax1.plot3D(bseries,xFPseries[idxje,:,0,0],xFPseries[idxje,:,0,1])#,'r',label=r'Fixed point under vectors $M$ basis')
    ax1.plot3D(bseries,np.tanh(xFPseries[idxje,:,1,0])+1,np.tanh(xFPseries[idxje,:,1,1])+1,'--')#,'b',label=r'Fixed point under vectors $m$ basis')

# idxje+=1
# ax0.plot3D(bseries,kappaMN[idxje,:,0],kappaMN[idxje,:,1])#,'r',label=r'$\kappa_{M}$ basis')
# ax0.plot3D(bseries,kappamnSVD[idxje,:,0],kappamnSVD[idxje,:,1])#,'b',label=r'$\kappa_{m}$ basis')
ax0.set_xlabel(r'ratio of $b/a$',fontsize=14)
ax0.set_ylabel(r'$\kappa_1^{M/m}$',fontsize=14)
ax0.set_zlabel(r'$\kappa_2^{M/m}$',fontsize=14)
ax0.set_title(r'Dynamics of $\mathbf{\kappa}^M$',fontsize=16)
plt.legend()
ax1.plot3D(bseries,xFPseries[idxje,:,0,0],xFPseries[idxje,:,0,1])#,'r',label=r'Fixed point under vectors $M$ basis')
ax1.plot3D(bseries,xFPseries[idxje,:,1,0],xFPseries[idxje,:,1,1])#,'b',label=r'Fixed point under vectors $m$ basis')
ax1.set_xlabel(r'ratio of $b/a$',fontsize=14)
ax1.set_ylabel(r'$x_{1(E)}$',fontsize=14)
ax1.set_zlabel(r'$x_{2(I)}$',fontsize=14)
ax1.set_title(r'Dynamics of neuronal responses',fontsize=16)
plt.legend()
plt.show()


# %%
print('kappa(m) 2')
abruptpoint = np.zeros((nje,3*2))
abruptpointsv=np.zeros((nje,2,2))
abruptpointsvec=np.zeros((nje,2,2,4))
abruptResp=np.zeros((nje,2,4))
for idxje in range(nje):
    diffkappa2=np.diff(kappamnSVD[idxje,:,1])
    diffkappa2/=bseries[2]-bseries[1]
    idxabrupt=np.where(diffkappa2<-10)
    if len(idxabrupt[0]>0):
        abruptpoint[idxje,0]=bseries[idxabrupt[0][0]]
        # print(idxabrupt[0])
        abruptpoint[idxje,1:3]=kappamnSVD[idxje,idxabrupt[0][0]:(idxabrupt[0][0]+2),1]
        abruptResp[idxje,:,:2]=np.tanh(xFPseries[idxje,idxabrupt[0][0]:idxabrupt[0][0]+2,1,:])+1
        abruptpointsv[idxje,0,:]=svdvalues[idxje,idxabrupt[0][0],:]
        abruptpointsvec[idxje,0,:,:]=svdvec[idxje,idxabrupt[0][0],:,:]
    idxabrupt=np.where(diffkappa2>10)
    if len(idxabrupt[0])>0:
        abruptpoint[idxje,3]=bseries[idxabrupt[0][0]]
        # print(idxabrupt[0])
        abruptpoint[idxje,4:]=kappamnSVD[idxje,idxabrupt[0][0]:(idxabrupt[0][0]+2),1]
        abruptResp[idxje,:,2:]=np.tanh(xFPseries[idxje,idxabrupt[0][0]:idxabrupt[0][0]+2,1,:])+1
        abruptpointsv[idxje,1,:]=svdvalues[idxje,idxabrupt[0][0],:]
        abruptpointsvec[idxje,1,:,:]=svdvec[idxje,idxabrupt[0][0],:,:]
# print(abruptpoint[:,0])
# print(abruptpoint[:,1])
# print(abruptpoint[:,2])
# print(abruptpointsv[:,0,:])
# print(abruptpoint[:,3])
# print(abruptpoint[:,4])
# print(abruptpoint[:,5])
# print(abruptpointsv[:,1,:])

# print((jeseries/JI)**(-1))


# %%
print('b:',abruptpoint[0,0],' and ',abruptpoint[0,3])
print('neuronal responses:',abruptResp[0,:,:2],' or ',abruptResp[0,:,2:])
print('singular values:',abruptpointsv[0,0,:],' or ',abruptpointsv[0,1,:])

plt.subplot(221)
plt.plot([0,abruptpointsvec[0,0,0,0]],[0,abruptpointsvec[0,0,1,0]],'r')
plt.plot([0,abruptpointsvec[0,0,0,1]],[0,abruptpointsvec[0,0,1,1]],'b')
plt.axis('square')
plt.subplot(222)
plt.plot([0,abruptpointsvec[0,0,0,2]],[0,abruptpointsvec[0,0,1,2]],'r')
plt.plot([0,abruptpointsvec[0,0,0,3]],[0,abruptpointsvec[0,0,1,3]],'b')
plt.axis('square')
plt.subplot(223)
plt.plot([0,abruptpointsvec[0,1,0,0]],[0,abruptpointsvec[0,1,1,0]],'r')
plt.plot([0,abruptpointsvec[0,1,0,1]],[0,abruptpointsvec[0,1,1,1]],'b')
plt.axis('square')
plt.subplot(224)
plt.plot([0,abruptpointsvec[0,1,0,2]],[0,abruptpointsvec[0,1,1,2]],'r')
plt.plot([0,abruptpointsvec[0,1,0,3]],[0,abruptpointsvec[0,1,1,3]],'b')
plt.axis('square')

null.tpl [markdown]
# We change the parameters $J_E,\ a$, then observe the dynamics under different sets of vectors.

# %%
# bseries= np.logspace(-2,  2.0,num =20,base=2)
JE,JI,a,b=1.6,0.8,0.10,0.10
aseries = np.linspace(-15.0,  15.0,num =500)  
jeseries=np.linspace(0.6,1.6,num=10)
nlen,nje=len(aseries),len(jeseries)
M=np.array([[1,1],[1,-1]])
xFPseries_ = np.zeros((nje,nlen,2,2))
kappaMN_,kappamnSVD_=np.zeros((nje,nlen,2)),np.zeros((nje,nlen,2))
for idxje,JE in enumerate(jeseries):
    for idxa,av in enumerate(aseries):
        av*=b
        # calculate SVD
        Jt = np.zeros((2,2))
        Jt[:,0],Jt[:,1]=JE,-JI
        Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-av,Jt[0,1]+b,Jt[1,0]+av,Jt[1,1]-b
        lvec,sv,rvech=la.svd(Jt)
        m,n=lvec,rvech.T
        m[:,0]*=sv[0]
        m[:,1]*=sv[1]
        if n[0,0]>0:
            n[:,0]*=(-1)   
            m[:,0]*=(-1)
        if n[0,1]<0:
            n[:,1]*=(-1)   
            m[:,1]*=(-1)
        # check eig
        eigv,eigvec=la.eig(Jt)
        if eigv[0]>1.0:
            xFPseries[idxje,idxb,:,:]=np.nan
            kappaMN[idxje,idxb,:]=np.nan
            kappamnSVD[idxje,idxb,:]=np.nan
            continue
        N=np.array([[JE,-av],[-JI,b]])
        data=(JE,JI,av,b,'tanh')
        x0=[1.0,1.0]
        results = fsolve(FPfunc,x0,data)
        kappaMN_[idxje,idxa,:]=results
        xFP= M@np.reshape(results,(2,1))
        xFPseries_[idxje,idxa,0,:]=xFP[:,0]
        
        resultSVD= fsolve(FPfuncSVD,x0,data)
        kappamnSVD_[idxje,idxa,:]=resultSVD
        xFPSVD= m@np.reshape(resultSVD,(2,1))
        xFPseries_[idxje,idxa,1,:]=xFPSVD[:,0]
        # print('Fixed points A:',xFP,' SVD(B):',xFPSVD)
        # overlap under MNT
        diagdphi=np.zeros((2,2))
        for i in range(2):
            diagdphi[i,i]=1/np.cosh(xFPSVD[i])**2
        stabilityMN=N.T@diagdphi@M#
        stabilitymn=n.T@diagdphi@m#
        eigvMN,eigvecMN=la.eig(stabilityMN)
        eigvmn,eigvecmn=la.eig(stabilitymn)
        # print('eigenvalues of overlap MN and mn:',eigvMN,' and ',eigvmn)


# %%
fig = plt.figure()
ax0 = fig.add_subplot(121,projection='3d')
ax1 = fig.add_subplot(122,projection='3d')
for idxje in range(nje-1):
    ax0.plot3D(aseries,kappaMN_[idxje,:,0],kappaMN_[idxje,:,1])#,'r',label=r'$\kappa_{M}$ basis')
    ax0.plot3D(aseries,kappamnSVD_[idxje,:,0],kappamnSVD_[idxje,:,1],'--')#,'b',label=r'$\kappa_{m}$ basis')
    ax1.plot3D(aseries,xFPseries_[idxje,:,0,0],xFPseries_[idxje,:,0,1])#,'r',label=r'Fixed point under vectors $M$ basis')
    ax1.plot3D(aseries,xFPseries_[idxje,:,1,0],xFPseries_[idxje,:,1,1])#,'b',label=r'Fixed point under vectors $m$ basis')

idxje+=1
ax0.plot3D(aseries,kappaMN_[idxje,:,0],kappaMN_[idxje,:,1])#,'r',label=r'$\kappa_{M}$ basis')
ax0.plot3D(aseries,kappamnSVD_[idxje,:,0],kappamnSVD_[idxje,:,1])#,'b',label=r'$\kappa_{m}$ basis')
ax0.set_xlabel(r'ratio of $a/b$',fontsize=14)
ax0.set_ylabel(r'$\kappa_1^{M/m}$',fontsize=14)
ax0.set_zlabel(r'$\kappa_2^{M/m}$',fontsize=14)
ax0.set_title(r'Dynamics of $\mathbf{\kappa}^M$',fontsize=16)
# plt.legend()
ax1.plot3D(bseries,xFPseries_[idxje,:,0,0],xFPseries_[idxje,:,0,1])#,'r',label=r'Fixed point under vectors $M$ basis')
ax1.plot3D(bseries,xFPseries_[idxje,:,1,0],xFPseries_[idxje,:,1,1])#,'b',label=r'Fixed point under vectors $m$ basis')
ax1.set_xlabel(r'ratio of $a/b$',fontsize=14)
ax1.set_ylabel(r'$x_{1(E)}$',fontsize=14)
ax1.set_zlabel(r'$x_{2(I)}$',fontsize=14)
ax1.set_title(r'Dynamics of neuronal responses',fontsize=16)
# plt.legend()
plt.show()

null.tpl [markdown]
# We change the parameters $J_I,\ b$, then observe the dynamics under different sets of vectors.

# %%
# bseries= np.logspace(-2,  2.0,num =20,base=2)
JE,JI,a,b=0.8,0.8,0.10,0.10
bseries = np.linspace(-50.0,  15.0,num =500)  
jiseries=np.linspace(0.6,1.6,num=10)
nlen,nji=len(bseries),len(jiseries)
M=np.array([[1,1],[1,-1]])
xFPseries = np.zeros((nji,nlen,2,2))
svdvalues = np.zeros((nji,nlen,2))
svdvec = np.zeros((nji,nlen,2,4))
kappaMN,kappamnSVD=np.zeros((nji,nlen,2)),np.zeros((nji,nlen,2))
for idxji,JI in enumerate(jiseries):
    for idxb,bv in enumerate(bseries):
        bv*=a
        # calculate SVD
        Jt = np.zeros((2,2))
        Jt[:,0],Jt[:,1]=JE,-JI
        Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-a,Jt[0,1]+bv,Jt[1,0]+a,Jt[1,1]-bv
        if (np.min(Jt[:,0])<0.0) or (np.max(Jt[:,1])>0.0):
            xFPseries[idxji,idxb,:,:]=np.nan
            kappaMN[idxji,idxb,:]=np.nan
            kappamnSVD[idxji,idxb,:]=np.nan
            continue     
        lvec,sv,rvech=la.svd(Jt)
        svdvalues[idxji,idxb,:]=sv
        m,n=lvec,rvech.T
              
        m[:,0]*=sv[0]
        m[:,1]*=sv[1]
        if n[0,0]>0:
            n[:,0]*=(-1)   
            m[:,0]*=(-1)
        if n[0,1]<0:
            n[:,1]*=(-1)   
            m[:,1]*=(-1)
        svdvec[idxji,idxb,:,:2]=m
        svdvec[idxji,idxb,:,2:]=n
        # check eig
        eigv,eigvec=la.eig(Jt)
        if eigv[0]>1.0:
            xFPseries[idxji,idxb,:,:]=np.nan
            kappaMN[idxji,idxb,:]=np.nan
            kappamnSVD[idxji,idxb,:]=np.nan
            continue
        N=np.array([[JE,-a],[-JI,bv]])
        data=(JE,JI,a,bv,'tanh')
        x0=[0.10,0.10]
        results = fsolve(FPfunc,x0,data)
        kappaMN[idxji,idxb,:]=results
        xFP= M@np.reshape(results,(2,1))
        xFPseries[idxji,idxb,0,:]=xFP[:,0]
        
        resultSVD= fsolve(FPfuncSVD,x0,data)
        kappamnSVD[idxji,idxb,:]=resultSVD
        xFPSVD= m@np.reshape(resultSVD,(2,1))
        xFPseries[idxji,idxb,1,:]=xFPSVD[:,0]
        # print('Fixed points A:',xFP,' SVD(B):',xFPSVD)
        # overlap under MNT
        diagdphi=np.zeros((2,2))
        for i in range(2):
            diagdphi[i,i]=1/np.cosh(xFPSVD[i])**2
        stabilityMN=N.T@diagdphi@M#
        stabilitymn=n.T@diagdphi@m#
        eigvMN,eigvecMN=la.eig(stabilityMN)
        eigvmn,eigvecmn=la.eig(stabilitymn)
        # print('eigenvalues of overlap MN and mn:',eigvMN,' and ',eigvmn)


# %%
fig = plt.figure()
ax0 = fig.add_subplot(121,projection='3d')
ax1 = fig.add_subplot(122,projection='3d')
for idxji in range(nji-1):
    ax0.plot3D(bseries,kappaMN[idxji,:,0],kappaMN[idxji,:,1])#,'r',label=r'$\kappa_{M}$ basis')
    ax0.plot3D(bseries,kappamnSVD[idxji,:,0],kappamnSVD[idxji,:,1])#,'b',label=r'$\kappa_{m}$ basis')
    ax1.plot3D(bseries,xFPseries[idxji,:,0,0],xFPseries[idxji,:,0,1])#,'r',label=r'Fixed point under vectors $M$ basis')
    ax1.plot3D(bseries,np.tanh(xFPseries[idxji,:,1,0])+1,np.tanh(xFPseries[idxji,:,1,1])+1,'--')#,'b',label=r'Fixed point under vectors $m$ basis')

idxji+=1
ax0.plot3D(bseries,kappaMN[idxji,:,0],kappaMN[idxji,:,1])#,'r',label=r'$\kappa_{M}$ basis')
ax0.plot3D(bseries,kappamnSVD[idxji,:,0],kappamnSVD[idxji,:,1])#,'b',label=r'$\kappa_{m}$ basis')
ax0.set_xlabel(r'ratio of $b/a$',fontsize=14)
ax0.set_ylabel(r'$\kappa_1^{M/m}$',fontsize=14)
ax0.set_zlabel(r'$\kappa_2^{M/m}$',fontsize=14)
ax0.set_title(r'Dynamics of $\mathbf{\kappa}^M$',fontsize=16)
# plt.legend()
ax1.plot3D(bseries,xFPseries[idxji,:,0,0],xFPseries[idxji,:,0,1])#,'r',label=r'Fixed point under vectors $M$ basis')
ax1.plot3D(bseries,xFPseries[idxji,:,1,0],xFPseries[idxji,:,1,1])#,'b',label=r'Fixed point under vectors $m$ basis')
ax1.set_xlabel(r'ratio of $b/a$',fontsize=14)
ax1.set_ylabel(r'$x_{1(E)}$',fontsize=14)
ax1.set_zlabel(r'$x_{2(I)}$',fontsize=14)
ax1.set_title(r'Dynamics of neuronal responses',fontsize=16)
# plt.legend()
plt.show()

null.tpl [markdown]
# 

# %%
import numpy as np
from scipy import integrate

# define system in terms of a Numpy array
def Sys(X, t=0, *data):
    # here X[0] = KAPPA1 and X[1] = KAPPA2
    # PARAMETERS
    JE,JI,a,b,functype=data[0],data[1],data[2],data[3],data[4]   
    if functype=='MN':
        return np.array([ (-X[0] + JE*(np.tanh(X[0]+X[1])+1)-JI*(np.tanh(X[0]-X[1])+1)) , (-X[1] -a*(np.tanh(X[0]+X[1])+1)+b*(np.tanh(X[0]-X[1])+1))])

# generate 1000 linearly spaced numbers for x-axes
t = np.linspace(0, 10,  1000)
JE,JI,a,b=1.2,0.8,0.1,0.15
# initial values: x0 = 10, y0 = 2
Sys0 = np.array([1.0, 1.0])
bseries = np.linspace(-15.0,  15.0,num =100) 
fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2) 
effectbseries=[]
for b in bseries: 
    # calculate SVD
    Jt = np.zeros((2,2))
    Jt[:,0],Jt[:,1]=JE,-JI
    Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-a,Jt[0,1]+b,Jt[1,0]+a,Jt[1,1]-b
    if (np.min(Jt[:,0])<0.0) or (np.max(Jt[:,1])>0.0):
        continue     
    # check eig
    eigv,eigvec=la.eig(Jt)
    if eigv[0]>1.0:
        continue
    data=(JE,JI,a,b,'MN')
    effectbseries.append(b)
    # type "help(integrate.odeint)" if you want more information about integrate.odeint inputs and outputs.
    X, infodict = integrate.odeint(Sys, Sys0, t, data,full_output=True)
    # infodict['message']                      # integration successful

    x,y = X.T

    #plot
    ax1.plot(x, 'r-', label=r'$kappa_1^{M}$')
    ax1.plot(y, 'b-', label=r'$kappa_2^{M}$')
    ax2.plot(x, y, color="blue")

ax1.set_title("Dynamics in time")
ax1.set_xlabel("time")
ax1.grid()
# ax1.legend(loc='best')
ax2.set_xlabel("x")
ax2.set_ylabel("y")  
ax2.set_title("Phase space")
ax2.grid()

null.tpl [markdown]
# Draw nullclines and quiver plot, also visualize the direction of the flow, as well as corresponding eigenvectors of overlap matrix $N^{\intercal}M$ and $n^{\intercal}m$

# %%
#plot
fig2 = plt.figure(figsize=(8,6))
ax4 = fig2.add_subplot(1,1,1)

# x = np.linspace(0,2,20)
# y = np.arange(0,2,20)

# # plot nullclines
# ax4.plot([0,2],[2,0], 'r-', lw=2, label='x-nullcline')
# ax4.plot([1,1],[0,2], 'b-', lw=2, label='y-nullcline')
 
# # plot fixed points
# for point in fp:
#     ax4.plot(point[0],point[1],"red", marker = "o", markersize = 10.0)
# ax4.set_title("Quiverplot with nullclines")
# ax4.legend(loc='best')

# quiverplot
# define a grid and compute direction at each point
x = np.linspace(-2.0, 2.0, 20)
y = np.linspace(-2.0, 2.0, 20)
b = 0.15
X1 , Y1  = np.meshgrid(x, y)                    # create a grid
data=(JE,JI,a,b,'MN')                           # parameters to be transferred
DX1, DY1 = Sys([X1, Y1],t,JE,JI,a,b,'MN')       # compute growth rate on the grid
M = (np.hypot(DX1, DY1))                        # norm growth rate 
M[ M == 0] = 1.                                 # avoid zero division errors 
# DX1 /= M                                        # normalize each arrows
# DY1 /= M

ax4.quiver(X1, Y1, DX1, DY1, M, pivot='mid')
# ax4.legend()
ax4.set_title("Quiverplot with nullclines")
ax4.grid()


# %%
#plot
fig2 = plt.figure(figsize=(8,6))
ax4 = fig2.add_subplot(1,1,1,projection='3d')
# quiverplot
# define a grid and compute direction at each point
z = np.linspace(-2.0, 2.0, 20)
y = np.linspace(-2.0, 2.0, 20)
x = effectbseries.copy()
X1 , Y1, Z1  = np.meshgrid(x, y, z)                   # create a grid
DZ1,DY1,DX1=np.zeros_like(Z1),np.zeros_like(Y1),np.zeros_like(X1)
Mn= np.zeros_like(Z1)
Mnorm=np.zeros_like(X1)
for idxb, b in enumerate(effectbseries[0::2]):
    data=(JE,JI,a,b,'MN')                             # parameters to be transferred
    ys,zs=np.squeeze(Y1[:,idxb,:]),np.squeeze(Z1[:,idxb,:])
    dy1, dz1 = Sys([ys, zs],t,JE,JI,a,b,'MN')         # compute growth rate on the grid
    M = np.sqrt((dy1**2+dz1**2))                       # norm growth rate                                          # avoid zero division errors 
    # dx1 /= M                                        # normalize each arrows
    # dy1 /= M
    DY1[:,idxb,:],DZ1[:,idxb,:],Mn[:,idxb,:]=dy1,dz1,M
    ax4.quiver(X1[:,idxb,:], Y1[:,idxb,:],Z1[:,idxb,:], DX1[:,idxb,:], DY1[:,idxb,:],DZ1[:,idxb,:],color='deepskyblue',length=0.1,arrow_length_ratio=0.3,pivot='tail')#,normalize=False)
# ax4.legend()
ax4.set_title("Quiverplot with nullclines")
ax4.grid()

null.tpl [markdown]
# Perturbation analysis, check whether the singular values change with the conditions of basis vectors (coordinate system)

# %%
def derivtransfer(FPs,type='tanh'):
    xe,xi=FPs[0],FPs[1]
    if type=='tanh':
        derive=1/np.cosh(xe)**2
        derivi=1/np.cosh(xi)**2
        return np.array([derive,derivi])

JE,JI,a,b=1.2,0.8,0.1,-0.15151515151505
print('JE,JI,a,b:',JE,JI,a,b)
# calculate SVD
Jt = np.zeros((2,2))
Jt[:,0],Jt[:,1]=JE,-JI
Jt[0,0],Jt[0,1],Jt[1,0],Jt[1,1]=Jt[0,0]-a,Jt[0,1]+b,Jt[1,0]+a,Jt[1,1]-b
if (np.min(Jt[:,0])<0.0) or (np.max(Jt[:,1])>0.0):
    print('un-biological!')   
lvec,sv,rvech=la.svd(Jt)
m,n=lvec,rvech.T        
m[:,0]*=sv[0]
m[:,1]*=sv[1]
if n[0,0]>0:
    n[:,0]*=(-1)   
    m[:,0]*=(-1)
if n[0,1]<0:
    n[:,1]*=(-1)   
    m[:,1]*=(-1)
# check eig
eigv,eigvec=la.eig(Jt)
if eigv[0]>1.0:
    print('unstable!')
M=np.array([[1,1],[1,-1]])
N=np.array([[JE,-a],[-JI,b]])
data=(JE,JI,a,b,'tanh')
x0=[0.10,0.10]
results = fsolve(FPfunc,x0,data)
print(results)
xFP= M@np.reshape(results,(2,1))
resultSVD= fsolve(FPfuncSVD,x0,data)
xFPSVD= m@np.reshape(resultSVD,(2,1))
print('activity x:',xFP,' or ',xFPSVD)
derivefp=derivtransfer(np.squeeze(xFP),'tanh')
derivphi=np.zeros((2,2))
derivphi[0,0],derivphi[1,1]=derivefp[0],derivefp[1]
JM=derivphi@M@N.T
lvec,sv,rvec=la.svd(JM)
JoM=N.T@derivphi@M
lvec_,sv_,rvec_=la.svd(JoM)
print('compare the singular values:',sv,' overlap ',sv_)

Jm=derivphi@m@n.T
lvecm,svm,rvecm=la.svd(Jm)
Jom=n.T@derivphi@m
lvecm_,svm_,rvecm_=la.svd(Jom)
print('compare the singular values:',svm,' overlap ',svm_)

# eigv,eigvec=la.eig(JM)
# eigv_,eigvec_=la.eig(JoM)
# print('compare the eigenvalues:',eigv,' overlap ',eigv_)


# %%
x =np.linspace(-2,2,100)
y = np.sqrt(4-x**2)
y/=np.sum(y)
y/=0.04
plt.plot(x,y)


# %%




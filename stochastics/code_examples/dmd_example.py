#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:44:23 2018

@author: tyler
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

xi = np.linspace(-10,10,100);
t = np.linspace(0,4*np.pi,80);
dt = t[1] - t[0];

xx,tt = np.meshgrid(xi,t)

f1 = 1/np.cosh(xx+3)*(1*np.exp(2.3j*tt))
f2 = (1/np.cosh(xx)*np.tanh(xx))*(2*np.exp(2.8j*tt))
f = f1+f2

#%%

X = f.T

X1 = X[:,:-1]
X2 = X[:,1:]

r = 2;
U,sigma,Vt = np.linalg.svd(X1,full_matrices=False)

U_hat = U[:,:r]
Sigma_hat = np.diag(sigma[:r])
V_hat = Vt[:r].T.conjugate()

S = U_hat.T.conjugate()@X2@V_hat@np.linalg.inv(Sigma_hat)

Lambda,W = np.linalg.eig(S)

#Phi = U_hat@W
Phi = X2@V_hat@np.linalg.inv(Sigma_hat)@W#@np.diag(Lambda)

plt.plot(xi,np.real(Phi[:,0]))
plt.plot(xi,np.real(Phi[:,1]))

omega = np.log(Lambda)/dt

b = np.linalg.lstsq(Phi,X[:,0])[0]

u_modes = np.zeros((r,len(t)))
for j,tt in enumerate(t):
    u_modes[:,j] = np.exp(omega*tt)*b

plt.figure()
plt.plot(t,np.real(u_modes[0]))
plt.plot(t,np.real(u_modes[1]))

#%%
X_dmd = Phi@u_modes
plt.pcolor(np.real(X_dmd))

plt.figure()
plt.pcolor(np.real(f.T))

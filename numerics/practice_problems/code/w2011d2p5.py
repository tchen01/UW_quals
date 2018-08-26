#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:56:24 2018

@author: tyler
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

#%%
for k in [5,10,60]:
    def f(x,y):
        return (k**2-5*np.pi**2)*np.sin(np.pi*x)*np.sin(2*np.pi*y)
    
    def u_true(x,y):
        return np.sin(np.pi*x)*np.sin(2*np.pi*y)
    
    mesh_sizes = np.array([3,9,30,99,299]);
    max_error = np.zeros(len(mesh_sizes));
    
    for j,m in enumerate(mesh_sizes): # number of interior mesh points in a given direction
        h = 1/(m+1)
        
        
        # construct A
        T = sp.sparse.diags([-4*np.ones(m),np.ones(m-1),np.ones(m-1)],[0,1,-1])
    
        A = sp.sparse.kron(sp.sparse.eye(m),T) + sp.sparse.kron(sp.sparse.diags([np.ones(m-1),np.ones(m-1)],[-1,1]),sp.sparse.eye(m))
        
        A /= h**2
        A += k**2 * sp.sparse.eye(m*m)
            
        # construct right hand side F
        xy = np.linspace(0,1,m+2)[1:-1] # get position of interior points
        
        F = np.reshape([[f(x,y) for x in xy] for y in xy],-1)
        
        # solve system
        U = sp.sparse.linalg.spsolve(A,F)
        
        U_true = np.reshape([[u_true(x,y) for x in xy] for y in xy],-1)
        
        max_error[j] = np.max(np.abs(U-U_true))


#    plt.figure()
#    plt.scatter(np.log10(1/(mesh_sizes+1)),np.log10(max_error),color='k')
#    plt.savefig('w2011d2p5_'+str(k)+'.pdf')

#%%
u = np.zeros((m+2,m+2))
u[1:-1,1:-1] = np.reshape(U,(m,m))

u_ = np.zeros((m+2,m+2))
u_[1:-1,1:-1] = np.reshape(U_true,(m,m))

plt.figure()
plt.pcolor(u_)
plt.colorbar()
plt.axis('image')

plt.figure()
plt.pcolor(u)
plt.colorbar()
plt.axis('image')


#%%

def jacobi(A,b,x,max_iter):
    M = sp.sparse.diags(A.diagonal())
    for n in range(max_iter):
        r = b-A@x
        x += sp.sparse.linalg.spsolve(M,r)
    return x

#%%
    
for k in [5,10,60.5]:
    def f(x,y):
        return (k**2-5*np.pi**2)*np.sin(np.pi*x)*np.sin(2*np.pi*y)
    
    def u_true(x,y):
        return np.sin(np.pi*x)*np.sin(2*np.pi*y)
    
    mesh_sizes = np.array([20]);
    max_error = np.zeros(len(mesh_sizes));
    
    m = 20;
    h = 1/(m+1)
        
        
    # construct A
    T = sp.sparse.diags([-4*np.ones(m),np.ones(m-1),np.ones(m-1)],[0,1,-1])

    A = sp.sparse.kron(sp.sparse.eye(m),T) + sp.sparse.kron(sp.sparse.diags([np.ones(m-1),np.ones(m-1)],[-1,1]),sp.sparse.eye(m))
    
    A /= h**2
    A += k**2 * sp.sparse.eye(m*m)
        
    # construct right hand side F
    xy = np.linspace(0,1,m+2)[1:-1] # get position of interior points
    
    F = np.reshape([[f(x,y) for x in xy] for y in xy],-1)
    
    # solve system
    U = jacobi(A,F,np.random.rand(m*m),200)
    
    U_true = np.reshape([[u_true(x,y) for x in xy] for y in xy],-1)
    
    max_error = np.max(np.abs(U-U_true))

u = np.zeros((m+2,m+2))
u[1:-1,1:-1] = np.reshape(U,(m,m))

u_ = np.zeros((m+2,m+2))
u_[1:-1,1:-1] = np.reshape(U_true,(m,m))

plt.figure()
plt.pcolor(u_)
plt.colorbar()
plt.axis('image')

plt.figure()
plt.pcolor(u)
plt.colorbar()
plt.axis('image')
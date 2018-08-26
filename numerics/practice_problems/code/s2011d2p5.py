
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

#%%

k = 75
ms = np.logspace(1,3,20,dtype='int')
Ah = np.zeros(len(ms))
Ahinv = np.zeros(len(ms))
error = np.zeros(len(ms))
spect = np.zeros(len(ms))

#%%
for j,m in enumerate(ms):
    h = 1/m
    
    A = sp.sparse.diags([(k**2-2/h**2)*np.ones(m),np.ones(m-1)/h**2,np.ones(m-1)/h**2],[0,1,-1])
    
    A = sp.sparse.csc_matrix(A,dtype='complex');
    
    A[-1,-1] = 1j*k/h+k**2/2-1/h**2
        
    F = np.zeros(m)
    F[0] = -1/h**2
    
    U = np.zeros(m+1,dtype='complex')
    U[0] = 1
    U[1:] = sp.sparse.linalg.spsolve(A,F)
    
    
    X = np.linspace(0,1,m+1)
    U_true = np.cos(k*X) + 1j*np.sin(k*X)
    
    Ah[j] = sp.sparse.linalg.norm(A,np.inf)
    Ahinv[j] = sp.sparse.linalg.norm(sp.sparse.linalg.inv(A),np.inf)
    
    M_inv = sp.sparse.diags(1/A.diagonal())
    
    G = sp.sparse.eye(m) - M_inv@A
    
    spect[j] = np.abs(sp.sparse.linalg.eigs(G,k=1,which='LM',return_eigenvectors=False,))
    
    error[j] = np.linalg.norm(U - U_true,np.inf)

    plt.figure()
    plt.plot(X,np.real(U))
    plt.plot(X,np.imag(U))

#%%

plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.scatter(1/ms,error)
plt.savefig('error.pdf')

plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.scatter(1/ms,Ah)
plt.savefig('Ah.pdf')

plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.scatter(1/ms,Ahinv)
plt.savefig('Ahinv.pdf')

plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.scatter(1/ms,spect)
plt.savefig('spect.pdf')

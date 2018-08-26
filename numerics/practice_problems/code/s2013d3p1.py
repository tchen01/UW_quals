import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

#%%

k = 100
m = 2*k
n = 3*k


T = sp.sparse.diags([-26*np.ones(m),4*np.ones(m-1),4*np.ones(m-1)],[0,1,-1])

T1 = sp.sparse.diags([9*np.ones(m),3*np.ones(m-1),-3*np.ones(m-1)],[0,1,-1])

T2 = T1.T

A = sp.sparse.kron(sp.sparse.eye(n,k=0),T)
A += sp.sparse.kron(sp.sparse.eye(n,k=1),T1)
A += sp.sparse.kron(sp.sparse.eye(n,k=-1),T2)

F = np.zeros(m*n)

for j in range(n):
    for i in range(m):
        k = j*m+i
        if i == 0: #left boundary
            F[k] -= 4*2
        if i == m-1: #right boundary
            F[k] -= 4*1
        if j == 0: #bottom boundary
            F[k] -= 9*0
        if j == n-1: #top boundary
            F[k] -= 9*3
            
U = sp.sparse.linalg.spsolve(A,F)

u = np.reshape(U,(n,m))
plt.pcolor(u)
plt.axis('image')
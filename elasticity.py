import numpy as np
import sys
import re

# Calculate homogeneous deformation, using 3 vectors
def deformation(T1,T2,deftype="SDA"):
    # T1 : ref ; T2 : deformed ; deftype=EA(Euler Almansi)-GL(Green-Lagrange)-SDA(Small deformation assumption)
    # Calcul distorsion
    k=0
    U=np.zeros((9))
    for i in range(0,3):
        for j in range(0,3):
            U[k]=T2[j][i]-T1[j][i]
            k=k+1
    F=np.zeros((3,3))
    F[0]=(np.linalg.tensorsolve(T1,U[0:3]))
    F[1]=(np.linalg.tensorsolve(T1,U[3:6]))
    F[2]=(np.linalg.tensorsolve(T1,U[6:9]))
    if deftype=="SDA" :
        eps=0.5*(F+np.transpose(F))
    if deftype=="GL":
        eps=0.5*( np.matmul((np.transpose(F)),F) + F + np.transpose(F) )
    if deftype=="EA":
        eps=0.5*(np.identity(3)-np.matmul(np.linalg.inv(np.identity(3)+np.transpose(F)),np.linalg.inv((np.identity(3)+F))))
    for i in range(0,3):
        for j in range(0,3):
            if abs(eps[i][j]) < 1E-10 : 
                eps[i][j] = 0
    return(eps)    

# Kronecker delta
def kro(i,j):
    if i==j :
        return 1
    else:
        return 0

# Convert xx yy zz xy xz yz to [i][j]
def conv(A):
	B=np.zeros((3,3))
	B[0][0]=A[0]
	B[1][1]=A[1]
	B[2][2]=A[2]
	B[0][1]=A[3] ; B[1][0]=B[0][1]
	B[1][2]=A[4] ; B[2][1]=B[1][2]
	B[0][2]=A[5] ; B[2][0]=B[0][2]
	return B

## Cijkl & Sijkl
def Hook(Cfile):
    C_v=np.loadtxt(Cfile)
    C_v.astype(np.float)
    C=np.zeros((3,3,3,3))
    # unVoigt
    for i in range(0,3):
        for j in range(0,3):
            if i == j :
                m=i
            else:
                m=6-i-j
            for k in range(0,3):
                for l in range(0,3):
                    if k == l :
                        n=l
                    else:
                        n=6-k-l
                    C[i][j][k][l]=C_v[m][n]
    # Compliance
    S_v=np.linalg.inv(C_v)
    # unVoigt too
    S=np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i == j :
                m=i
            else:
                m=6-i-j
            for k in range(0,3):
                for l in range(0,3):
                    if k == l :
                        n=l
                    else:
                        n=6-k-l
                    if m < 3 and n < 3 :
                        S[i][j][k][l]=S_v[m][n]
                    else : 
                        if m > 2 and n > 2 :
                            S[i][j][k][l]=0.25*S_v[m][n]
                        else :
                            if m > 2 and n < 3  :
                                S[i][j][k][l]=0.5*S_v[m][n]
                            elif m < 3 and n > 2  :
                                S[i][j][k][l]=0.5*S_v[m][n]
    return(C_v,S_v,C,S)
    

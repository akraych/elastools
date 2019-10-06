import numpy as np
import sys
import re

# Fast calculation of elastic constants in dislocation referential
def cdislo(C11,C12,C44):
    newC=np.zeros((7))
    newC[0]=0.5*(C11+C12)+C44
    newC[1]=1/6.*(C11+5*C12-2*C44)
    newC[2]=1/3.*(C11+2*C12-2*C44)
    newC[3]=1/3.*(C11+2*C12+4*C44)
    newC[4]=1/3.*(C11-C12+C44)
    newC[5]=1/6.*(C11-C12+4*C44)
    newC[6]=np.sqrt(2.)/6*(C11-C12-2*C44)
    return "{0:.1f} {1:.1f} {2:.1f} {3:.1f} {4:.1f} {5:.1f} {6:.1f}".format(*newC) 

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
    if deftype=="F":
        return(F)
    else :
        for i in range(0,3):
            for j in range(0,3):
                if abs(eps[i][j]) < 1E-10 : 
                    eps[i][j] = 0
        return(eps)    

def applyshear(eps,vec):
    new_vec=vec+np.einsum('ij,j',eps,vec)
    return(new_vec)

# Kronecker delta
def kro(i,j):
    if i==j :
        return 1
    else:
        return 0

# voigt tools

def index_voigt(m):
    if m==0 :
        return [0,0]
    if m==1 :
        return [1,1]
    if m==2 :
        return [2,2]
    if m==3 :
        return [1,2]
    if m==4 :
        return [0,2]
    if m==5 :
        return [0,1]


def index_unvoigt(i,j):
    if i == 0:
        if j == 0:
            return 0
        if j == 1:
            return 5
        if j == 2:
            return 4
    if i == 1:
        if j == 0:
            return 5
        if j == 1:
            return 1
        if j == 2:
            return 3
    if i == 2:
        if j == 0:
            return 4
        if j == 1:
            return 3
        if j == 2:
            return 2

# Convert xx yy zz xy xz yz to [i][j]
def novoigtshape(A):
    B=np.zeros((3,3))
    B[0][0]=A[0]
    B[1][1]=A[1]
    B[2][2]=A[2]
    B[0][1]=A[3] ; B[1][0]=B[0][1]
    B[1][2]=A[4] ; B[2][1]=B[1][2]
    B[0][2]=A[5] ; B[2][0]=B[0][2]
    return B

def voigtshape(A):
    B=np.zeros(6)
    for m in range(6):
        i,j=index_voigt(m)
        B[m]=A[i][j]
    return B

## Cijkl & Sijkl
class elastic(object):

    Cv=np.zeros((6,6))
    Sv=np.zeros((6,6))
    C=np.zeros((3,3,3,3))
    S=np.zeros((3,3,3,3))
    infos="whatisit"

    def __init__(self,C=C,S=S,Cv=Cv,Sv=Sv):
        self.C=C
        self.S=S
        self.Cv=Cv
        self.Sv=Sv
        self.infos="infos"

    def load(self,Cfile) :
        with open(Cfile,"r") as myfile :
            line1=myfile.readline()
            if line1[0]=="#" :
                self.infos=line1
                print("elastic constants loaded : ",line1)
        self.Cv=np.loadtxt(Cfile)
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
                        self.C[i][j][k][l]=self.Cv[m][n]
        # Compliance
        self.Sv=np.linalg.inv(self.Cv)
        # unVoigt too
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
                            self.S[i][j][k][l]=self.Sv[m][n]
                        else : 
                            if m > 2 and n > 2 :
                                self.S[i][j][k][l]=0.25*self.Sv[m][n]
                            else :
                                if m > 2 and n < 3  :
                                    self.S[i][j][k][l]=0.5*self.Sv[m][n]
                                elif m < 3 and n > 2  :
                                    self.S[i][j][k][l]=0.5*self.Sv[m][n]


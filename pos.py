# Calculate the trajectory of the dislocation

import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
# my tools
from elasticity import *
from vasp_tools import *

def calcA(stress,eps):
    Ax=Surf/h/C_v[3][3] *(stress[0][2] - ((C_v[4][0]*eps[0][0]+C_v[4][1]*eps[1][1]) +2*C_v[3][3]*eps[2][0]) )
    Ay=Surf/h/C_v[3][3] *(stress[1][2] - (2*C_v[3][3]*eps[1][2]+2*C_v[3][5]*eps[0][1]))
    A=[Ax,Ay,0]
    return A

#def calcXI(M):
#    XI=np.einsum('ijkl,kl',S,M)
#    return XI

######### C'est parti
### INPUT

# Cij
cij_file="/home/akraych/scripts/Cij/Cij_k32_W_Rc800"
C_v=Hook(cij_file)[0]
C=Hook(cij_file)[2]
S_v=Hook(cij_file)[1]
S=Hook(cij_file)[3]
## atoms
# cella
contcar_ref="/home/akraych/workdir/2_Wbcc/PP_neb/Moments_dipolaires/calculs/Maxparam/ref/CONTCAR"
# dislo cells
contcar_dis=sorted(glob.glob("/home/akraych/workdir/2_Wbcc/PP_neb/Moments_dipolaires/calculs/Maxparam/1_calc/NEB/ulPP/CONTCAR*"))
#for i,name in enumerate(contcar_dis):
#    if i < 9 :
#        print(name)
# -> Read def

# def_type="EA" ; print("Euler selected") # Euler 
# def_type="GL" ; print("Green Lagrange selected") # Green Lagrange
def_type="SDA" ; print("Small deformation selected") # Small deformation
eps=deformation(readposcar(contcar_ref)[1],readposcar(contcar_dis[0])[1],deftype=def_type)
print(eps)

# -> Some constants
a_0=readposcar(contcar_ref)[0]
T=readposcar(contcar_ref)[1]*a_0
Surf=np.cross(T[0],T[1])[2]
burg=[0,0,T[2][2]]
h=np.linalg.norm(burg)

## -> stress
# outcar of the perfect crystal
outcar_ref="/home/akraych/workdir/2_Wbcc/PP_neb/Moments_dipolaires/calculs/Maxparam/ref/OUTCAR"
# outcar files along the path 
outcar_dis=sorted(glob.glob("/home/akraych/workdir/2_Wbcc/PP_neb/Moments_dipolaires/calculs/Maxparam/1_calc/NEB/ulPP/OUTCAR*"))
stress_ref=findstress(outcar_ref)
Ax=[] ; Ay=[]
for i,name in enumerate(outcar_dis):
    if i < 9 :
        stress=findstress(outcar_dis[i])-stress_ref 
        Ax.append(np.abs(calcA(stress,eps))[1]/a_0)
        Ay.append(np.abs(calcA(stress,eps))[0]/a_0)

## Position atomes
x_atom=[] ; y_atom=[]
x_atom.append(np.sqrt(6.0)+np.sqrt(6)/3.0) ; y_atom.append(2*np.sqrt(2.0))
x_atom.append(np.sqrt(6.0)+np.sqrt(6)/2.0); y_atom.append(2*np.sqrt(2.0)+np.sqrt(2.0)/2)
x_atom.append(np.sqrt(6.0)+2*np.sqrt(6)/3.0); y_atom.append(2*np.sqrt(2.0))
x_atom.append(2*np.sqrt(6.0)-np.sqrt(6)/6.0); y_atom.append(2*np.sqrt(2.0)+np.sqrt(2.0)/2)
x_atom.append(2*np.sqrt(6.0)); y_atom.append(2*np.sqrt(2.0))

## Reperes position dislo
repx=[] ; repy=[]
repx.append(2*np.sqrt(6.0)-1/np.sqrt(6.0)) ; repy.append(2*np.sqrt(2.0)+np.sqrt(2.0)/6.0)
repx.append(np.sqrt(6.0)+np.sqrt(6.0)/2.0) ; repy.append(2*np.sqrt(2.0)+np.sqrt(2.0)/6.0)
repx.append(2*np.sqrt(6.0)-np.sqrt(6.0)/3) ; repy.append(2*np.sqrt(2.0)+np.sqrt(2.0)/3.0)

## Trajectoires dislocations
lP=np.sqrt(6.0)/3 ; xc=2*np.sqrt(6.0)-np.sqrt(6.0)/6+7.5*lP/2 ; yc=2*np.sqrt(2.0)+np.sqrt(2.0)/4 

# Dislocation 2
# initiale & finale
x2_dis_ini=xc+Ax[0]/2.-8*lP ;           y2_dis_ini=yc+Ay[0]/2.
x2_dis_fin=xc-lP/2.+Ax[-1]/2.-8*lP ;    y2_dis_fin=yc+Ay[-1]/2.
# trajectoire (linéaire)
x2_dis=[] ; y2_dis=[]
x2_dis.append(x2_dis_ini) ; y2_dis.append(y2_dis_ini)
for i in range(1,9) :
    x2_dis.append(x2_dis[i-1]-abs(x2_dis_fin-x2_dis_ini)/8.0)
    y2_dis.append(y2_dis[i-1]+abs(y2_dis_fin-y2_dis_ini)/8.0)

# Dislocation 1
# initiale & finale
x1_dis_ini=xc-Ax[0]/2.0 ;           y1_dis_ini=yc-Ay[0]/2.0
x1_dis_fin=xc-lP/2.0-Ax[-1]/2.0 ;   y1_dis_fin=yc-Ay[-1]/2.0
# trajectoire
x1_dis=[] ; y1_dis=[]
xp=[x1_dis_ini,x1_dis_fin] ; yp=[y1_dis_ini,y1_dis_fin]
#x1_dis.append(x1_dis_ini) ; y1_dis.append(y1_dis_ini)
for i in range(0,9):
    x1_dis.append(x2_dis[i]+8*lP-Ax[i]) ; y1_dis.append(y2_dis[i]-Ay[i])


## Graph
fig=plt.figure(figsize=(10,5))
# atomes
ax=fig.add_subplot(111)
ax.plot(x_atom,y_atom,'ro',markersize=20,color='blue')
ax.plot(repx,repy,'+',markersize=10,color='black')
ax.plot(x1_dis,y1_dis,'ro',markersize=5,color='black')
ax.plot(x2_dis,y2_dis,'ro',markersize=5,color='black')
ax.plot(xp,yp,'ro',markersize=5,color='green')

plt.show()

# Sortie : positions réelles (non modifiees pour le graph)
x2_dis=x2_dis+8.0*lP
np.savetxt('xy.txt',np.c_[x1_dis,y1_dis,x2_dis,y2_dis],header="x1 y1 x2 y2")



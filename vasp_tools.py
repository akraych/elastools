import numpy as np
import sys
import re

## Read POSCAR

# initialize the base
# myfile=sys.argv[1]

def readposcar(poscarfile):
    T=np.zeros((3,3))
    with open(poscarfile,"r") as myfile:
        comment1=str(myfile.readline())
        a_0=float(myfile.readline())
        for i in [0,1,2]:
            T[i]=np.array(re.findall(r"[-+]?\d*\.*\d+",myfile.readline())).astype(float)
        A=myfile.readline()
        if isinstance(A,str) == True :
            species=str(A)
            nbatom=np.array(re.findall(r"(?<![-.])\b[0-9]+\b(?!\.[0-9])",myfile.readline())).astype(int) 
        if isinstance(A,int) == True :
            nbatom=np.array(re.findall(r"(?<![-.])\b[0-9]+\b(?!\.[0-9])",myfile.readline())).astype(int) 
        dataformat=myfile.readline()[0]   # dataformat="D" : fractional ; dataformat="C" : cartesian
        #print(dataformat)
        atoms=np.zeros((nbatom[0],3))
        for i in range(0,nbatom[0]):
            atoms[i]=np.array(re.findall(r"[-+]?\d*\.*\d+",myfile.readline())).astype(float)
            if dataformat == "D" :
                atoms[i]=np.matmul(np.transpose(T),atoms[i])
    myfile.close()
    return a_0,T,atoms,species

## Write POSCAR

def echo(arr,new_pos):
    for k in arr:
        new_pos.write("%s " % k)
#formatted = [[format(v) for v in r] for r in m]

def writeposcar(a_0,T,atoms,species,myfile):
    new_pos=open(myfile,"w")
    new_pos.write("I am temporary \n")
    new_pos.write("%s\n" % a_0)
    for i in [0,1,2]:
        echo(T[i],new_pos)
        new_pos.write("\n")
    new_pos.write("%s" % species)
    new_pos.write("%s\n" % atoms.shape[0])
    new_pos.write("%s\n" % "C")
    for i in range(0,atoms.shape[0]):
        echo(atoms[i],new_pos)
        new_pos.write("\n")

# Extract data from outcar and convert in GPa

# Create stress from chi, sigma, tau
def crea_sig(elem):
    sigout=np.zeros((3,3))
    chi=elem[0]
    sigma=elem[1]
    tau=elem[2]
    sigout[0][0]=-sigma*np.cos(2*chi/180*np.pi)
    sigout[1][1]=sigma*np.cos(2*chi/180*np.pi)
    sigout[0][1]=-sigma*np.sin(2*chi/180*np.pi)
    sigout[0][2]=-tau*np.sin(chi/180*np.pi)
    sigout[1][2]=tau*np.cos(chi/180*np.pi)
    sigout[1][0]=sigout[0][1]
    sigout[2][0]=sigout[0][2]
    sigout[2][1]=sigout[1][2]
    return sigout

def findstress(outcar):
    for line in open(outcar):
        if "in kB" in line:
            stress=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
            stress=stress.astype(np.float)
            # Conversion en GPa
            stress=-stress*0.1
    B=np.zeros((3,3))
    B[0][0]=stress[0]
    B[1][1]=stress[1]
    B[2][2]=stress[2]
    B[0][1]=stress[3] ; B[1][0]=B[0][1]
    B[1][2]=stress[4] ; B[2][1]=B[1][2]
    B[0][2]=stress[5] ; B[2][0]=B[0][2]
    return B

    return stress

def findenergy(outcar):
    for line in open(outcar):
        if "TOTEN" in line:
            energy=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
            energy=energy.astype(np.float)
    return energy[0]



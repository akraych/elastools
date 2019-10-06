import numpy as np
import sys
import re

class Vasp :
    executable="/home/akraych/bin/vasp.5.4.1/bin/vasp_std"
    def __init__(self):
        self.version="5.4.1"

    def linetofloat(line):
        line=np.array(line.split()).astype(float)
        return line

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
    
    def findstress(outcar="./OUTCAR"):
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
    
    def findcij(outcar="./OUTCAR"): #not tested yet
        acv=0
        C=[]
        for line in open(outcar):
            if acv==0:
                if "SYMMETRIZED ELASTIC MODULI" in line:
                    acv=1 
                    pass
            elif acv==1:
                C.append(line.split()[1:-1])
                acv=acv+1
            elif acv==7:
                print(C)
                break

#        return stress  ##### What the heck???
    def findmag(oszicar='OSZICAR'):
        mag=[]
        for line in open(oszicar):
            if "mag=" in line :
                tmpmag=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
                tmpmag=tmpmag.astype(np.float)
                tmpmag=tmpmag[-1]
        return(tmpmag)

    def findenergy(outcar='OUTCAR'):
        for line in open(outcar):
            if "TOTEN" in line:
                energy=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
                energy=energy.astype(np.float)
        return energy[0]

    def last_step(oszicar):
        isteps=[]
        esteps=[]
        i=0
        for line in open(oszicar):
            if "F=" in line :
                isteps.append(line)
                i=i+1
                esteps.append(re.findall(r"[-+]?\d*\.*\d+",line_1)[0])
            line_1=line
        return(len(isteps),esteps)

    def band_structure(outcar="OUTCAR",eigenval="EIGENVAL"):
        # Find number of kpoints and bands
        with open(outcar,"r") as myfile:
            for line in myfile:
                if "NBANDS=" in line:
                    nkp=int(line.split()[3])
                    nbands=int(line.split()[-1])
                if "E-fermi" in line:
                    E_f=float(line.split()[2])
        
        # Read the eigenvalues
        band_structure=np.zeros((2,nbands,nkp))
        with open(eigenval,"r") as myfile:
            for i in range(7):
                myfile.readline()
            for i in range(nkp):
                myfile.readline()
                for j in range(nbands):
                    line=myfile.readline().split()
                    # Spin up
                    band_structure[0][j][i]=np.array(line[1]).astype(float)
                    # Spin down
                    band_structure[1][j][i]=np.array(line[2]).astype(float)
                myfile.readline()
        return [band_structure,nbands,nkp,E_f]

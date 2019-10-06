import numpy as np
import re
import os

## Write simulation cell

class dftb :
    executable="/home/akraych/bin/dftbplus-18.1.x86_64-linux/bin/dftb+"

    def __init__(self,executable=executable,version="version"):
        self.version = version
        self.executable = executable

    # Extract data from detailed.out
    def findenergy(filename='detailed.out',mod="normal"):
        def HtoeV(a):
            return(a*27.211384523)
        for line in open(filename):
            if "Total energy" in line:
                Etot=HtoeV(float(line.split()[2]))
            if "Repulsive energy" in line :
                Erep=HtoeV(float(line.split()[2]))
            if "Total Electronic energy" in line :
                E_el=HtoeV(float(line.split()[3]))
        #print(Etot,Erep,E_el)
        if mod=="normal" :
            return Etot
        else :
            return Etot,E_el,Erep
    # Create Energy and force file from dftb.out
    def ef_file(filename='dftb.out'):
        E,F=[],[]
        for line in open(filename,'r'):
            if "Energy" in line :
                E.append(float(line.split()[:-1][4]))
            if "force" in line :
                F.append(float(line.split()[-1]))
        return np.array(E),np.array(F)

    def findEmag(filename='detailed.out'):
        for line in open(filename):
            if "Energy SPIN:" in line:
                mag=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
                mag=mag.astype(np.float)[1]
        return mag
    
    def findmag(filename='detailed.out'):
        for line in open(filename):
            #print(line)
            if "Input / Output electrons (down)" in line:
                down=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
                down=down.astype(np.float)[-1]
            if "Input / Output electrons (up)" in line:
                up=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
                up=up.astype(np.float)[-1] #; print(up)
        mag=abs(up-down)
        return mag

    def findstress(filename='detailed.out'):
        stress=np.zeros((3,3))
        with open(filename,"r") as myfile:
            read = int(3)
            for line in myfile:
                if read < 3:
                    stress[read]=line.split()
                    read=read+1
                if "Total stress tensor" in line:
                    read=0
        return(-1*stress*29421.02648438959) # Converti en GPa

    # Counting KP and number of bands, finding fermy energy
    def band_structure(dftb_input="dftb_in.hsd",bandfile="band.out",outfile="detailed.out"):
        with open(dftb_input,"r") as myfile:
            for line in myfile:
                if "Klines" in line :
                    nkp=0
                    kpline=myfile.readline()
                    while "}" not in kpline:
                        nkp=nkp+int(kpline.split()[0])
                        kpline=myfile.readline()
        
        with open(bandfile,"r") as myfile:
            nbands=0
            line=myfile.readline()
            line=myfile.readline()
            while len(line.split()) > 0:
                nbands=nbands+1
                line=myfile.readline()
        
        with open(outfile,"r") as myfile:
            for line in myfile:
                if "Fermi" in line:
                    E_f=float(line.split()[4])
        
        # Read the eigenvalues
        band_structure=np.zeros((2,nbands,nkp))
        with open(bandfile,"r") as myfile:
            for i in range(nkp):
                myfile.readline()
                # Spin Up : 0 / Down : 1
                for j in range(nbands):
                    line=myfile.readline().split()
                    band_structure[1][j][i]=np.array(line[1]).astype(float)
                myfile.readline()
            for i in range(nkp):
                myfile.readline()
                for j in range(nbands):
                    line=myfile.readline().split()
                    band_structure[0][j][i]=np.array(line[1]).astype(float)
                myfile.readline()

        return [band_structure,nbands,nkp,E_f]
        
    # Execution of a calculation 
    def Exec(command=executable):
        os.system(command)
        # Ajouter la gestion des erreurs (try)

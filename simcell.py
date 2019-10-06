import numpy as np
import os
from elements import ELEMENTS
import datetime

# Object : a simulation cell

version='simcell_3.0'

class simcell(object):
    a0 = 1.0
    pervec = np.empty((3,3))
    coortype = 'C' # C pour coord. cartesienes et F pour coord. fractionelles
    comment=''
    center=np.array([0,0,0])
    nbatom = [1]
    atcoor = [np.empty((int(sum(nbatom)),3))]
    sp = ['H']

    def __init__(self,a0=a0,pervec=pervec,nbatom=nbatom,atcoor=atcoor,comment=comment,sp=sp,coortype=coortype,center=center):
        self.a0 = a0
        self.sp = sp
        self.pervec = pervec
        self.nbatom = nbatom
        self.atcoor = atcoor
        self.comment = comment
        self.coortype = coortype
        self.center = center
    

    def destroy_a0(self):
    # set a_0 (the scaling parameter of the cell) to 1.0
       if self.coortype == 'C' :
           self.pervec = self.pervec*self.a0
           for i,groups in enumerate(self.atcoor):
               self.atcoor[i]=self.atcoor[i]*self.a0
           self.a0=1.0
       if self.coortype == 'F' :
           self.pervec = self.pervec*self.a0
           self.a0=1.0

    def make_a0(self,a_0):
    # make a_0
        self.pervec=self.pervec/a_0
        for i,atom_groups in enumerate(self.atcoor) :
            self.atcoor[i] = self.atcoor[i]/a_0
        self.a0=a_0

    def switch(self):
    # Switch from cartesian coordinates to fractional coordinates (or the opposite)

        def build_atom(nbatom,sp):
        
            array_ini=[]
            for i,sp in enumerate(sp):
                myarray=np.zeros((nbatom[i],3))
                array_ini.append(myarray)
            return np.asarray(array_ini)

        P=np.transpose(self.pervec)
        newatom=build_atom(self.nbatom,self.sp)
        if self.coortype=='C':
            for i,groups in enumerate(self.atcoor):
                for j,vec in enumerate(groups):
                    newatom[i][j]=np.matmul(np.linalg.inv(P),vec)
                self.coortype='F'
        elif self.coortype=='F':
            for i,groups in enumerate(self.atcoor):
                for j,vec in enumerate(groups):
                    newatom[i][j]=np.matmul(P,vec)
                self.coortype='C'
        else :
            print("Atomic format non informed - cant switch")
            exit()
        self.atcoor=newatom

    def unwrap(self,ref): 
    # unwrap cell to match ref
        for myrange in [[0,1,2],[1,2,0],[2,0,1],[1,2,0]]: # On repete plusieurs fois pour eviter de se planter sur des combinaisons de vecteurs de periodicite
        #for myrange in [[0,1,2]]: # On repete plusieurs fois pour eviter de se planter sur des combinaisons de vecteurs de periodicite
            for i in range(int(sum(self.nbatom))):
                for j in myrange:
                    factor = 0.5
                    dif=ref.atcoor[i][j] - self.atcoor[i][j]
                    if dif > 0 :
                        if dif > factor*np.linalg.norm(self.pervec[j]) :
                            self.atcoor[i]=self.atcoor[i]+self.pervec[j]
                    elif dif < 0 :
                        if dif < -factor*np.linalg.norm(ref.pervec[j]) :
                            self.atcoor[i]=self.atcoor[i]-self.pervec[j]

    def interpolate(ei,ef,r,burg=[0,0,5]):
        # Calculate the center of mass motion of the cell (avoid drift during interpolation)
        def CDMM(ei=ei,ef=ef):
            cdm=0
            for i in range(len(ei.atcoor)):
                Xdif=ef.atcoor[i]-ei.atcoor[i]
                cdm+=np.array([np.sum(Xdif[:,0]),np.sum(Xdif[:,1]),np.sum(Xdif[:,2])])/Xdif.shape[0]
            return cdm
        cdm=CDMM()
       
        Xint=[]
        for o in range(len(ei.atcoor)):
            print("Looking for {} ...".format(ei.sp[o]))
            Xi=ei.atcoor[o] ; Xf=ef.atcoor[o] ; Xdif=Xf-Xi
            detected = 0
            for i,vec in enumerate(Xdif):
                if np.linalg.norm(vec) > np.linalg.norm(burg/2):
                    detected = 1
                    print("LARGE DISP. atom {} vec {} dist {}".format(i+1,vec,np.linalg.norm(vec)))
                    if vec[2] > 0:
                        Xdif[i]=vec-burg
                    else :
                        Xdif[i]=vec+burg

            checker=0
            for i,vec in enumerate(Xdif):
                if np.linalg.norm(vec) > np.linalg.norm(burg/2):
                    print("Unsolved for atom {}".format(i+1))
                    checker=1
            if checker == 0 and detected == 1:
                print("However, problem seems to be solved")
            if detected == 0 and checker == 0:
                print("Everything is fine with this one")
            print("\n")
            # X interpolated
            Xint.append(Xi+r*(Xdif-cdm))
        return simcell(a0=ei.a0,pervec=ei.pervec,nbatom=ei.nbatom,coortype=ei.coortype,atcoor=Xint,sp=ei.sp)


    def shear(self,eps,mod="1B"):
    # shear a cell
        def applyshear(eps,vec):
            new_vec=vec+np.einsum('ij,j',eps,vec)
            return(new_vec)
        if self.coortype=='F':
            self.switch()
        for i,vec in enumerate(self.pervec) : 
            self.pervec[i]=applyshear(eps,vec)
        for i,atom_groups in enumerate(self.atcoor) :
            for j,vec in enumerate(atom_groups) :
                self.atcoor[i][j]=applyshear(eps,vec)

    def elastic(self,C,d1,d2):
        binar_babel='/home/akraych/bin/Babel_V10.0/bin/babel'
        babel_model='/home/akraych/scripts/py/input_babel_model'
        # extracted from cell : filename to get periodicity vectors
        # C : elastic constants in the good orientation, in vogt notation
        # xd1 : (x1,y1,b1) : x,y coordinates of the dislocation 1, and burgers vector
        # xd2 : ---------- : ------------------------------ 2
        self.write('tmp_cell.xyz',fmt='xyz')
        filename='tmp_cell.xyz'
        babel_file=open(babel_model,'r')
        text=babel_file.read()
        text=text.replace('INPUTFILE','tmp_cell.xyz',1)
        Ctxt=''
        for i in range(6):
            for j in range(6):
                if C[i][j]==0:
                    pass
                else:
                    Ctxt+='Cvoigt({},{})={} \n'.format(i+1,j+1,C[i][j])
        text=text.replace('INPUTFILE','tmp_cell.xyz')
        text=text.replace('CXX',Ctxt)

        text=text.replace('d1x','{}'.format(d1[0]))
        text=text.replace('d1y','{}'.format(d1[1]))
        text=text.replace('b1','{}'.format(d1[2]))

        text=text.replace('d2x','{}'.format(d2[0]))
        text=text.replace('d2y','{}'.format(d2[1]))
        text=text.replace('b2','{}'.format(d2[2]))
        
        text=text.replace('CUTOFF','{}'.format(abs(d1[2])))
        with open('tmpbabel','w') as myfile:
            myfile.write(text)
        os.system('{} {} > tmpbabel_output'.format(binar_babel,'tmpbabel'))
        babelfile='tmpbabel_output'
        with open(babelfile,'r') as myfile:
            for line in myfile:
                if 'Total elastic energy' in line:
                    E0=float(line.split()[-6])
        return(E0)

    def read(self,name_of_file,fmt='nothing'):
    # read a cell file
        # Recognize format
        def rec(name_of_file):
            # vasp
            if 'POSCAR' in name_of_file or 'CONTCAR' in name_of_file :
                return 'vasp'
            else:
                if name_of_file.split('.')[-1] == 'gen':
                    return 'gen'
                if name_of_file.split('.')[-1] == 'xyz':
                    return 'xyz'
        
        if fmt=='nothing':
            fmt=rec(name_of_file)

        # General functions to read a line
        def ltf(line,integer='no'):
            if integer=='yes':
                return np.array(line.split()).astype(int)
            else :
                return np.array(line.split()).astype(float)
        #self.pervec=np.empty((3,3)) # Avoid memory overlap

        # Est-ce que c'est un entier?
        def try_int(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False

        # To avoid memory problems
        def end(T,atoms,species,nbatom):
            self.pervec=T
            self.atcoor=atoms
            self.sp=species
            self.nbatom=nbatom

        # Initialize
        T=np.zeros((3,3))
        species=[]
        nbatom=[]
        def build_atom(nbatom):
            array_ini=[]
            for i,nbre in enumerate(nbatom):
                myarray=np.zeros((nbre,3))
                array_ini.append(myarray)
            return np.asarray(array_ini)
        
        # Collect species
        def collect(line,species,atoms):
            sp=line[0]

            if try_int(sp) == True :
                sp=int(sp)
            sp=ELEMENTS[sp].symbol
            at=line[1:]
            if not species :
                species = [sp]
                atoms = [[]]
            elif sp in species : 
                pass
            else :
                species.append(sp)
                atoms.append([])
            atoms[species.index(sp)].append(at)
            return species,atoms

        with open(name_of_file,'r') as myfile:
            self.comment="{} {}".format(version,datetime.datetime.now())
            # VASP
            if fmt=='vasp':
                comment=str(myfile.readline())[:-1]
                self.comment="{1} - modified : {0}".format(self.comment,comment)
                self.a0=ltf(myfile.readline())[0]
                for i in range(3):
                    T[i]=ltf(myfile.readline())
                A=myfile.readline().split()
                if try_int(A[0]) ==  False :
                    for sp in A:
                        species.append(ELEMENTS[sp].symbol)
                    B=myfile.readline().split()
                    for nb in B :
                        nbatom.append(int(nb))
                else :
                    for nb in A :
                        nbatom.append(int(nb))
                coty=str(myfile.readline())[0]
                if coty == 'D' : # Selective dynamics is not read
                    self.coortype = 'F'
                elif coty == 'C' :
                    self.coortype = 'C'
                atoms=build_atom(nbatom)
                for i,sp in enumerate(atoms):
                    for j,nb in enumerate(atoms[i]):
                        atoms[i][j]=ltf(myfile.readline())
                end(T,atoms,species,nbatom)

            # XYZ format
            if fmt=='xyz':
                self.coortype = 'C'
                nb_total=int(myfile.readline().split()[0])
                comment=str(myfile.readline())[:-1]
                self.comment="{1} - modified : {0}".format(self.comment,comment)
                # Counting the number of different species
                atoms_l=[]
                species=[]
                for i in range(nb_total):
                    species,atoms_l=collect(myfile.readline().split(),species,atoms_l)
                line=myfile.readline()
                line=myfile.readline()
                if not line : 
                    print("No periodicity vectors in {}".format(name_of_file))
                else : 
                    T[0]=ltf(line)
                    T[1]=ltf(myfile.readline())
                    T[2]=ltf(myfile.readline())
                    myfile.readline()
                    self.a0=float(myfile.readline().split()[0])
                    for parts in atoms_l:
                        nbatom.append(len(parts))
                # To have atoms as numpy elements 
                atoms=build_atom(nbatom)
                for i,groups in enumerate(atoms_l):
                    atoms[i]=atoms_l[i]
                end(T,atoms,species,nbatom)
    
    # write a cell file
    def write(self,name_of_file,fmt='nothing'):

        # Recognize format
        def rec(name_of_file):
            # vasp
            if name_of_file == 'POSCAR' or name_of_file == 'CONTCAR' :
                return 'vasp'
            else:
                if name_of_file.split('.')[-1] == 'gen':
                    return 'gen'
                if name_of_file.split('.')[-1] == 'xyz':
                    return 'xyz'
                if name_of_file.split('.')[-1] == 'cfg':
                    return 'cfg'
        
        if fmt=='nothing':
            fmt=rec(name_of_file)

        # Formats : cfg / vasp / xyz
        with open(name_of_file,'w') as myfile:
            if fmt=='cfg':
                if self.coortype=='C':
                    self.switch()
                myfile.write("Number of particles = {}\n".format(int(sum(self.nbatom))))
                myfile.write("#{}\n".format(self.comment))
                myfile.write("A = {} Angstrom (basic length-scale)\n".format(self.a0))
                for i in range(3):
                    for j in range(3):
                        myfile.write("H0({0},{1}) = {2:.8f} A\n".format(i+1,j+1,self.pervec[i][j]))
                myfile.write(".NO_VELOCITY.\n")
                myfile.write("entry_count = 3\n") # Le nombre de colonne (a modifier si on veut ajouter des proprietes aux atomes   
                for i,blocks in enumerate(self.atcoor):
                    myfile.write("{}\n".format(ELEMENTS[self.sp[i]].mass))
                    myfile.write("{}\n".format(self.sp[i]))
                    for j,vec in enumerate(self.atcoor[i]):
                        myfile.write("{0:.8f} {1:.8f} {2:.8f}\n".format(*vec))

            elif fmt=='vasp':
                myfile.write("{}\n".format(self.comment))
                myfile.write("{}\n".format(self.a0))
                for i in range(3):
                    myfile.write("{0:.8f} {1:.8f} {2:.8f}\n".format(self.pervec[i][0],self.pervec[i][1],self.pervec[i][2]))
                for sp in self.sp:
                    myfile.write("{} ".format(sp))
                myfile.write("\n")
                for nb in self.nbatom:
                    myfile.write("{} ".format(int(nb)))
                myfile.write("\n")
                myfile.write("{}\n".format(self.coortype))
                for blocks in self.atcoor:
                    for vec in blocks:
                        myfile.write("{0:.8f} {1:.8f} {2:.8f}\n".format(vec[0],vec[1],vec[2]))

            elif fmt=='xyz':
                if self.coortype=='F':
                    self.switch()
                myfile.write("{}\n".format(int(sum(self.nbatom))))
                myfile.write("{}\n".format(self.comment))
                for i,blocks in enumerate(self.atcoor):
                    for vec in blocks:
                        myfile.write("{0} {1:.8f} {2:.8f} {3:.8f}\n".format(ELEMENTS[self.sp[i]].number,*vec))
                myfile.write("\n")
                for i in range(3):
                    myfile.write("{0:.12f} {1:.12f} {2:.12f}\n".format(self.pervec[i][0],self.pervec[i][1],self.pervec[i][2]))
                myfile.write("\n")
                myfile.write("{}".format(self.a0))
            elif fmt=='gen':
                #self.destroy_a0()
                if self.coortype=='C':
                    CT='S'
                elif self.coortype=='F':
                    CT='F'
                myfile.write("{} {}\n".format(int(sum(self.nbatom)),CT))
                for elem in self.sp:
                    myfile.write("{} ".format(ELEMENTS[elem].symbol))
                myfile.write("\n")
                cnt=int(0)
                for i,blocks in enumerate(self.atcoor):
                    for vec in blocks:
                        cnt=cnt+1
                        vec=vec*self.a0
                        myfile.write("{0} {1} {2:.8f} {3:.8f} {4:.8f}\n".format(cnt,i+1,*vec))
                myfile.write("{0:.12f} {1:.12f} {2:.12f} \n" .format(*self.center))
                for i in range(3):
                    myfile.write("{0:.12f} {1:.12f} {2:.12f}\n".format(self.a0*self.pervec[i][0],self.a0*self.pervec[i][1],self.a0*self.pervec[i][2]))

    def change_geometry(self,geom="TO"):
        if self.coortype == 'F':
            self.switch()
        newT=self.pervec
        if geom == "TO" : # trap to ortho
           newT[0]=self.pervec[0]-self.pervec[1]
           self.pervec=newT
        if geom == "OT" : # ortho to trap
           newT[0]=self.pervec[0]+self.pervec[1]
           self.pervec=newT

import os
import numpy as np
import sys
import re

class Lammps :
    executable="/home/akraych/bin/lammps-11Aug17/src/lmp_serial"
    def __init__(self):
        self.version="11Aug17"
    def writeatoms(a_0,T,atoms,species,myfile): # species is a tuble (species="Fe","C" for instance)
        new_pos=open(myfile,"w")
        new_pos.write("# Hello yes this is dog \n")
        new_pos.write(" \n")
        new_pos.write("{0} atoms \n".format(len(atoms)))
        new_pos.write("{0} atom types \n".format(len(species)))
        new_pos.write(" \n")
        T=T*a_0 ; atoms=atoms*a_0
        a_x=np.linalg.norm(T[0]) # A,B,C=T[0],T[1],T[2]
        b_x=np.dot(T[1],T[0]/a_x)
        b_y=np.linalg.norm(np.cross(T[0]/a_x,T[1]))
        c_x=np.dot(T[2],T[0]/a_x)
        c_y=np.dot(T[2],np.cross(np.cross(T[0],T[1])/np.linalg.norm(np.cross(T[0],T[1])),T[0]/a_x))
        c_z=np.linalg.norm( np.dot( T[2] , np.cross(T[0],T[1])/np.linalg.norm(np.cross(T[0],T[1])) ) )
        a,b,c=[a_x,0,0],[b_x,b_y,0],[c_x,c_y,c_z] 
        new_pos.write("0.0 {:.8f} xlo xhi \n".format(a_x))
        new_pos.write("0.0 {:.8f} ylo yhi \n".format(b_y))
        new_pos.write("0.0 {:.8f} zlo zhi \n".format(c_z))
        new_pos.write("{:.8f} {:.8f} {:.8f} xy xz yz \n".format(b_x,c_x,c_y))
        new_pos.write(" \n")
        new_pos.write("Atoms \n")
        new_pos.write(" \n")
        for i in range(0,len(atoms)):
            new_pos.write("   {}   1   {:.8f}   {:.8f}   {:.8f}  \n".format(i+1,atoms[i][0],atoms[i][1],atoms[i][2])) # Attention les espaces sont importants

    def findenergy(outname="log.lammps"):
        with open(outname) as myfile:
            eline=myfile.readlines()[-100:-1]
            ind=eline.index('  Energy initial, next-to-last, final = \n')
            energy=re.findall(r"[-+]?\d*\.*\d+",eline[ind+1])
            return(energy[2])
    

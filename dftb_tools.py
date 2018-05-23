import numpy as np
import sys
import re

## Write simulation cell

def echo(arr,new_pos):
    for k in arr:
        new_pos.write("%s " % k)

def writegen(a_0,T,atoms,species,myfile):
    new_pos=open(myfile,"w")
    new_pos.write("%s " % atoms.shape[0])
    new_pos.write("%s \n" % "S")
    new_pos.write("%s" % species)
    for i in range(0,atoms.shape[0]):
        new_pos.write("%s %s " % (i,"1"))
        echo(atoms[i]*a_0,new_pos)
        new_pos.write("\n")
    new_pos.write("%s" % "0.0 0.0 0.0\n")
    for i in [0,1,2]:
        echo(T[i]*a_0,new_pos)
        new_pos.write("\n")

# Extract data from output

def findenergy(filename):
    for line in open(filename):
        if "Total energy" in line:
            energy=np.array(re.findall(r"[-+]?\d*\.*\d+",line))
            energy=energy.astype(np.float)
    return energy[0]


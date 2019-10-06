from simcell import *
from elasticity import *
import random

# Elastic constants
cij_file='./Cij/Cij_k16_Fe_dft'
ela=elastic() # Object : constantes elastiques
ela.load(cij_file)
print('# Cij Voigt notation')
for line in ela.Cv:
    print("{0:.3f} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}".format(*line))
print("")
print('# Sij Voigt notation')
for line in ela.Sv:
    print("{0:.5f} {1:.5f} {2:.5f} {3:.5f} {4:.5f} {5:.5f}".format(*line))
print("")

# Read simuation cell POSCAR
cell=simcell()
cell.read("POSCAR")
# Define a stress (units:GPa)
stress=np.zeros((3,3))
stress[0][0]=1
stress[1][1]=-1
print('Stress is:\n{}\n'.format(stress))
# Calculate the corresponding deformation
eps=np.einsum('ijkl,kl',ela.S,stress)
print('Corresponding deformation is:\n{}\n'.format(eps))
# Apply this deformation to our simulation cell
cell.shear(eps)
# Write the new cell
cell.write("POSCAR_sheared",fmt='vasp')
# Re-find the deformation that was applied
cell2=simcell()
cell2.read("POSCAR")
Def=deformation(cell2.pervec,cell.pervec)
print("check:\n{}\n".format(Def))
# Change the element
for i,ele in enumerate(cell.sp):
    cell.sp[i]=ELEMENTS[random.randint(1,109)]
print("New elements are:")
for ele in cell.sp:
    print(ele)


import signal
import time as time
import numpy as np
import sys
import re
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# ASE
from ase.calculators.dftb import Dftb
from ase import Atoms
from ase.optimize import BFGS
from ase.io import read,write
from ase.io.trajectory import Trajectory

# Optimizer
from scipy.optimize import minimize
from scipy.optimize import basinhopping

# Dependences
sys.path.insert(1,"/home/akraych/scripts/py/")
from vasp_tools import *
from dftb_tools import *
from elasticity import *
from simcell_v3 import *
#from graph_settings import *

#### Reference DFT calculation ####
##
# Constantes elastiques calculees en DFT
C11_DFT=291 ; C12_DFT=150 ; C44_DFT=104 ; a0_DFT=2.83
## Energy de reference 
OUTCAR="/home/akraych/workdir/4_FeC/2_bulk_prop_Fe/0_variation_a0/1_DFT/0_nomag/smearing0.2/isolated_atom/OUTCAR"
E0_DFT=Vasp.findenergy(outcar=OUTCAR)
## Energy | a0 range (1.95 -> 3.2)
path="/home/akraych/workdir/4_FeC/2_bulk_prop_Fe/0_variation_a0/1_DFT/0_nomag/smearing0.2"
File="{}/Erep.dat.save".format(path)
x=np.loadtxt(File)[:,0] ; x1=min(x) ; x2=max(x)
y=np.loadtxt(File)[:,1]-E0_DFT
E_DFT=UnivariateSpline(x,y,k=4,s=0) 
# save
#with open("E_DFT.dat","w") as myfile:
#    for i in range(len(x)):
#        myfile.write("{} {} \n".format(x[i],y[i]))
## Electronic band structure
path="/home/akraych/workdir/4_FeC/2_bulk_prop_Fe/2_eBS/1_DFT"
BS=Vasp.band_structure(outcar="{}/OUTCAR".format(path),eigenval="{}/EIGENVAL".format(path))
band_structure_dft=BS[0]
nbands_dft,nkp_dft=BS[1],BS[2]
E_f_dft=BS[-1]

## Magnetisation(a0)
path="/home/akraych/workdir/4_FeC/2_bulk_prop_Fe/0_variation_a0/1_DFT/1_withmag/res_Occ"
File="{}/E_DFT.dat".format(path)
x=np.loadtxt(File)[:,0][::-1]
y=np.loadtxt(File)[:,2][::-1]
mu_DFT=UnivariateSpline(x,y,k=5,s=0)
X=np.linspace(min(x),max(x),100)

# Basic functions
def RM(File):
    if os.path.exists("./{}".format(File)):
        os.system("rm {}".format(File))

def Reset(a0=1):
    pervec=np.array([[0.5,0.5,-0.5],[-0.5,0.5,0.5],[0.5,-0.5,0.5]])
    atoms=[np.zeros((1,3))]
    poscar=simcell(a0=a0,pervec=pervec,nbatom=[1],atcoor=atoms,comment='the dude abides',sp=['Fe'],coortype='C',center=[0,0,0])
    return poscar

#### Functions to construct the skf file #####
##
## 1 - Constructing the frame of a skf file (pseudo-atom H, Slater det.)
## List of modules used : numpy | os 
## We just use a bash script + scfatom | all the environment is located in the folder work_fold
def qHub(r0_phi_q,work_fold="/home/akraych/workdir/4_FeC/1_parametrisation_dftb/2_scaling_spin_coupling/qHub"):
    File="{}/qHub.dat".format(work_fold)
    r_hub=np.loadtxt(File)[:,0][::-1]
    hub_s=np.loadtxt(File)[:,1][::-1] ; sps=UnivariateSpline(r_hub,hub_s,k=3,s=0)
    hub_p=np.loadtxt(File)[:,2][::-1] ; spp=UnivariateSpline(r_hub,hub_p,k=5,s=1e-4)
    hub_d=np.loadtxt(File)[:,3][::-1] ; spd=UnivariateSpline(r_hub,hub_d,k=3,s=0)
    X=np.linspace(min(r_hub),20,100)
    plt.xlim(left=0,right=20)
    plt.plot(r_hub,hub_s,'.',label='$U_s$',color='C0') ; plt.plot(X,sps(X),color='C0')
    plt.plot(r_hub,hub_p,'.',label='$U_p$',color='C1') ; plt.plot(X,spp(X),color='C1')
    plt.plot(r_hub,hub_d,'.',label='$U_d$',color='C2') ; plt.plot(X,spd(X),color='C2')
    plt.legend()
    plt.xlabel('$r_{U}$ (ua)')
    plt.ylabel('$U_{Fe,l}$ (Ha)')
    plt.savefig("U_A.svg")
    plt.clf()
    return(sps(r0_phi_q),spp(r0_phi_q),spd(r0_phi_q))

def create_skf(r0_V,r0_phi,r0_phi_q,work_fold="/home/akraych/scripts/dftb+_skf/create_skf"):
    with open(work_fold+"/mktwo","r") as oldfile , open(work_fold+"/mktwo_tmp424","w") as newfile:
        text=oldfile.read().replace('r0phiXXX',str(r0_phi))
        text=text.replace('r0VXXX',str(r0_V))
        newfile.write(text)
    os.system("cp {}/.mktworc .".format(work_fold))
    # Create the skc file
    if not os.path.exists("./mktwo_out"):
        os.system("mkdir mktwo_out") 
    os.system("bash {}/mktwo_tmp424 Fe Fe > mktwo_out/skc_{}_{}_{}.out".format(work_fold,r0_V,r0_phi,r0_phi_q)) # dump FE-FE.skc
    #file_skc="Fe_Fe_r0V{}_r0phi{}_r0phiq{}.skc".format(r0_V,r0_phi,r0_phi_q)
    file_skc="Fe_Fe.skc"
    # Read it and change a few stuff
    Us,Up,Ud=qHub(r0_phi_q)
    with open("FE-FE.skc","r") as myfile, open(file_skc,"w") as newfile:
        # line1 : Radial mesh | line 2 : on-site energies (3), spin correctionm (1), hubbard parameters (3), last states occupancy (3) (https://www.dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf)
        cnt=0
        for line in myfile:
            if len(line.split()) > 0:
                if line.split()[0] == '#' :
                    continue
                elif cnt == 0:
                    newfile.write(line)
                    cnt=cnt+1
                elif cnt == 1:
                    l2=line.split()
                    newline="{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(*l2[0:4],Ud,Up,Us,*l2[7:])
                    newfile.write(newline)
                    cnt=cnt+1
                elif cnt == 2:
                    newfile.write(line)
    return file_skc

def create_sp(r0_V_EM,spFile="/home/akraych/workdir/4_FeC/1_parametrisation_dftb/2_scaling_spin_coupling/SpinHub/sHub.dat"):
    r_hub=np.loadtxt(spFile)[:,0][::-1]
    W=np.zeros((6,len(r_hub)))
    sp=[]
    X=np.linspace(1,20,100)
    color=['blue','green','red','orange','purple','black']
    error=[0,1e-5,0,0,0,0]
    plt.clf()
    label=['$W_{Fe,ss}$','$W_{Fe,pp}$','$W_{Fe,dd}$','$W_{Fe,pd}$','$W_{Fe,sd}$','$W_{Fe,sp}$']
    for i in range(6):
        W[i]=np.loadtxt(spFile)[:,i+1][::-1]
        sp.append(UnivariateSpline(r_hub,W[i],k=4,s=error[i]))
        plt.plot(r_hub,W[i],'.',label=label[i],color=color[i])
        plt.plot(X,sp[i](X),color=color[i])
    plt.xlim(left=0,right=20)
    plt.legend()
    plt.ylabel('$W_{Fe,(l,l\')}$ (Ha)')
    plt.xlabel('$r_{U}$ (ua)')
    plt.savefig('sHub.svg')
    plt.clf()
    A=np.zeros((3,3))
    A[0][0]=sp[0](r0_V_EM)
    A[1][1]=sp[1](r0_V_EM)
    A[2][2]=sp[2](r0_V_EM)
    A[1][2]=sp[3](r0_V_EM)
    A[0][2]=sp[4](r0_V_EM)
    A[0][1]=sp[5](r0_V_EM)
    A[2][1]=sp[3](r0_V_EM)
    A[2][0]=sp[4](r0_V_EM)
    A[1][0]=sp[5](r0_V_EM)
    return np.around(A,decimals=4)

def add_rep(Repulsion,skf_file="Fe_Fe.skc"):
    name="Fe_Fe.skf"
    with open(skf_file,"r") as oldfile, open(name,"w") as newfile: 
        for line in oldfile:
            if "Spline" not in line:
                newfile.write(line)
            else:
                newfile.write(line)
                break
        newfile.write(Repulsion)
        return(name)

#### Calculation styles dftb ####
## Minimisation | Mag OFF#
def dftb_mag_off(skf,TMP=2000,UPE=2):
    Filling="MethfesselPaxton { \n Temperature [Kelvin] = %.1d\n Order= 2\n}" %(TMP)
    MAN=[["Fe","d"]] # Max angular momentum ([sp1,couche],[sp2,couche],...)
    scaling=1.0
    A=scaling*np.array([[-0.016,-0.012,-0.003],[-0.012,-0.029,-0.001],[-0.003,-0.001,-0.015]]) # Spin coupling
    calc=Dftb(label='relax_mag_OFF',
              MAG=False,
              skf='{}'.format(skf),
              RSPIN='Yes',
              UPE=UPE,
              Hamiltonian_SCC="Yes",
              Hamiltonian_SCCTolerance ="1E-5",
              Hamiltonian_MaxAngularMomentum=MAN,
              Hamiltonian_KPointsAndWeights = "SupercellFolding {\n 16 0 0\n 0 16 0\n 0 0 16\n 0 0 0\n }",
              Hamiltonian_OrbitalResolvedSCC = "Yes",
              Hamiltonian_MaxSCCIterations = "1000",
              Hamiltonian_Charge="0.0",
              Options_WriteChargesAsText="Yes",
              Hamiltonian_Filling=Filling,
              Parallel_UseOmpThreads=  "Yes",
              Hamiltonian_Mixer="Broyden {\n MixingParameter = 0.05 \n InverseJacobiWeight = 0.01 \n MinimalWeight = 1.0 \n MaximalWeight = 1e5 \n WeightFactor = 100 \n}"
              )
    return calc

## Minimisation | Mag ON 
def dftb_mag_on(skf,TMP=2000,kp=[16,16,16],UPE=3,A=np.array([[-0.016,-0.012,-0.003],[-0.012,-0.029,-0.001],[-0.003,-0.001,-0.015]])):
    Filling="MethfesselPaxton { \n Temperature [Kelvin] = %.1d\n Order= 2\n}" %(TMP)
    MAN=[["Fe","d"]] # Max angular momentum ([sp1,couche],[sp2,couche],...)
    calc=Dftb(label='relax_mag_ON',
              MAG=True,
              skf='{}'.format(skf),
              RSPIN='Yes',
              UPE=UPE,
              Hamiltonian_SCC="Yes",
              Hamiltonian_SCCTolerance ="1E-5",
              Hamiltonian_MaxAngularMomentum=MAN,
              Hamiltonian_KPointsAndWeights = "SupercellFolding {{\n {} 0 0\n 0 {} 0\n 0 0 {}\n 0 0 0\n }}".format(*kp),
              Hamiltonian_OrbitalResolvedSCC = "Yes",
              Hamiltonian_MaxSCCIterations = "1000",
              Hamiltonian_Charge="0.0",
              Hamiltonian_SpinConstants=' { \n  Fe = {\n  '+'{} {} {} \n {} {} {} \n {} {} {} \n'.format(A[0][0],A[0][1],A[0][2],A[1][0],A[1][1],A[1][2],A[2][0],A[2][1],A[2][2])+'  }\n }',
              Options_WriteChargesAsText="Yes",
              Hamiltonian_Filling=Filling,
              Parallel_UseOmpThreads=  "Yes",
              Hamiltonian_Mixer="Broyden {\n MixingParameter = 0.05 \n InverseJacobiWeight = 0.01 \n MinimalWeight = 1.0 \n MaximalWeight = 1e5 \n WeightFactor = 100 \n}"
              )
    return calc

## Minimisation | Mag ON | recherche de parametre de maille
def dftb_mag_on_opt(skf,TMP=2000,kp=[16,16,16],UPE=3,A=np.array([[-0.016,-0.012,-0.003],[-0.012,-0.029,-0.001],[-0.003,-0.001,-0.015]])):
    Filling="MethfesselPaxton { \n Temperature [Kelvin] = %.1d\n Order= 2\n}" %(TMP)
    MAN=[["Fe","d"]] # Max angular momentum ([sp1,couche],[sp2,couche],...)
    calc=Dftb(label='relax_mag_ON',
              MAG=True,
              skf='{}'.format(skf),
              RSPIN='Yes',
              UPE=UPE,
              Hamiltonian_SCC="Yes",
              Driver_LatticeOpt="Yes",
              Hamiltonian_SCCTolerance ="1E-5",
              Hamiltonian_MaxAngularMomentum=MAN,
              Hamiltonian_KPointsAndWeights = "SupercellFolding {{\n {} 0 0\n 0 {} 0\n 0 0 {}\n 0 0 0\n }}".format(*kp),
              Hamiltonian_OrbitalResolvedSCC = "Yes",
              Hamiltonian_MaxSCCIterations = "1000",
              Hamiltonian_Charge="0.0",
              Hamiltonian_SpinConstants=' { \n  Fe = {\n  '+'{} {} {} \n {} {} {} \n {} {} {} \n'.format(A[0][0],A[0][1],A[0][2],A[1][0],A[1][1],A[1][2],A[2][0],A[2][1],A[2][2])+'  }\n }',
              Options_WriteChargesAsText="Yes",
              Hamiltonian_Filling=Filling,
              Parallel_UseOmpThreads=  "Yes",
              Hamiltonian_Mixer="Broyden {\n MixingParameter = 0.05 \n InverseJacobiWeight = 0.01 \n MinimalWeight = 1.0 \n MaximalWeight = 1e5 \n WeightFactor = 100 \n}"
              )
    return calc

# Minimisation of non periodic cell (used to determine the single atom energy)
def dftb_1A(skf,TMP=2000,UPE=2):
    print(skf)
    Filling="MethfesselPaxton { \n Temperature [Kelvin] = %.1d\n Order= 2\n}" %(TMP)
    MAN=[["Fe","d"]] # Max angular momentum ([sp1,couche],[sp2,couche],...)
    scaling=1.0
    A=scaling*np.array([[-0.016,-0.012,-0.003],[-0.012,-0.029,-0.001],[-0.003,-0.001,-0.015]]) # Spin coupling
    calc=Dftb(label='relax_mag_OFF',
              MAG=False,
              skf='{}'.format(skf),
              RSPIN='Yes',
              UPE=UPE,
              Hamiltonian_SCC="Yes",
              Hamiltonian_SCCTolerance ="1E-5",
              Hamiltonian_MaxAngularMomentum=MAN,
              Hamiltonian_OrbitalResolvedSCC = "Yes",
              Hamiltonian_MaxSCCIterations = "1000",
              Hamiltonian_Charge="0.0",
              Options_WriteChargesAsText="Yes",
              Hamiltonian_Filling=Filling,
              Parallel_UseOmpThreads=  "Yes",
              Hamiltonian_Mixer="Broyden {\n MixingParameter = 0.05 \n InverseJacobiWeight = 0.01 \n MinimalWeight = 1.0 \n MaximalWeight = 1e5 \n WeightFactor = 100 \n}"
              )
    return calc
## Minimisation | Mag ON | kline (for BS calculation)
def dftb_BS(skf,TMP=2000,kp=[16,16,16],UPE=3,A=np.array([[-0.016,-0.012,-0.003],[-0.012,-0.029,-0.001],[-0.003,-0.001,-0.015]])):
    Filling="MethfesselPaxton { \n Temperature [Kelvin] = %.1d\n Order= 2\n}" %(TMP)
    MAN=[["Fe","d"]] # Max angular momentum ([sp1,couche],[sp2,couche],...)
    scaling=1.0
    calc=Dftb(label='relax_BS',
              MAG=True,
              skf='{}'.format(skf),
              RSPIN='Yes',
              UPE=UPE,
              Hamiltonian_SCC="Yes",
              Hamiltonian_SCCTolerance ="1E-5",
              Hamiltonian_MaxAngularMomentum=MAN,
              Hamiltonian_MaxSCCIterations = 1,
              Hamiltonian_KPointsAndWeights = "Klines {\n  0 0 0 0           # G\n  50 0.5 -0.5 0.5   # H\n  50 0 0 0.5        # N\n  50 0 0 0          # G\n  50 0.25 0.25 0.25 # P\n  50 0.5 -0.5 0.5   # H\n  0  0.25 0.25 0.25 # P\n  50  0  0  0.5     # N\n  }",
              Hamiltonian_OrbitalResolvedSCC = "Yes",
              Hamiltonian_Charge="0.0",
              Hamiltonian_SpinConstants=' { \n  Fe = {\n  '+'{} {} {} \n {} {} {} \n {} {} {} \n'.format(A[0][0],A[0][1],A[0][2],A[1][0],A[1][1],A[1][2],A[2][0],A[2][1],A[2][2])+'  }\n }',
              Options_WriteChargesAsText="Yes",
              Hamiltonian_Filling=Filling,
              Parallel_UseOmpThreads=  "Yes",
              Hamiltonian_Mixer="Broyden {\n MixingParameter = 0.05 \n InverseJacobiWeight = 0.01 \n MinimalWeight = 1.0 \n MaximalWeight = 1e5 \n WeightFactor = 100 \n}"
              )
    return calc

## Calcul de Erep, methode recursive
def Erep(E_el,E_DFT,x1,Rc): # E_el : DFTB electronic energy
    Rcuts=np.linspace(x1,x2,1e5)
    for Rc in Rcuts:
        spX=UnivariateSpline(X,(E_DFT(Rcuts)-E_DFT(Rc))-(E_el(Rcuts)-E_el(Rc)),k=4,s=0)
        roots=spX.derivative(n=1).roots()
        if abs(spX(roots[0])) < 1E-7 :
            liste_Rc.append(roots[0])
        Rc=np.median(liste_Rc)
    Rcmin = Rc*np.sqrt(3.0/2)/2
    X=np.linspace(Rcmin,Rc,200)                                        # x1 et x2 : limite of the a0 range used in DFT
    fsp=UnivariateSpline(X,(E_DFT(X)-E_DFT(Rc))-(E_el(X)-E_el(Rc)),k=5,s=0)              # Difference DFT DFTB (Energie qui sert a calculer Erep)
    plt.plot(X,E_DFT(X)-E_DFT(Rc),'--',label='DFT')
    plt.plot(X,E_el(X)-E_el(Rc),label='DFTB')
    plt.legend()
    plt.savefig('Repulsion1.png')
    plt.clf()
    a=np.sqrt(3.0)/2                                                # rcut*a : rayon de coupure de la fonction de paire
    x=[] ; fx=[]
    def frep(n,frep,rho_0):                                         # Erep for 1st and 2nd neighbout
        return 1/8*(fsp(a**n*rho_0)-6*frep)
    for rho_0 in np.linspace(a*Rc,Rc-1e-10,50):
        # Initialize
        if a*rho_0 > Rcmin:
            x.append(a*rho_0) ; fx.append(1/8*fsp(rho_0))
        for n in np.arange(1,10,1):
            if a**(n+1)*rho_0 > x1:
                x.append(rho_0*a**(n+1))
                fx.append(frep(n,fx[-1],rho_0))
            else :
                break
    plt.clf()
    plt.plot(x,fx,'-o',color='C3')
    plt.show()
    x_EOS=np.array(sorted(x))                           
    fx_EOS=np.array(sorted(fx,reverse=True))             
    x=np.array(sorted(x))/0.5291777249                              # Bohr
    fx=np.array(sorted(fx,reverse=True))/13.605698066               # Ry
    NPoints=120
    sp1=UnivariateSpline(x,fx,k=4,s=0) ; X=np.linspace(x[0],x[-1],NPoints) ; interval=(x[-1]-x[0])/(NPoints-1)
    sp2=UnivariateSpline(X,sp1(X),k=4,s=0)                          # To obtain the derivatives...
    # Reconstruction EOS
    for val in np.linspace(np.sqrt(3)/2*Rc,Rc,20):
        x_EOS=np.append(x_EOS,val)
        fx_EOS=np.append(fx_EOS,0)
    spx_eos=UnivariateSpline(x_EOS,fx_EOS,k=5,s=0)
    # 2 - Repulsive equation for x < x1 (exp(-a1 x + a2)+a3)
    def func(x,alpha,beta,gamma):
        return np.exp(-alpha*x+beta)+gamma
    coeff=(sp2(X[1])-sp2(X[0]))/(X[1]-X[0])
    res,cov=curve_fit(func,X[0:5],sp2(X[0:5]),[1/X[0],X[0],-X[0]/sp2(X[0])],maxfev=1000000)
    # 3 - Writing at the good format
    Repulsion=""
    Repulsion=Repulsion+("{} {}\n" .format(len(X),X[-1]))
    Repulsion=Repulsion+"{} {} {}\n".format(*res)
    for i,vec in enumerate(X):
        if i < len(X)-1:
            Repulsion=Repulsion+"{0:.8f} {1:.8f} {2:.10f} {3:.10f} {4:.10f} {5:.10f}\n".format(vec,vec+interval,sp2.derivatives(vec)[0],sp2.derivatives(vec)[1],sp2.derivatives(vec)[2]/2,sp2.derivatives(vec)[3]/6)
        else :
            Repulsion=Repulsion+"{0:.8f} {1:.8f} {2:.10f} {3:.10f} {4:.10f} {5:.10f} {6} {7} \n".format(vec,vec,sp2.derivatives(vec)[0],sp2.derivatives(vec)[1],sp2.derivatives(vec)[2]/2,sp2.derivatives(vec)[3]/6,0,0)
    # Final verification
    plt.clf()
    plt.plot(X,sp2(X),'o')
    Xe=np.linspace(X[0]-0.2,X[0]+0.2,10)
    plt.plot(Xe,func(Xe,*res))
    plt.savefig('Erep_skf.png')
    plt.clf()
    return(Repulsion)

### Optimisation Cij et mu
def opt_Cij_mu(params,cnt,calculate=True):
    # Range repulsive
    x1=2.6 ; x2=3.1
    # Params
    r1,r2,r3,r4=params
    print("parameters tested : {} {} {} {}".format(r1,r2,r3,r4))
    with open("lastparams.dat","w") as myfile:
        myfile.write("{} {} {} {}".format(r1,r2,r3,r4))
    RM("charges.dat")
    RM("charges.bin")
    spmatrix=create_sp(r4)        # Spin coupling matrix
    print(spmatrix) 
    with open('spin_constants.dat','w') as myfile:
        for lines in spmatrix:
            myfile.write("{0:.4f} {1:.4f} {2:.4f}\n".format(*lines))
    # Calcul m0 
    poscar=Reset()
    Efile="E_{}_{}_{}_{}.dat".format(r1,r2,r3,r4)
    os.system("rm relax_bulk.traj")
    if calculate :
        skf_file=create_skf(r1,r2,r3) # Skc file
        E_DFTB=[]
        Range=np.linspace(x1,rcut,50)
        print("looking for repulsive potential")
        for A in Range:
            poscar.a0=A
            poscar.write("POSCAR")
            bulk=read("POSCAR")
            calc=dftb_mag_off(skf_file)
            bulk.set_calculator(calc) # Attach the calculator to the geometry
            relax=BFGS(bulk,trajectory='relax_bulk.traj')                  
            try:
                relax.run(fmax=0.0001)
            except UnboundLocalError:
                os.system('rm charges.dat')
                calc=dftb_mag_off(skf_file,TMP=2500)
                relax.run(fmax=0.0001)
            # Electronic energy
            E_DFTB.append(dftb.findenergy(mod='all')[1])
        # save
        with open(Efile,'w') as myfile:
            for i in range(len(Range)):
                myfile.write("{} {}\n".format(Range[i],E_DFTB[i]))
    else:
        Range=np.loadtxt(Efile)[:,0]
        E_DFTB=np.loadtxt(Efile)[:,1]   # Electronic energy
    E_el=UnivariateSpline(Range,E_DFTB,s=0,k=4)
    Repulsion=Erep(E_el,E_DFT,x1,x2)
    new_skf_file=add_rep(Repulsion) # cest bon on a un skf avec le potentiel repulsif
    
    # calcul des constantes elastiques

    # Calculation of the lattice parameter
    RM("charges.dat")
    poscar=Reset(a0=2.83) ; poscar.write("POSCAR") ; bulk=read("POSCAR")
    calc=dftb_mag_on_opt(new_skf_file,A=spmatrix) ; bulk.set_calculator(calc) # Attach the calculator to the geometry
    relax=BFGS(bulk,trajectory='a0_opt.traj') ; relax.run(fmax=0.0001)
    opt=read("geo_end.gen") ; a0=abs(opt.cell[0][0]*2) # a0
    print("lattice parameter is {}".format(a0))

    # Now m0, C11, C12, C44
    E=np.zeros(4)
    stress=np.zeros((4,3,3))
    mag=[] ; val_a0=[]  

    # 2/ On calcule C44 C11 C12
    eps=5e-3
    EPS_C44=np.zeros((3,3)) ; EPS_C44[1][2]=eps ; EPS_C44[2][1]=eps
    EPS_C11=np.zeros((3,3)) ; EPS_C11[0][0]=eps ; EPS_C11[1][1]=-eps
    EPS_C12=np.zeros((3,3)) ; EPS_C12[0][0]=eps ; EPS_C12[1][1]=eps
    e_range=[np.zeros((3,3)),EPS_C44,EPS_C11,EPS_C12]

    for i,DEF in enumerate(e_range):
        poscar=Reset(a0=a0)
        poscar.shear(DEF)
        poscar.write("POSCAR")
        bulk=read("POSCAR")
        calc=dftb_mag_on(new_skf_file,A=spmatrix)
        bulk.set_calculator(calc) # Attach the calculator to the geometry
        relax=BFGS(bulk,trajectory='cij.traj') ; relax.run(fmax=0.0001)
        E[i]=dftb.findenergy()
        stress[i]=dftb.findstress()
        if i == 0:
            mag.append(dftb.findmag()) ; val_a0.append(a0)
        else:
            stress[i]=stress[i]-stress[0]
    C44=stress[1][1][2]/2/eps
    C11=(stress[2][0][0]+stress[3][0][0])/2/eps
    C12=(stress[2][1][1]+stress[3][0][0])/2/eps
    print("mu_0={} C11={} C12={} C44={}".format(mag[0],C11,C12,C44))

    # 3/ On calcule deux points de magnetisation
    Range=[2.75,2.9]
    RM('rm charges.dat')
    if calculate :
        for A in Range:
            poscar=Reset(a0=A)
            poscar.write("POSCAR")
            bulk=read("POSCAR")
            calc=dftb_mag_on(new_skf_file,A=spmatrix)
            bulk.set_calculator(calc) # Attach the calculator to the geometry
            relax=BFGS(bulk,trajectory='relax_bulk.traj')                  
            try:
                relax.run(fmax=0.0001)
            except UnboundLocalError:
                RM("charges.dat")
                calc=dftb_mag_off(skf_file,TMP=2500)
                relax.run(fmax=0.0001)
            # Electronic energy
            mag.append(dftb.findmag())
            val_a0.append(A)
        # save
        with open("calc.dat",'w') as myfile:
            myfile.write("# C44={0:.2f} C11={1:.2f} C12={2:.2f} \n".format(C44,C11,C12))
            for i in range(len(val_a0)):
                myfile.write("{} {}\n".format(val_a0[i],mag[i]))
    else :
        val_a0=np.loadtxt('calc.dat')[:,0]
        mag=np.loadtxt('calc.dat')[:,1]
    
    # valeurs ref DFT et DFTB et erreur
    def Error(val_a,val_b):
        Poids=[1,1,2,2,2,2]
        error=0
        for o in range(len(val_a)) :
            error+=Poids[o]*((val_a[o]-val_b[o])/val_a[o])**2
        error=np.sqrt(error/len(val_a))
        return error

    # magmom
    v_DFT=[] ; v_DFTB=[]
    for i,A in enumerate(val_a0):
        v_DFT.append(mu_DFT(A))
        v_DFTB.append(mag[i])
    v_DFT.append(C11_DFT) ; v_DFT.append(C12_DFT) ; v_DFT.append(C44_DFT)
    v_DFTB.append(C11)     ; v_DFTB.append(C12)     ; v_DFTB.append(C44)
    Error=Error(v_DFT,v_DFTB)
    cnt.k+=1
    with open("params.dat","a") as myfile:
        myfile.write("{} {} {} {} {} {}\n".format(cnt.k,r1,r2,r3,r4,Error))
    with open("constantes.dat","a") as myfile:
        myfile.write("{0} {1:.2f} {2:.2f} {3:.2f} {4:.2f} {5:.3f} {6:.3f} {7:.3f}\n".format(cnt.k,a0,C11,C12,C44,mag[0],mag[1],mag[2]))
    return Error

# Calcul 1
os.system("echo \"# step r1 r2 r3 r4 Error \" > params.dat") 
os.system("echo \"# C11 C12 C44 mu1 mu2 mu3\" > constantes.dat") 
os.system("echo \"# DFT : {} {} {} {} {} {} {}\" >> constantes.dat".format(a0_DFT,C11_DFT,C12_DFT,C44_DFT,mu_DFT(2.83),mu_DFT(2.71),mu_DFT(2.99))) 

class step :
    k=0
cnt=step()

# Escape module for freezed calculation
def handler(signum,frame):
    raise Exception('Action took too much time')
signal.signal(signal.SIGALRM,handler)
TIMELIMIT=20*60
signal.alarm(TIMELIMIT)
os.system("echo \"# Journal of error rc1 rc2 rc3 rc4\" > errors.dat")

# General view
for a in np.linspace(7,16,100):
    for b in np.linspace(2.5,4,100):
        for c in np.linspace(2,5,100):
            for d in np.linspace(2,5,100):
                try:
                    opt_Cij_mu([a,b,c,d],cnt,calculate=True)
                except:
                    print("Too long")
                    os.system("echo {} {} {} {} >> errors.dat".format(a,b,c,d))

# Optimisation from a starting point
a=12 ; b=3.2 ; c=2.8 ; d=2.8
result=basinhopping(opt_Cij_mu,T=2,x0=[a,b,c,d],minimizer_kwargs={'args':cnt,'options':{'eps':5e-02}})

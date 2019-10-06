import matplotlib.pyplot as plt
from pylab import cm

# Article plot
plt.rcParams['text.usetex'] = True
plt.figure(figsize=(8,6))
plt.style.use('seaborn-paper')
params = {'font.size':16 , 'legend.fontsize': 14,'xtick.labelsize' : 14,'ytick.labelsize' : 14,'axes.labelsize' : 16}
plt.rcParams.update(params)

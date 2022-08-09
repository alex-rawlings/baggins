import numpy as np
import matplotlib.pyplot as plt
import pygad
from scipy.optimize import curve_fit

def galaxy_center(snap):
    return pygad.analysis.shrinking_sphere(snap, 
            pygad.analysis.center_of_mass(snap.stars),
            pygad.UnitQty(30, 'kpc'))

snap = pygad.Snapshot('/scratch/pjohanss/ikostamo/test/ketju_no_bh/output/snap_009.hdf5')
snap.to_physical_units()
time = pygad.UnitQty(snap.time, f"({snap.gadget_units['LENGTH']})/({snap.gadget_units['VELOCITY']})").in_units_of('Gyr', subs=snap)
print(f'time: {time}')

fig, ax, r, prof = pygad.plotting.profile(snap.stars,100,'mass',av=None, units=None, dens=True, proj=2, center=galaxy_center(snap),
            N=50, logbin=True, minlog=0.01, logscale=True, ylabel=None,
            labelsize=14, ignoreZeros=False, ax=None, label=f'KETJU output, t = {np.round(time,1)} Gyr')

def sersic_function(R, I_e, R_e,n):
    b=1.9992*n-0.3271
    return I_e * np.exp(-b*((R/R_e)**(1/n)-1))

initialguess = [1e9,10,4]
popt, pcov = curve_fit(lambda *args: np.log10(sersic_function(*args)), r, np.log10(prof), initialguess, bounds=([0,0,0], [np.inf, 15,10]))
plt.plot(r, sersic_function(r, *popt), label =f'Sérsic fit (R_e = {round(popt[1], 2)} kpc, n = {round(popt[2], 2)})', color='orange', linestyle='-')
plt.legend()
print(popt)
plt.show()
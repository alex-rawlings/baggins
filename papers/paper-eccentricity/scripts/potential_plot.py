import numpy as np
import matplotlib.pyplot as plt
import pickle

if True:
    fname = '../data/e90_merger_potential.pkl'
    e_spheroid_all = [0.2, 0.2, 0.2, 0.7, 0.2, 0.9, 0.2, 0.2, 0.2, 0.2]
    phi_all = [0, 0, 0, 60, 0, 40, 0, 0, 0, 0]
    figname = "e90"
else:
    fname = '../data/e99_merger_potential.pkl'
    e_spheroid_all = [0.2, 0.995, 0.8, 0.995, 0.7, 0.2, 0.8, 0.9, 0.2, 0.9]
    phi_all = [-55, -52, -55, -55, -65, -55, -55, -55, -55, -60]
    figname = "e99"


with open(fname, 'rb') as f:
    data = pickle.load(f)

X, Y = data['X'], data['Y']

fig, ax = plt.subplots(2 ,5, sharex="all", sharey="all", figsize=(12,5))

for i, axi in enumerate(ax.flatten()):
    print(f"Plot for {i}")
    axi.set_aspect('equal')
    pot = data['pots'][i]
    min_pot = max([np.min(pot), -7e9])
    t = data['times'][i]*1e3

    if np.any(np.isnan(pot)): continue

    print(f"{np.min(pot):.2e}", f"{np.max(pot):.2e}")
    e_spheroid = e_spheroid_all[i]
    phi = np.radians(phi_all[i])
    axi.set_title(f"{t:.1f} Myr (e_s:{e_spheroid:.4f})", fontsize="small")
    axi.contour((np.cos(phi)*X - np.sin(phi)*Y), (np.cos(phi)*Y + np.sin(phi)*X), pot,
                colors='tab:blue', levels=min_pot+np.linspace(2e9,8e9,20), 
                linestyles='-')

    e2s = e_spheroid**2
    A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
    A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)

    axi.contour(X,Y, A3*X**2 + A1*Y**2, colors='tab:orange', levels=np.linspace(0,1,10), linestyles='--')
    axi.set_xlabel('x/kpc')
    axi.set_ylabel('y/kpc')
plt.suptitle(f"Run {figname}")
plt.savefig(f"{figname}.pdf")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('data/hernquist_merger_potential.pkl', 'rb') as f:
    data = pickle.load(f)

X, Y = data['X'], data['Y']
pot = data['pots'][8]

print(np.min(pot), np.max(pot))
phi = np.radians(-35)
plt.contour((np.cos(phi)*X - np.sin(phi)*Y), (np.cos(phi)*Y + np.sin(phi)*X), pot,
            colors='tab:blue', levels=np.min(pot)+np.linspace(1.4e10,4e10,20), linestyles='-')

e_spheroid = 0.8
e2s = e_spheroid**2
A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)

plt.contour(X,Y, A3*X**2 + A1*Y**2, colors='tab:orange', levels=np.linspace(0,1,10), linestyles='--')
plt.gca().set_aspect('equal')
plt.xlabel('x/kpc')
plt.ylabel('y/kpc')

plt.show()

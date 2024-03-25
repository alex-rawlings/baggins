import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs


tests = [
    (0.002, 3.5e-3, "o"),
    (0.002, 20e-3, "o"),
    (0.002, 35e-3, "o"),
    (0.01, 3.5e-3, "x"), 
    (0.02, 3.5e-3, "x"),
    (0.03, 3.5e-3, "x"),
    (0.05, 3.5e-3, "x")
]

eta = np.geomspace(0.0005, 0.2, 100)
epsilon = np.geomspace(0.1, 200, 101)*1e-3
ETA, EPSILON = np.meshgrid(eta, epsilon)
delta_t = np.log10(np.sqrt(2 * EPSILON * ETA))


c = plt.contourf(ETA, EPSILON, delta_t, 50)
plt.axvline(0.025, c="tab:orange")
for params in tests:
    plt.scatter(params[0], params[1], c="tab:red", marker=params[2])
cbar = plt.colorbar(c)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"ErrTolIntAcc ($\eta$)")
plt.ylabel("Softening Length [kpc]")
cbar.ax.set_ylabel(r"$\log(\Delta t \cdot \sqrt{|a|})$")
plt.show()
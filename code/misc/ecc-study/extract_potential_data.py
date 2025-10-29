import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pygad
from scipy.interpolate import griddata
import baggins as bgs

maindir = "/scratch/pjohanss/madleevi/emm/e-090/1M/D_1M_a-D_1M_b-3.720-0.279/output"
#maindir = "/scratch/pjohanss/madleevi/emm/e-099/1M/D_1M_a-D_1M_b-3.720-0.028/output"
#maindir = "/scratch/pjohanss/madleevi/emm/e-040/1M/D_1M_a-D_1M_b-3.720-1.674/output"
#maindir = "/scratch/pjohanss/madleevi/emm/e-070/1M/D_1M_a-D_1M_b-3.720-0.837/output"
extent = 1
snap_num = 3
oblate = True
X,Y = np.meshgrid(*2*[np.linspace(-extent, extent, 400)])


snap = pygad.Snapshot(os.path.join(maindir, f"snap_{snap_num:03d}.hdf5"), physical=True)
pygad.Translation(-pygad.analysis.center_of_mass(snap.bh)).apply(snap)

# determine influence radius
rinfl = bgs.analysis.influence_radius(snap)
print(f"Influence radii: {rinfl}")

L = np.cross(np.diff(snap.bh['pos'],axis=0),np.diff(snap.bh['vel'],axis=0))
Lhat = L/np.linalg.norm(L)

stars = snap.stars[abs(np.dot(snap.stars['pos'],Lhat[0]))<1e-2]
print(f"There are {len(stars)} stars")
pot = np.array(griddata(stars['pos'][:,[0,2]], stars['Epot'], (X,Y)))

R = bgs.mathematics.radial_separation(np.stack((X.flatten(), Y.flatten()), axis=-1))
plt.loglog(
    R,
    -pot.flatten(),
    marker=".", ls="", markersize=1
)
mask = R < R[np.argmin(pot.flatten())]
med_inner_pot = np.nanmedian(pot.flatten()[mask])
plt.axhline(-med_inner_pot, c="tab:red")
plt.axhline(-np.interp(np.diff(snap.bh["pos"][:,[0,2]])[0], stars["r"], stars["Epot"]), c="tab:orange")
plt.xlabel("r/pc")
plt.ylabel("potential")
plt.savefig("pot_1d.png", dpi=300)
plt.close()

levels = np.nanquantile(pot, [1e-3, 0.005, 0.01, 0.02, 0.1])
levels = np.append(levels, med_inner_pot)
print(f"Median potential plot will be axis {np.argsort(levels)[-1]}")
levels = np.sort(levels)

p = plt.contour(X, Y, pot, levels=levels, colors="k", linestyles="-", lw=0.5)
xc, yc = bgs.plotting.extract_contours_from_plot(p=p)
plt.close()

fig, ax = plt.subplots(2, 3, sharex="all", sharey="all")

colours = ["k"] * len(levels)
all_espheroids = []
all_semimajors = []
all_phis = []

for i, (axi, x, y) in enumerate(zip(ax.flat, xc, yc)):
    if i>=len(levels): break
    axi.set_aspect("equal")
    # set influence radii
    for bhid, _rinfl in rinfl.items():
        idmask = pygad.IDMask(bhid)
        circ = Circle(
            snap.bh[idmask]["pos"][:, [0,2]].view(np.ndarray).flatten(),
            float(_rinfl.view(np.ndarray)),
            ls=":", ec="tab:orange", fc="none", lw=1,
            zorder=2
        )
        axi.add_patch(circ)

    ellip, a, b, phi = bgs.mathematics.fit_ellipse(x, y)
    e_spheroid = bgs.mathematics.eccentricity(a, b)
    all_espheroids.append(e_spheroid)
    all_semimajors.append(a)
    all_phis.append(phi)
    print(f"Eccentricity is {e_spheroid:.3f}")
    axi.contourf(X, Y, pot, levels=50)
    colours[i] = "blue"
    axi.contour(X, Y, pot, levels=levels, colors=colours, linestyles="-", linewidths=0.5)
    axi.plot(*ellip, c="green")
    colours[i] = "k"
    e2s = e_spheroid**2
    if oblate:
        A3 = np.sqrt(1-e2s)/e2s * (np.arcsin(e_spheroid) / e_spheroid - np.sqrt(1 - e2s))
        A1 = 2 * np.sqrt(1-e2s)/e2s * (1/np.sqrt(1 - e2s) - np.arcsin(e_spheroid) / e_spheroid)
    else:
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
        A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)
    print(f"Rotate angle: {np.degrees(phi):.3f}")
    XA = np.cos(phi)*X - np.sin(phi)*Y
    YA = np.cos(phi)*Y + np.sin(phi)*X
    axi.contour(XA, YA, A3*X**2 + A1*Y**2, levels=np.geomspace(1e-4, 1, 5), linestyles='-', colors="red", linewidths=0.5)
    axi.set_title(f"es = {e_spheroid:.3f}")

plt.savefig("pot.png", dpi=500)
plt.close()

es2_func = lambda r: np.interp(r, all_semimajors, all_espheroids)**2
A3 = lambda r: np.sqrt(1-es2_func(r))/es2_func(r) * (np.arcsin(np.sqrt(es2_func(r))) / np.sqrt(es2_func(r)) - np.sqrt(1 - es2_func(r)))
A1 = lambda r: 2 * np.sqrt(1-es2_func(r))/es2_func(r) * (1/np.sqrt(1 - es2_func(r)) - np.arcsin(np.sqrt(es2_func(r))) / np.sqrt(es2_func(r)))

phi_func = lambda r: np.interp(r, all_semimajors, all_phis)
XA = lambda r: np.cos(phi_func(r))*X - np.sin(phi_func(r))*Y
YA = lambda r: np.cos(phi_func(r))*Y + np.sin(phi_func(r))*X

R = R.reshape(X.shape)
pot_vary = A3(R)*X**2 + A1(R)*Y**2
levels = np.nanquantile(pot_vary, np.geomspace(1e-4, 0.5, 20))
levels.sort()
plt.contourf(XA(R), YA(R), pot_vary, levels=levels)
plt.contour(XA(R), YA(R), pot_vary, levels=levels, linewidths=0.5, colors="w", linestyles="-")
plt.contour(X, Y, pot, levels=20, colors="k", linestyles="-", linewidths=0.5)
plt.savefig("pot_vary_with_r.png", dpi=300)

import numpy as np
import pygad
import matplotlib.pyplot as plt

pidx = "009"
snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/A-C-3.0-0.05/perturbations/{pidx}/output/AC_perturb_{pidx}_080.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

#snap["pos"] -= pygad.analysis.center_of_mass(snap.stars)
#snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.stars, "vel")

if False:
    # Virial theorem
    K = 0.5 * np.sum(snap["mass"]*(np.sum(snap["vel"], axis=-1))**2)
    W = np.sum(snap["mass"]*snap["pot"])
    print(K, W)
    print(2*K+W)

    # Dynamical timescale at virial radius
    rvir, mvir = pygad.analysis.virial_info(snap, pygad.analysis.center_of_mass(snap.stars))
    print(rvir, mvir)
    tdyn = np.sqrt(rvir**3 / (pygad.physics.G * np.sum(snap["mass"])))
    print(tdyn.in_units_of("Gyr"))

# my own virial radius calculation
# sim was run with h_0=1 --> be careful with critical density
critdens = (3 * pygad.UnitScalar(100, units="km/s/Mpc")**2 / (8 * np.pi * pygad.physics.G)).in_units_of("Msol/kpc**3")
print(critdens)

def virialRad(snap, critdens, overdens=200):
    c1 = pygad.analysis.center_of_mass(snap.stars)
    c2 = pygad.analysis.shrinking_sphere(snap.stars, c1, 100)
    #print(c1)
    #print(c2)
    #snap["pos"] -= c1
    snap["pos"] -= c1
    RhoVir = overdens * critdens
    idx = np.argsort(snap["r"])
    m = snap["mass"][idx]
    r = snap["r"][idx]
    if True:
        rho = np.cumsum(m) / (4.0/3.0*np.pi * r**3)
        print(rho.units)
        RhoVir = pygad.UnitScalar(RhoVir, units=rho.units)
        viridx = np.argmax(rho<RhoVir)
        plt.loglog(snap["r"][idx], rho)
        plt.axhline(RhoVir, c="tab:red")
        return r[viridx]
    else:
        M = 0
        for mi, ri in zip(m, r):
            M += mi
            p = M / (4./3. * np.pi * ri**3)
            if p < RhoVir: return ri

print("z: {}".format(snap.redshift))
print("H: {}".format(snap.cosmology))
rvir = virialRad(snap, critdens)
print(rvir)

rvir2, mvir = pygad.analysis.virial_info(snap, pygad.analysis.center_of_mass(snap.stars))
print(rvir2)
plt.show()
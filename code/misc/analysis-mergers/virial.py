import numpy as np
import pygad
import matplotlib.pyplot as plt
import cm_functions as cmf

centering = True
pidx = "002"
snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/A-D-3.0-1.0/perturbations/{pidx}/output/AD_perturb_{pidx}_050.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)


c1 = pygad.analysis.center_of_mass(snap.stars)
if centering:
    c2 = pygad.analysis.shrinking_sphere(snap.stars, c1, 20)

    min_pot_idx = np.argmin(snap["pot"])
    c3 = snap["pos"][min_pot_idx,:]

    print(c1)
    print(c2)
    print(c3)
    print(snap["pos"][0,:])
    snap["pos"] -= c3
    print(snap["pos"][0,:])
else:
    snap["pos"] -= c1
mask_30_kpc = pygad.BallMask(pygad.UnitScalar(30, "kpc"))
snap["vel"] -= pygad.analysis.mass_weighted_mean(snap.stars[mask_30_kpc], "vel")

if True:
    # Virial theorem
    #K = 0.5 * np.sum(snap["mass"]*(np.sum(snap["vel"]**2, axis=-1)))
    v2 = pygad.UnitArr(cmf.mathematics.radial_separation(snap["vel"]), units=snap["vel"].units)**2
    K = 0.5 * np.sum(snap["mass"] * v2, axis=-1)
    W = np.sum(snap["mass"]*snap["pot"])
    #print(K, W)
    #print(2*K+W)
    print(2*K/W)

    # Dynamical timescale at virial radius
    rvir, mvir = pygad.analysis.virial_info(snap)
    print(rvir, mvir)

    if centering:
        ball_mask = pygad.BallMask(rvir)
        c4 = pygad.analysis.center_of_mass(snap[ball_mask])
        print(np.linalg.norm(c3-c4)/rvir)
    
    if False:
        tdyn = np.sqrt(rvir**3 / (pygad.physics.G * np.sum(snap["mass"])))
        print(tdyn.in_units_of("Gyr"))

        cmf.plotting.plot_galaxies_with_pygad(snap)
        plt.show()


if False:
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
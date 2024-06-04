## For calculating the (velocity) anisotropy parameter beta as a funtion of radius
import numpy as np
import matplotlib.pyplot as plt
import pygad
import gc
import time
# import figure_config
from matplotlib import cycler, rcParams
rcParams["axes.prop_cycle"] = cycler(ls=["-", ":", "--", "-."]) * rcParams["axes.prop_cycle"]

start_time = time.time()


## Return polar angle given xyz position
def theta(x, y, z):
    return np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))


## Return azimuthal angle given xy position
def phi(x, y):
    return np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))


## Return spherical velocities (v_r, v_theta, v_phi) for a given (sub)snapshot
def spherical_velocity(snapshot):
    x, y, z = (
        snapshot["pos"][:, 0],
        snapshot["pos"][:, 1],
        snapshot["pos"][:, 2],
    )
    vx, vy, vz = (
        snapshot["vel"][:, 0],
        snapshot["vel"][:, 1],
        snapshot["vel"][:, 2],
    )
    t, p = theta(x, y, z), phi(x, y) ## theta, phi arrays
    v_r = np.sin(t) * np.cos(p) * vx + np.sin(t) * np.sin(p) * vy + np.cos(t) * vz ## v_r
    v_t = np.cos(t) * np.cos(p) * vx + np.cos(t) * np.sin(p) * vy - np.sin(t) * vz ## v_theta
    v_p = -np.sin(p) * vx + np.cos(p) * vy ## v_phi
    return v_r, v_t, v_p


## Computes anisotropy parameter beta for a given (sub)snapshot
def beta(snapshot):
    v_r, v_t, v_p = spherical_velocity(snapshot)
    if np.std(v_r) == 0:
        print("ERROR: divide by zero in beta()")
    if len(v_r) == 0 or len(v_t) == 0 or len(v_p) == 0:
        print("ERROR: empty velocity array in beta()")
    return 1 - (np.std(v_t) ** 2 + np.std(v_p) ** 2) / (2 * np.std(v_r) ** 2)


## Computes anisotropy parameter beta profile for a given (sub)snapshot
## Takes minimum and maximum radii for the logarithmic radial bins in [kpc], the number of bins and method for binning ("log" or "linear")
def beta_profile(snapshot, r_min=0.1, r_max=30, bins=50, bin_type="log"):
    if bin_type == "log":
        r_arr = np.geomspace(start=r_min, stop=r_max, num=bins)
    elif bin_type == "linear":
        r_arr = np.linspace(start=r_min, stop=r_max, num=bins)
    else:
        print("ERROR: incorrect bin_type in beta_profile()")
        exit()
    b_arr = []
    bin_counts = []
    for i in range(len(r_arr) - 1):
        r1, r2 = r_arr[i], r_arr[i + 1] ## bin edges
        m1, m2 = pygad.ExprMask("r >= " + str(r1)), pygad.ExprMask("r < " + str(r2)) ## radius masks for the edges, exclude outer edge
        s = snapshot[m1 & m2] ## take particles within the radial bin
        if min(s["r"]) < r1 or max(s["r"]) >= r2:
            print("ERROR: radius out of bounds")
        b_arr.append(beta(s))
        bin_counts.append(len(s["r"]))
    ## return bin center points and corresponding beta values
    return (
        np.array([(r_arr[i] + r_arr[i + 1]) / 2 for i in range(len(r_arr) - 1)]),
        np.array(b_arr),
        np.array(bin_counts),
    )


## Initialize figure
fig, ax = plt.subplots(figsize=(3, 3), tight_layout=True)
ax.set_xlabel("r [kpc]")
ax.set_ylabel(r"$\beta$")

# dir = "/home/mcmatter/Desktop/TER/alex_paper/snap_048.hdf5"
dir = (
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/"
)
vels = ["0000", "0060", "0120", "0180", "0240", "0300", "0360", "0420", "0480", "0540", "0600", "0660", "0720", "0780", "0840", "0900", "0960", "1020", "2000"]
snaps = [2, 4, 4, 9, 9, 13, 31, 50, 70, 60, 73, 74, 100, 143, 193, 199, 240, 275, 267]

## Compute profiles
for i in range(len(vels)):
    run = "kick-vel-" + vels[i] + "/"
    print("---", run, "---")
    print()
    snap = pygad.Snapshot(dir + run + "output/snap_" + "0" * (3 - len(str(snaps[i]))) + str(snaps[i]) + ".hdf5", physical=True)

    ## Find center using a shrinking sphere on the stars
    center = pygad.analysis.shrinking_sphere(
        snap.stars,
        center=[snap.boxsize / 2, snap.boxsize / 2, snap.boxsize / 2],
        R=snap.boxsize,
    )
    ## Shift coordinates of all particles so that the snap is centered on the shrinking sphere result
    pygad.Translation(-center).apply(snap)

    radii, betas, bincount = beta_profile(snap.stars, r_min=0.1, r_max=30, bins=20, bin_type="log")
    print("min and max bin counts:", min(bincount), max(bincount))
    print()
    print("min and max beta:", min(betas), max(betas))
    print()

    del snap
    gc.collect()
    pygad.gc_full_collect()

    ax.plot(radii, betas, label=run.split("-")[-1][:-1], linewidth=1)
    #ax.scatter(radii, betas, s=bincount / max(bincount) * 50)

ax.axhline(y=0, xmin=0, xmax=1, linestyle=":", color="gray")
ax.legend(loc="lower right", ncol=4)
ax.set_xscale("log")
plt.savefig("/users/mcmatter/collisionless-merger-sample/papers/paper-corekick/figures/anisotropy_profiles_std_log.pdf", dpi=300)
print("Time taken:", time.time() - start_time)
plt.show()


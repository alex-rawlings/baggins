## For calculating the (velocity) anisotropy parameter beta as a funtion of radius
import numpy as np
import matplotlib.pyplot as plt
import pygad
import gc
import time
from matplotlib import cycler, rcParams
# import figure_config

rcParams["axes.prop_cycle"] = (
    cycler(ls=["-", ":", "--", "-."]) * rcParams["axes.prop_cycle"]
)
start_time = time.time()


## Whether to show and save the figure and save the values into a file
show_fig = 1
save_fig = 1
save_file = 1

filename = "anisotropy_profiles" ## base filename for the output figure and value file
filepath = "../figures/" ## where to save output figure and value file


## Compute polar angle given xyz position
def theta(x, y, z):
    return np.arccos(z / np.sqrt(x**2 + y**2 + z**2))


## Compute azimuthal angle given xy position
def phi(x, y):
    return np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2))


## Compute spherical velocities (v_r, v_theta, v_phi) for a given (sub)snapshot
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
    t, p = theta(x, y, z), phi(x, y)  ## theta, phi arrays
    v_r = (
        np.sin(t) * np.cos(p) * vx + np.sin(t) * np.sin(p) * vy + np.cos(t) * vz
    )  ## v_r
    v_t = (
        np.cos(t) * np.cos(p) * vx + np.cos(t) * np.sin(p) * vy - np.sin(t) * vz
    )  ## v_theta
    v_p = -np.sin(p) * vx + np.cos(p) * vy  ## v_phi
    return v_r, v_t, v_p


## Compute anisotropy parameter beta for a given (sub)snapshot
def beta(snapshot):
    v_r, v_t, v_p = spherical_velocity(snapshot)
    if np.std(v_r) == 0:
        print("ERROR: divide by zero in beta()")
    if len(v_r) == 0 or len(v_t) == 0 or len(v_p) == 0:
        print("ERROR: empty velocity array in beta()")
    return 1 - (np.std(v_t) ** 2 + np.std(v_p) ** 2) / (2 * np.std(v_r) ** 2)


## Compute anisotropy parameter beta profile for a given (sub)snapshot.
## Takes minimum and maximum radii for the radial bins in [kpc],
## the number of bins and the method for binning ("log", "linear", "hybdrid" or "r_core_split").
## r_switch sets the radius in [kpc] at which "hybdrid" binning is switched from linear to log.
def beta_profile(snapshot, r_min=0.1, r_max=30, r_switch=1, bins=50, bin_type="log"):
    b_arr = []
    bin_counts = []
    if bin_type == "log":
        r_arr0 = np.geomspace(start=r_min, stop=r_max, num=bins)
    elif bin_type == "linear":
        r_arr0 = np.linspace(start=r_min, stop=r_max, num=bins)
    elif bin_type == "hybdrid":
        if bins < 5:
            print("ERROR: please use >= 5 'hybdrid' bins")
            exit()
        a1 = np.linspace(start=r_min, stop=r_switch, num=5)
        a2 = np.geomspace(start=r_switch, stop=r_max, num=bins-5)
        r_arr0 = np.concatenate((a1, a2[1:]), axis=0)  ## include r_switch only once
    elif bin_type != "r_core_split":
        print("ERROR: incorrect bin_type in beta_profile()")
        exit()

    ## log, linear or hybdrid binning
    if bin_type != "r_core_split":
        for i in range(len(r_arr0) - 1):
            r1, r2 = r_arr0[i], r_arr0[i + 1]  ## bin edges
            m1, m2 = pygad.ExprMask("r >= " + str(r1)), pygad.ExprMask(
                "r < " + str(r2)
            )  ## radius masks for the edges, exclude outer edge
            s = snapshot[m1 & m2]  ## take particles within the radial bin
            if min(s["r"]) < r1 or max(s["r"]) >= r2:
                print("ERROR: radius out of bounds")
            b_arr.append(beta(s))
            bin_counts.append(len(s["r"]))
        r_arr = np.array(
            [(r_arr0[i] + r_arr0[i + 1]) / 2 for i in range(len(r_arr0) - 1)]
        ) ## compute bin center points

    ## r_core_split binning (2 bins, r <= r_max = r_core and r > r_max = r_core)
    else:
        m1, m2 = pygad.ExprMask("r <= " + str(r_max)), pygad.ExprMask(
            "r > " + str(r_max)
        )
        for m in [m1, m2]:
            b_arr.append(beta(snapshot[m]))
            bin_counts.append(len(snapshot[m]["r"]))
        r_arr = np.array([np.mean(snapshot[m1]["r"]), np.mean(snapshot[m2]["r"])])

    ## return bin center points and corresponding beta values and bin counts
    return (
        r_arr,
        np.array(b_arr),
        np.array(bin_counts),
    )


## Source directory and the snapshots to plot
dir = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/"
vels = [
    "0000",
    "0060",
    "0120",
    "0180",
    "0240",
    "0300",
    "0360",
    "0420",
    "0480",
    "0540",
    "0600",
    "0660",
    "0720",
    "0780",
    "0840",
    "0900",
    "0960",
    "1020",
]
snaps = [2, 4, 4, 9, 9, 13, 31, 50, 70, 60, 73, 74, 100, 143, 193, 199, 240, 275]
coreradii = [
    0.5225785000000001,
    0.5642145000000001,
    0.65247,
    0.66808,
    0.7299105,
    0.96869,
    0.927019,
    1.040925,
    1.0412949999999999,
    1.191115,
    1.12596,
    1.19127,
    1.270385,
    1.32713,
    1.30975,
    1.3308550000000001,
    1.246245,
    1.3778,
]


## Initialize figure
fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
# ax.set_xlabel("r [kpc]")
ax.set_xlabel("r/r$_b$")
ax.set_ylabel(r"$\beta$")


## Compute and store profiles
if save_file == 1:
    file = open(filepath + filename + ".txt", "w")
for i in range(len(vels)):
    run = "kick-vel-" + vels[i] + "/"
    print("---", run, "---")
    print()
    snap = pygad.Snapshot(
        dir
        + run
        + "output/snap_"
        + "0" * (3 - len(str(snaps[i])))
        + str(snaps[i])
        + ".hdf5",
        physical=True,
    )

    ## Find center using a shrinking sphere on the stars
    center = pygad.analysis.shrinking_sphere(
        snap.stars,
        center=[snap.boxsize / 2, snap.boxsize / 2, snap.boxsize / 2],
        R=snap.boxsize,
    )
    ## Shift coordinates of all particles so that the snap is centered on the shrinking sphere result
    pygad.Translation(-center).apply(snap)

    # radii, betas, bincount = beta_profile(snap.stars, r_min=0.1 * coreradii[i], r_max=30 * coreradii[i], bins=20, bin_type="log")
    # radii, betas, bincount = beta_profile(snap.stars, r_min=0.1 * coreradii[i], r_max=30 * coreradii[i], bins=200, bin_type="linear")
    # radii, betas, bincount = beta_profile(snap.stars, r_max=coreradii[i], bin_type="r_core_split")
    radii, betas, bincount = beta_profile(snap.stars, r_min=0.1 * coreradii[i], r_max=30 * coreradii[i], r_switch=1.0 * coreradii[i], bins=25, bin_type="hybdrid")

    if save_file == 1:
        file.write(" ".join([str(val/coreradii[i]) for val in radii]) + "\n") ## write radii in [r_core]
        file.write(" ".join([str(val) for val in betas]) + "\n") ## write betas
        file.write("\n")

    print("min and max bin count:", min(bincount), max(bincount))
    print("min and max beta:", min(betas), max(betas))
    print()

    del snap
    gc.collect()
    pygad.gc_full_collect()

    ax.plot(radii / coreradii[i], betas, label=run.split("-")[-1][:-1], linewidth=1)
    # ax.scatter(radii/coreradii[i], betas, s=bincount / max(bincount) * 50, marker="s", edgecolors="black")

if save_file == 1:
    file.close()
ax.axhline(y=0, xmin=0, xmax=1, linestyle=":", color="gray")
ax.legend(loc="lower right", ncol=4)
ax.set_xscale("log")
if save_fig == 1:
    plt.savefig(
        filepath
        + filename
        + ".pdf",
        dpi=300,
    )
    # plt.savefig(figure_config.fig_path( filename + ".pdf"))
print("Time taken:", time.time() - start_time)
if show_fig == 1:
    plt.show()

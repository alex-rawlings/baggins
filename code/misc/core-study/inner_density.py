import numpy as np
from scipy.stats import binned_statistic_2d
import scipy.spatial.transform
import matplotlib.pyplot as plt
import baggins as bgs
import pygad
import arviz as az
from tqdm import tqdm

bgs.plotting.check_backend()

#snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0720/output/snap_100.hdf5"
#snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0240/output/snap_009.hdf5"
snapfiles = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0060/output/snap_004.hdf5",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0240/output/snap_009.hdf5",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0720/output/snap_100.hdf5",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0960/output/snap_240.hdf5"
]

coord_labels = "xyz"
rng = np.random.default_rng(31)
plot_proj = False
num_rots = 20

if plot_proj:
    fig, ax = plt.subplots(3,len(snapfiles))
else:
    fig, ax = plt.subplots(1,1)

def ellipticity(a, b):
    return (a-b)/a

for i, (snapfile, col) in enumerate(zip(snapfiles, bgs.plotting.mplColours())):
    print(f"Doing snap {snapfile}")
    # load snapshot
    snap = pygad.Snapshot(snapfile, physical=True)
    ellipticities = []

    ball_mask = pygad.BallMask(30)

    # centre
    xcom = pygad.analysis.shrinking_sphere(
        snap.stars, pygad.analysis.center_of_mass(snap.stars[ball_mask]), 30
    )
    trans = pygad.Translation(-xcom)
    trans.apply(snap, total=True)

    # set up random rotations
    rot_axis = rng.uniform(-1, 1, (num_rots, 3))
    rot_angle = rng.uniform(0, np.pi, num_rots)

    for R in tqdm(range(num_rots), desc="Rotations"):

        # mask to an annulus
        inner_mask = pygad.BallMask(1) & ~pygad.BallMask(0.9)

        for j, (xi, yi) in enumerate(zip((0,0,1), (1,2,2))):
            # different projections for this rotation
            x = snap.stars[inner_mask]["pos"][:,xi]
            y = snap.stars[inner_mask]["pos"][:,yi]
            rand_idxs = rng.choice(np.arange(len(x)), size=min(1000, len(x)), replace=False)
            dens, xe, ye, bn = binned_statistic_2d(
                            x = x,
                            y = y, 
                            values = None,
                            statistic = "count",
                            bins = 50)
            xb = bgs.mathematics.get_histogram_bin_centres(xe)
            yb = bgs.mathematics.get_histogram_bin_centres(ye)

            # fit the ellipse
            tt = np.linspace(0, 2*np.pi, 1000)
            circle = np.stack((np.cos(tt), np.sin(tt)))
            _x = x - np.mean(x)
            _y = y - np.mean(y)
            U, S, V = np.linalg.svd(np.stack((_x[rand_idxs], _y[rand_idxs])))
            a, b = np.sqrt(2/len(_x[rand_idxs])) * S

            ellipticities.append(ellipticity(a,b))

            if plot_proj:
                transform = np.sqrt(2/len(x[rand_idxs])) * U.dot(np.diag(S))
                ellipse = transform.dot(circle)
                # plot the contour
                CS = ax[j,i].contourf(xb, yb, dens, 10)
                # plot the ellipse
                ax[j,i].plot(ellipse[0,:], ellipse[1,:], "tab:red", lw=2)

                ax[j,i].set_xlabel(f"{coord_labels[xi]}/kpc")
                ax[j,i].set_ylabel(f"{coord_labels[yi]}/kpc")
                ax[j,i].set_aspect("equal")
                ax[j,i].set_facecolor("k")

        # randomly rotate snapshot
        rot = pygad.transformation.rot_from_axis_angle(
                rot_axis[R] - xcom, rot_angle[R]
            )
        rot.apply(snap, total=True)
        _rot = scipy.spatial.transform.Rotation.from_matrix(rot.rotmat)
        xcom = _rot.apply(xcom)
        trans = pygad.Translation(-xcom)
        trans.apply(snap, total=True)

    if not plot_proj:
        az.plot_dist(ellipticities, ax=ax, color=col)
        print(f"75% quantile for ellipticity is {np.quantile(ellipticities, 0.75):.2f}")

if not plot_proj:
    bgs.plotting.savefig("ellipticity.png")

plt.show()
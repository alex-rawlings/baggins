import os.path
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    from matplotlib import use
    use("Agg")
    import matplotlib.pyplot as plt
import pygad
from tqdm import tqdm
import baggins as bgs

bgs.plotting.check_backend()

snapfiles = bgs.utils.read_parameters(
        os.path.join(
            bgs.HOME,
            "projects/collisionless-merger-sample/parameters/parameters-analysis/corekick_files.yml",
        )
    )

snapshots = dict()
for k, v in snapfiles["snap_nums"].items():
    if v is None:
        continue
    snapshots[k] = os.path.join(
        snapfiles["parent_dir"],
        f"kick-vel-{k.lstrip('v')}/output/snap_{v:03d}.hdf5",
    )

core_data_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"
data_file = "betas.pickle"

core_data = bgs.utils.load_data(core_data_file)["rb"]

r_edges_rb = np.array([1e-2, 1, 20])
Nreps = 100
rng = np.random.default_rng(42)
ball_mask = pygad.BallMask(30)

if False:
    betas = dict.fromkeys(snapshots)
    for i, (k, v) in enumerate(snapshots.items()):
        if "2000" in k:
            continue
        snap = pygad.Snapshot(v, physical=True)

        xcom = bgs.analysis.get_com_of_each_galaxy(snap, "ss", family="stars")
        vcom = bgs.analysis.get_com_velocity_of_each_galaxy(snap, xcom, family="stars")

        # move into centre of mass frame
        pygad.Translation(-list(xcom.values())[0]).apply(snap, total=True)
        pygad.Boost(-list(vcom.values())[0]).apply(snap, total=True)

        betas[k] = np.full((2,Nreps), np.nan)

        for j in tqdm(range(Nreps), desc=f"Beta for {k}"):
            rcore = rng.choice(core_data[k.lstrip("v")].flatten(), size=1)
            r_edges = r_edges_rb * rcore
            beta, Npart = bgs.analysis.velocity_anisotropy(snap.stars[ball_mask], r_edges)
            betas[k][:,j] = beta
        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
    # save data
    bgs.utils.save_data(betas, data_file)
else:
    betas = bgs.utils.load_data(data_file)

# plot
fig, ax = plt.subplots(1,1)

offset = len(betas)
for i, (k, v) in enumerate(betas.items()):
    if "2000" in k:
        continue
    kickvel = float(k.lstrip("v"))
    med, yerr = bgs.mathematics.quantiles_relative_to_median(v, axis=1)
    if i==0:
        labels = [r"$r<r_\mathrm{b}$", r"$r \geq r_\mathrm{b}$"]
    else:
        labels = ["", ""]
    for j in range(2):
        ax.errorbar(
            kickvel,
            med[j],
            yerr=np.atleast_2d(yerr[:,j]).T,
            color=bgs.plotting.mplColours()[j],
            fmt=".",
            marker = bgs.plotting.mplChars()[j],
            label = labels[j]
        )

ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{km}\,\mathrm{s}^{-1}$")
ax.set_ylabel(r"$\beta$")

ax.axhline(0, c="gray", ls=":", alpha=0.7)
ax.legend()
bgs.plotting.savefig("beta.png")
plt.show()
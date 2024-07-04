import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import baggins as bgs
import pygad


snap_nums = {"0060": 4, "0900": 199}
mainpath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick"

hist_kwargs = {
    "0060": {
        "norm": colors.Normalize(0.1, 650),
        "bins": 50
    },
    "0900": {
        "norm": colors.Normalize(0.1, 650),
        "bins": 50
    }
}
# get all the orbit files
orbitfilebases = [
    d.path
    for d in os.scandir(
        os.path.join(mainpath, "orbit_analysis")
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()

for kv, snapnum in snap_nums.items():
    print(f"Doing {kv}")
    # get the snapshot
    snapfile = bgs.utils.get_snapshots_in_dir(os.path.join(mainpath, f"kick-vel-{kv}/output"))[snapnum]
    snap = pygad.Snapshot(snapfile, physical=True)

    # recentre the snapshot same as orbit analysis
    pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
    pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

    # load orbit file
    obf = [f for f in orbitfilebases if kv in f]
    assert len(obf)==1
    orbitcl = bgs.utils.get_files_in_dir(obf[0], ext=".cl", recursive=True)[0]
    res = bgs.analysis.orbits_radial_frequency(orbitcl, returnextra=True)

    # set up figure
    fig_para, ax_para = plt.subplots(2, 4, sharex="all", sharey="all")
    fig_orth, ax_orth = plt.subplots(2, 4, sharex="all", sharey="all")

    # loop through each orbit class
    possibleclasses = np.unique(res["classids"])
    for i, (axpi, axoi) in enumerate(zip(ax_para.flat, ax_orth.flat)):
        mask = np.logical_and(res["classids"] == i, res["rad"]<5)
        id_mask = pygad.IDMask(np.array(res["pid"][mask], dtype="uint32"))
        hp, *_ = axpi.hist2d(snap[id_mask]["pos"][:,1], snap[id_mask]["pos"][:,2], **hist_kwargs[kv])
        ho, *_ = axoi.hist2d(snap[id_mask]["pos"][:,0], snap[id_mask]["pos"][:,2], **hist_kwargs[kv])
        for axi in (axpi, axoi):
            axi.set_xlim(-5, 5)
            axi.set_ylim(-5, 5)
        print(f"Para range: {np.min(hp):.3e} - {np.max(hp):.3e}")
        print(f"Orth range: {np.min(ho):.3e} - {np.max(ho):.3e}")
        print("---")
    bgs.plotting.savefig(f"slice_para_{kv}.png", fig_para)
    bgs.plotting.savefig(f"slice_orth_{kv}.png", fig_orth)
print("Showing")
plt.show()


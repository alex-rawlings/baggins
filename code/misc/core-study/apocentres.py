import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs
import arviz as az

orbitfilebases = [
    d.path
    for d in os.scandir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis"
    )
    if d.is_dir() and "kick" in d.name
]
orbitfilebases.sort()

fig, ax = plt.subplots(2,1)
cmapper, sm = bgs.plotting.create_normed_colours(0, 1020)

for obf in orbitfilebases:
    if "2000" in obf:
        continue
    print(f"Reading {obf}")
    orbitcl = bgs.utils.get_files_in_dir(obf, ext=".cl", recursive=True)[0]
    kv = float(os.path.basename(obf).replace("kick-vel-", ""))
    res = bgs.analysis.orbits_radial_frequency(orbitcl, returnextra=True)
    mask = np.logical_and(res["apo"] < np.nanquantile(res["apo"], 0.95), res["apo"]>0)
    az.plot_kde(res["apo"][mask], ax=ax[0], plot_kwargs={"lw":2, "color":cmapper(kv)})
    mask2 = np.logical_and(mask, res["meanposrad"]<4)
    az.plot_kde(res["apo"][mask2], ax=ax[1], plot_kwargs={"lw":2, "color":cmapper(kv)})
plt.colorbar(sm, ax=ax, label="kick vel")
print("Showing plot")
plt.show()
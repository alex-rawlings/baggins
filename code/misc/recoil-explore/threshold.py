import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import baggins as bgs
import pygad

bgs.plotting.check_backend()

if False:
    # theoretical observability
    rdetect = {
        "0000": -1,
        "0060": -1,
        "0120": -1,
        "0180": -1,
        "0240": -1,
        "0300": -1,
        "0360": -1,
        "0420": 4,
        "0480": 3,
        "0540": 3,
        "0600": 4,
        "0660": 4,
        "0720": 4,
        "0780": 4,
        "0840": 4,
        "0900": 4,
        "0960": 3,
        "1020": 4,
        "1080": 4,
        "1140": 4,
        "1200": 4,
        "1260": 4,
        "1320": 4,
        "1380": 3,
        "1440": 3,
        "1500": 4,
        "1560": 4,
        "1620": 5,
        "1680": 5,
        "1740": 4,
        "1800": 4
    }
    fname = "threshold_theoretical"
else:
    rdetect = {
        "0000": -1,
        "0060": -1,
        "0120": -1,
        "0180": -1,
        "0240": -1,
        "0300": 4,
        "0360": 3,
        "0420": 3,
        "0480": 4,
        "0540": 5,
        "0600": 5,
        "0660": 7,
        "0720": 8,
        "0780": 8,
        "0840": 6,
        "0900": 8,
        "0960": 8,
        "1020": 8,
        "1080": 8,
        "1140": -1,
        "1200": -1,
        "1260": -1,
        "1320": -1,
        "1380": -1,
        "1440": -1,
        "1500": -1,
        "1560": -1,
        "1620": -1,
        "1680": -1,
        "1740": -1,
        "1800": -1
    }
    fname = "threshold_obs"

fig, ax = plt.subplots()
vks = []
rs = []

for k, v in rdetect.items():
    if v < 0:
        continue
    print(k)
    snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{k}/output/snap_{v:03d}.hdf5"
    snap = pygad.Snapshot(snapfile, physical=True)
    print(f"  BH: {snap.bh['pos']}")
    bgs.analysis.basic_snapshot_centring(snap)
    print(f"  BH: {snap.bh['pos']}")

    vks.append(float(k))
    rs.append(pygad.utils.geo.dist(snap.bh["pos"][0,:]))

    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

# fit linear regression
vks = np.array(vks).flatten()
rs = np.array(rs).flatten()
# only use those that have a distance > 1
vks = vks[rs>1]
rs = rs[rs>1]
regress = scipy.stats.linregress(vks, rs)
print("Regression:")
print(f"  slope: {regress.slope:.2e}")
print(f"  intercept: {regress.intercept:.2e}")

ax.scatter(vks, rs, zorder=2, color="tab:red")
ax.plot(vks, regress.slope*vks+regress.intercept, label=f"{regress.slope:.2e}v{regress.intercept:+.2e}")
ax.set_xlabel("v/kms")
ax.set_ylabel("Threshold distance/kpc")
ax.legend()
bgs.plotting.savefig(os.path.join(bgs.FIGDIR, f"kicksurvey-study/{fname}.png"))
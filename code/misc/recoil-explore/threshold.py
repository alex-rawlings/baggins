import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import baggins as bgs
import pygad

bgs.plotting.check_backend()

rdetect = {
    "0000": -1,
    "0060": -1,
    "0120": -1,
    "0180": -1,
    "0240": -1,
    "0300": -1,
    "0360": -1,
    "0420": -1,
    "0480": 5,
    "0540": 5,
    "0600": 5,
    "0660": 5,
    "0720": 5,
    "0780": 4,
    "0840": 5,
    "0900": 4,
    "0960": 5,
    "1020": 5,
    "1080": 5,
    "1140": 5,
    "1200": 5,
    "1260": -1,
    "1320": 5,
    "1380": -1,
    "1440": -1,
    "1500": -1,
    "1560": -1,
    "1620": -1
}

fig, ax = plt.subplots()
vks = []
rs = []

for k, v in rdetect.items():
    if v < 0:
        continue
    print(k)
    snapfile = f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{k}/output/snap_{v:03d}.hdf5"
    snap = pygad.Snapshot(snapfile, physical=True)
    # move to CoM frame
    pre_ball_mask = pygad.BallMask(5)
    centre = pygad.analysis.shrinking_sphere(
        snap.stars,
        pygad.analysis.center_of_mass(snap.stars),
        30,
    )
    vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
    pygad.Translation(-centre).apply(snap, total=True)
    pygad.Boost(-vcom).apply(snap, total=True)

    vks.append(float(k))
    rs.append(pygad.utils.geo.dist(snap.bh["pos"][0,:]))

    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

# fit linear regression
vks = np.array(vks).flatten()
rs = np.array(rs).flatten()
regress = scipy.stats.linregress(vks, rs)
print("Regression:")
print(f"  slope: {regress.slope:.2e}")
print(f"  intercept: {regress.intercept:.2e}")

ax.scatter(vks, rs, zorder=2, color="tab:red")
ax.semilogy(vks, regress.slope*vks+regress.intercept)
ax.set_xlabel("v/kms")
ax.set_ylabel("Threshold distance/kpc")
bgs.plotting.savefig("threshold.png")
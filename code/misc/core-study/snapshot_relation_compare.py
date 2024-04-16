import numpy as np
import matplotlib.pyplot as plt
import pygad
import baggins as bgs


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_010.hdf5"

snap = pygad.Snapshot(snapfile, physical=True)

print(f"There is {len(snap.bh)} BHs in the snapshot")

msigma = bgs.literature.LiteratureTables.load_vdBosch_2016_data()
remgal = bgs.literature.LiteratureTables.load_sahu_2020_data()
ax1 = msigma.scatter("logsigma", "logBHMass", xerr="e_logsigma", yerr="e_logBHMass")

f = bgs.mathematics.stat_interval(x=msigma.table["logsigma"],
                                  y=msigma.table["logBHMass"],
                                  type="pred")
x = np.linspace(msigma.table["logsigma"].min(), msigma.table["logsigma"].max(), 100)
rmse, slope, intercept = bgs.mathematics.vertical_RMSE(
                            x=msigma.table["logsigma"],
                            y=msigma.table["logBHMass"],
                            return_linregress=True)

ax1.fill_between(x, y1=slope*x+intercept-f(x), y2=slope*x+intercept+f(x), alpha=0.3)



ax2 = remgal.scatter("logRe_maj_kpc", "logM*_sph", yerr="logM*_sph_ERR")

ball_mask = pygad.BallMask(7)#, center=pygad.analysis.shrinking_sphere(snap.stars, center=[0,0,0], R=30))

sigma = []
re = []
rng = np.random.default_rng()
for i in range(3):
    print(f"Doing observation {i}")
    for j in range(3):
        sigma.append(
            pygad.analysis.los_velocity_dispersion(snap.stars[ball_mask], proj=j)**2
        )
        re.append(
            pygad.analysis.half_mass_radius(snap.stars[ball_mask], proj=i)
        )
    u = rng.uniform(0,1,size=3)
    ang = rng.uniform(0, 2*np.pi)
    rot = pygad.rot_from_axis_angle(u, ang)
    rot.apply(snap, total=True)
sigma = np.sqrt(np.mean(sigma))
re = np.mean(re)


ax1.scatter(np.log10(sigma), np.log10(np.sum(snap.bh["mass"])))

ball_mask = pygad.BallMask(30, center=pygad.analysis.shrinking_sphere(snap.stars, center=[0,0,0], R=30))
ax2.scatter(np.log10(re), np.log10(np.sum(snap.stars[ball_mask]["mass"])))


plt.show()
import numpy as np
import pygad
import baggins as bgs



snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_002.hdf5"

single_mass_species = False

snap = pygad.Snapshot(snapfile, physical=True)

if single_mass_species:
    mass_ratio = snap.dm["mass"][0] / snap.bh["mass"][0]
else:
    star_num_dens = len(snap.stars)/(len(snap.stars) + len(snap.dm))
    dm_num_dens = len(snap.dm)/(len(snap.stars) + len(snap.dm))
    mass_ratio = ((star_num_dens*snap.stars["mass"][0]**2 + dm_num_dens*snap.dm["mass"][0]**2) / (star_num_dens*snap.stars["mass"][0] + dm_num_dens*snap.dm["mass"][0])) / snap.bh["mass"][0]
print(f"Mass ratio: {mass_ratio:.2e}")

mask_30 = pygad.BallMask(30)

centre = pygad.analysis.shrinking_sphere(
    snap.stars,
    pygad.analysis.center_of_mass(snap.stars[mask_30]),
    30
)
trans = pygad.Translation(-centre)
trans.apply(snap, total=True)

rinfl = list(bgs.analysis.influence_radius(snap).values())[0]
print(f"Influence radius: {rinfl:.2e}")
ball_mask = pygad.BallMask(rinfl)
fam_mask = pygad.IDMask(snap.stars["ID"]) | pygad.IDMask(snap.dm["ID"])

vel_disp_stars = np.sqrt(np.linalg.norm(np.nanvar(snap.stars[ball_mask]["vel"], axis=0)))
vel_disp_dm = np.sqrt(np.linalg.norm(np.nanvar(snap.dm[ball_mask]["vel"], axis=0)))
vel_disp = np.sqrt(np.linalg.norm([vel_disp_stars**2, vel_disp_dm**2]))
print(f"Velocity dispersion: {vel_disp:.2e}")

# Vrms = mass_ratio * vrms; where vrms = sqrt(3) * sigma
Vrms = np.sqrt(3*mass_ratio) * vel_disp
print(f"Vrms: {Vrms:.2e}")
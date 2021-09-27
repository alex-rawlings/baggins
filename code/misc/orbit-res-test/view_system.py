import matplotlib.pyplot as plt
import matplotlib.patches as mplp
import pygad
import cm_functions as cmf

snapfile = "/scratch/pjohanss/arawling/collisionless_merger/res-tests/fiducial/0-001/Af-Cf-3.0-0.001.hdf5"

snap = pygad.Snapshot(snapfile)
snap.to_physical_units()

star_id_masks = cmf.analysis.get_all_id_masks(snap, family='stars')
dm_id_masks = cmf.analysis.get_all_id_masks(snap, family='dm')
xcom = cmf.analysis.get_com_of_each_galaxy(snap, initial_radius=10, masks=dm_id_masks, verbose=True)
virial_mass, virial_radius = cmf.analysis.get_virial_info_of_each_galaxy(snap, xcom=xcom, masks=[star_id_masks, dm_id_masks])

print(virial_radius)

linestyles = ['-', '--', ':']
fig, ax = plt.subplots(1,1)
ax.set_facecolor('black')
ax.set_title('DM Halo')
_,ax,*_ = pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax)
for key, gal_com in xcom.items():
    #ax.scatter(gal_com[0], gal_com[2], c='tab:red')
    for i in range(1, 4):
        circle = mplp.Circle((gal_com[0], gal_com[2]), radius=i*virial_radius[key], fill=False, ec='red', ls=linestyles[i-1])
        ax.add_artist(circle)
fig.tight_layout()
plt.savefig("/users/arawling/figures/set_up.png", dpi=300)
plt.show()
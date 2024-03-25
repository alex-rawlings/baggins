import matplotlib.animation as animation
import matplotlib.pyplot as plt
import baggins as bgs


datapath = "/scratch/pjohanss/arawling/gadget4-ketju/hardening_convergence/mergers/H_1-000/H_1-000-a-H_1-000-a-0.05-0.02/output"

snapfiles = bgs.utils.get_snapshots_in_dir(datapath)
ketjufile = bgs.utils.get_ketjubhs_in_dir(datapath)[0]

fig, ax = plt.subplots(1, 1)
ax.set_facecolor("k")
ax.set_aspect("equal")

smbh_anim = bgs.visualisation.SMBHtrajectory(
    ketjufile,
    ax,
    stepping={"start": 5000, "step": 5},
    centre=1,
    axis_offset=0.04,
    fix_centre=None,
    trails=10,
)

anim = animation.FuncAnimation(
    fig,
    smbh_anim,
    frames=smbh_anim.step_gen,
    save_count=smbh_anim.save_count,
    blit=True,
    repeat=False,
)

plt.show()

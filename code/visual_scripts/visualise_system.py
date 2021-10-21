import matplotlib.animation
import matplotlib.pyplot as plt
import cm_functions as cmf

path = "/scratch/pjohanss/arawling/collisionless_merger/stability-tests/triaxial/NGCa3348t/output/"
snaplist = cmf.utils.get_snapshots_in_dir(path)
snaplist = snaplist[:10]


fig, ax = plt.subplots(2,2, figsize=(6,6))
overview_anim = cmf.visualisation.OverviewAnimation(snaplist, fig, ax, orientate="red I")
anim = matplotlib.animation.FuncAnimation(fig, overview_anim, init_func=overview_anim, frames=len(snaplist), interval=100, blit=True)
plt.show()
#writer = matplotlib.animation.FFMpegWriter(fps=5, extra_args=["-crf", "28"], bitrate=-1)
#anim.save("/users/arawling/figures/movie.mp4", writer=writer, dpi=300)

"""
cmf.visualisation.overview_animation(snaplist, "red I")
plt.show()
"""
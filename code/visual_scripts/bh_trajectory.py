import matplotlib.animation
import matplotlib.pyplot as plt
import cm_functions as cmf

fig, ax = plt.subplots(1, 1)
bht = cmf.visualisation.SMBHtrajectory(
    "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations_eta_0002/000/output/ketju_bhs.hdf5",
    ax,
    axis_offset=0.01,
    stepping={"start": 0, "step": 100},
    only_bound=True,
    trails=500,
)
print(bht.save_count)
anim = matplotlib.animation.FuncAnimation(
    fig,
    bht,
    frames=bht.step_gen,
    blit=True,
    repeat_delay=500,
    save_count=bht.save_count,
)

writemovie = matplotlib.animation.FFMpegWriter(fps=30)
anim.save("/scratch/pjohanss/arawling/misc/my_movie.mp4", writemovie)

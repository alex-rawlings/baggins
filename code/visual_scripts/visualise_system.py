import argparse
import os
import matplotlib.animation
import matplotlib.pyplot as plt
import cm_functions as cmf


parser = argparse.ArgumentParser(
    description="Create an animation of the systems from snapshots", allow_abbrev=False
)
parser.add_argument(type=str, help="Directory containing all snapshots", dest="path")
parser.add_argument(type=str, help="Output directory", dest="outdir")
parser.add_argument(
    "-c",
    "--centre",
    type=str,
    help="Centre of animation",
    choices=["big", "small"],
    dest="centre",
    default=None,
)
parser.add_argument(
    "-fps", "--framespersec", type=int, help="frames per second", dest="fps", default=4
)
parser.add_argument(
    "-SE",
    "--StarExtent",
    type=float,
    help="axis offset for stellar plot",
    dest="starextent",
    default=500,
)
parser.add_argument(
    "-HE",
    "--HaloExtent",
    type=float,
    help="axis offset for halo plot",
    dest="haloextent",
    default=1000,
)
args = parser.parse_args()

snaplist = cmf.utils.get_snapshots_in_dir(args.path)

fig, ax = plt.subplots(2, 2, figsize=(6, 6), subplot_kw={"rasterized": True})
axis_offsets = {"stars": args.starextent, "dm": args.haloextent}

overview_anim = cmf.visualisation.OverviewAnimation(
    snaplist, fig, ax, centre=args.centre, axis_offsets=axis_offsets
)

anim = matplotlib.animation.FuncAnimation(
    fig,
    overview_anim,
    frames=overview_anim.step_gen,
    save_count=overview_anim.savecount,
    blit=True,
    repeat=False,
)

writemovie = matplotlib.animation.FFMpegWriter(fps=args.fps)
# TODO how to handle files with a _ in the name?
filename = os.path.splitext(snaplist[0])[0].split("/")[-1].split("_")[0]
save_file = os.path.join(args.outdir, "{}_overview.mp4".format(filename))
anim.save(save_file, writemovie)

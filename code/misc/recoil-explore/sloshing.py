import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygad
from arviz import plot_kde
from PIL import Image, GifImagePlugin
import baggins as bgs

bgs.plotting.check_backend()
GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY

parser = argparse.ArgumentParser(
    description="Investigate sloshing of sigma peak",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-kv", "--kick-vel", dest="kv", type=int, help="kick velocity", default=600
)
args = parser.parse_args()

snapfiles = bgs.utils.get_snapshots_in_dir(f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{args.kv:04d}/output")
save_path = f"/scratch/pjohanss/arawling/collisionless_merger/visualisations/recoil-explore/sloshing/v{args.kv:04d}_samex"
os.makedirs(save_path, exist_ok=True)

max_snap = 51

cmapper, _ = bgs.plotting.create_normed_colours(0, 1, cmap="viridis")
hdi_probs = [0.01, 0.1, 0.25, 0.5, 0.75]

if True:
    for j, snapfile in enumerate(snapfiles[:max_snap]):
        snap = pygad.Snapshot(snapfile, physical=True)
        bgs.analysis.basic_snapshot_centring(snap)
        t = bgs.general.convert_gadget_time(snap)

        # plot surface of section
        fig, ax = plt.subplots(1, 3, sharex="all", sharey="all")
        fig.set_figwidth(2 * fig.get_figwidth())
        for i, k in enumerate("xyz"):
            xmask = pygad.ExprMask("abs(pos[:,0]) < 15")
            vmask = pygad.ExprMask(f"abs(vel[:,{i}]) < 500")
            mask = xmask & vmask
            #print(f"proj {i} quantiles {np.quantile(snap.stars[mask]['vel'][:,0], [0.25, 0.5, 0.75])}")
            try:
                plot_kde(
                    snap.stars[mask]["pos"][:,0].view(np.ndarray),
                    snap.stars[mask]["vel"][:, i].view(np.ndarray),
                    ax=ax[i],
                    hdi_probs=hdi_probs,
                    quantiles=[],
                    contour_kwargs={"levels":[]},
                    fill_last=True
                )
            except ValueError:
                print(f"No contours made for projection {k}")
            ax[i].plot(snap.bh["pos"][:,0], snap.bh["vel"][:,i], marker="*", color="w", ls="")
            ax[i].set_xlabel("x/kpc")
            ax[i].set_ylabel(f"v{k}/km/s")
            ax[i].set_xlim(-15, 15)
            ax[i].set_ylim(-500, 500)
            ax[i].set_facecolor(cmapper(0))
        fig.suptitle(f"Time: {t:.3f} Gyr")
        bgs.plotting.savefig(os.path.join(save_path, f"slosh_frame_x_{j:03d}.png"), force_ext=True)
        plt.close()

        # conserve memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()

frames = bgs.utils.get_files_in_dir(save_path, ".png")
frames = [Image.open(img) for img in frames if "slosh_frame_x_" in img]  # Load images
frames.extend([frames[-1]] * 10)

# Save as an animated GIF
frames[0].save(
    os.path.join(save_path, f"animation_slosh_x_{args.kv:04d}.gif"),
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=200,
    loop=0,
    optimize=False
)

plt.close()
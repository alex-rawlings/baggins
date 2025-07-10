from itertools import chain
import os.path
import matplotlib.pyplot as plt
import baggins as bgs


fig, ax = plt.subplots(4, 3, figsize=(5, 7))

muse_files = [
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/ifu/muse_ifu_mock_02.pickle",
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/minor_mergers/ifu/muse_ifu_mock_minor_02.pickle"
]
data_major = bgs.utils.load_data(muse_files[0])
data_minor = bgs.utils.load_data(muse_files[1])

#bgs.general.print_dict_summary(data_minor)


# get the colour limits
def vor_generator_M():
    for v in data_major["0540"]["voronoi"]:
        yield v

def vor_generator_m():
    for v in data_minor["0000"]["voronoi"]:
        yield v

def vor_generator():
    vgM = vor_generator_M()
    vgm = vor_generator_m()
    for v in chain(vgM, vgm):
        yield v

vor_gen_M = vor_generator_M()
vor_gen_m = vor_generator_m()
clims_M = bgs.analysis.unify_IFU_colour_scheme(vor_gen_M)
clims_m = bgs.analysis.unify_IFU_colour_scheme(vor_gen_m)

vor_gen = vor_generator()
cbar_kwargs = {"labelsize":6}

for i, vg in enumerate(vor_gen):
    voronoi = bgs.analysis.VoronoiKinematics.load_from_dict(vg)
    r = (i // 3) * 2
    c = i % 3
    clims = clims_M if r == 0 else clims_m
    print(f"Map {i}: Doing panel {r, c}")
    voronoi.plot_kinematic_maps(
        ax = ax[r, c],
        moments = "2",
        clims = clims,
        cbar="inset",
        cbar_kwargs=cbar_kwargs
    )
    voronoi.plot_kinematic_maps(
        ax = ax[r+1, c],
        moments = "3",
        clims = clims,
        cbar="inset",
        cbar_kwargs=cbar_kwargs
    )
    for rr in (r, r+1):
        ax[rr, c].set_xticks([])
        ax[rr, c].set_yticks([])
        bgs.plotting.draw_sizebar(ax=ax[rr,c], length=10, units="kpc", size_vertical=0.5)

for i, t in enumerate(["Major", "Major", "Minor", "Minor"]):
    ax[i, 0].text(0.1, 0.1, t, transform=ax[i, 0].transAxes)


bgs.plotting.savefig(os.path.join(bgs.FIGDIR, "kicksurvey-study/referee_ifu.png"))
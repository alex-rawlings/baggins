import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot core fits given a Stan sample",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-e", "--extract", help="extract data", action="store_true", dest="extract"
)
parser.add_argument(
    "-p",
    "--parameter",
    help="parameter to plot",
    choices=["Re", "rb", "n", "a", "log10densb", "g"],
    default="rb",
    dest="param",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()


SL = bgs.setup_logger("script", args.verbosity)
data_dir = "/scratch/pjohanss/arawling/antti-cores/processed-data/"
data_file = os.path.join(data_dir, "bayesian_fits.pickle")
rng = np.random.default_rng(42)
bgs.plotting.set_publishing_style()


if args.extract:
    main_path = "/scratch/pjohanss/arawling/collisionless_merger/stan_files/density/antti"
    analysis_params = bgs.utils.read_parameters(
        "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-analysis/HMQcubes.yml"
    )

    with os.scandir(main_path) as _it:
        subdirs = [entry.path for entry in _it if entry.is_dir()]
    subdirs.sort()
    for s in subdirs:
        SL.debug(f"Reading: {s}")

    figname_base = "antti-core-all"

    # put the data into a format we can pickle as numpy arrays for faster
    # plotting
    data_cs = {
        "rb": {},
        "Re": {},
        "n": {},
        "log10densb": {},
        "g": {},
        "a": {},
        "R_OOS": {},
        "log10_surf_rho": {},
    }

    # load the fits
    for subdir in subdirs:
        csv_files = bgs.utils.get_files_in_dir(subdir, ext=".csv")[-4:]
        try:
            graham_model = bgs.analysis.GrahamModelSimple.load_fit(
                fit_files=csv_files,
                figname_base=figname_base,
            )
        except ValueError as e:
            SL.error(f"Unable to load data from directory: {subdir}: {e}. Skipping")
            continue
        SL.info(f"Loaded model from csv files {csv_files[0]}")

        graham_model.extract_data(analysis_params, None, binary=False)
        graham_model.set_stan_data()
        graham_model.sample_model(
            sample_kwargs=analysis_params["stan"]["density_sample_kwargs"],
            diagnose=False,
        )
        gid = graham_model.merger_id.replace("snapshot_last_", "")
        for k in data_cs.keys():
            if k == "R_OOS":
                data_cs[k][gid] = graham_model.stan_data[k]
            else:
                try:
                    data_cs[k][gid] = graham_model.sample_generated_quantity(
                        f"{k}_posterior", state="OOS"
                    )
                except ValueError:
                    data_cs[k][gid] = graham_model.sample_generated_quantity(
                        k, state="OOS"
                    )
    bgs.utils.save_data(data_cs, data_file)
else:
    SL.debug(f"Reading {data_file}")
    data_cs = bgs.utils.load_data(data_file)

# create figures
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_figwidth(2.5 * fig.get_figwidth())
cmapper_g, sm_g = bgs.plotting.create_normed_colours(1, 2, "crest")

figd, axd = plt.subplots(1, 5, sharex="all", sharey="all")
figd.set_figwidth(2.5 * figd.get_figwidth())
cmapper_d, sm_d = bgs.plotting.create_normed_colours(1e8, 2e9, "crest", norm="LogNorm")

cs_label = dict(
    rb = r"$r_\mathrm{b}/\mathrm{kpc}$",
    Re = r"$R_\mathrm{e}/\mathrm{kpc}$",
    log10densb = r"$\log_{10}\left(\Sigma_\mathrm{b}/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$",
    g = r"$\gamma$",
    n = r"$n$",
    a = r"$a$"
)

# first axis
ax1.set_xlabel(r"$M_\bullet/\mathrm{M}_\odot$")
ax1.set_ylabel(r"$r_\mathrm{SOI}/\mathrm{kpc}$")
ax1.set_xscale("log")
ax1.set_yscale("log")

# second axis
ax2.set_xlabel(r"$M_\bullet/\mathrm{M}_\odot$")
ax2.set_ylabel(cs_label[args.param])
ax2.set_xscale("log")
if args.param != "log10densb":
    ax2.set_yscale("log")

# third axis
ax3.set_xlabel(r"$r_\mathrm{SOI}/\mathrm{kpc}$")
ax3.set_ylabel(cs_label[args.param])
ax3.set_xscale("log")
if args.param != "log10densb":
    ax3.set_yscale("log")

# density axis
for axi in axd:
    axi.set_xlabel(r"$r/\mathrm{kpc}$")
axd[0].set_ylabel(r"$\log_{10}\left(\Sigma/(\mathrm{M}_\odot\mathrm{kpc}^{-2})\right)$")


all_pickle_files = bgs.utils.get_files_in_dir(data_dir, ext=".pickle")
snap_data_files = [f for f in all_pickle_files if "snapshot" in f]

gammas = []

for i, snap_data_file in enumerate(snap_data_files):
    key_cs = os.path.splitext(os.path.basename(snap_data_file))[0].replace("snapshot_last_", "")
    SL.info(f"Doing {key_cs}")
    gamma = float(key_cs[6:9])/100
    if gamma not in gammas:
        gammas.append(gamma)
    snap_data = bgs.utils.load_data(snap_data_file)
    mbh = np.sum(snap_data["particle_masses"]["bh"])

    # add to first plot
    ax1.plot(
        mbh,
        snap_data["rinf"],
        marker="o",
        c=cmapper_g(gamma),
        mec="k",
        mew=0.5
    )

    # add to second plot
    ymed, yerr = bgs.mathematics.quantiles_relative_to_median(data_cs[args.param][key_cs])
    ax2.errorbar(
        mbh,
        ymed,
        yerr=yerr,
        fmt="o",
        ecolor=cmapper_g(gamma),
        mfc=cmapper_g(gamma),
        mec="k",
        capthick=2,
        mew=0.5
    )

    # add to third plot
    ax3.errorbar(
        snap_data["rinf"],
        ymed,
        yerr=yerr,
        fmt="o",
        ecolor=cmapper_g(gamma),
        mfc=cmapper_g(gamma),
        mec="k",
        capthick=2,
        mew=0.5
    )

    # now density plot
    axd[int((gamma-1)*4)].loglog(
        data_cs["R_OOS"][key_cs],
        10**np.median(data_cs["log10_surf_rho"][key_cs], axis=0),
        lw = 2,
        c = cmapper_d(mbh),
        )

for axi, gamma in zip(axd, gammas):
    axi.text(1, 1e10, f"$\gamma_0 = {gamma:.2f}$")

'''
# add Thomas+16 relations, inverted
# rb - Mbh
x = np.geomspace(*ax2.get_xlim(), 10)
ax2.plot(x, 10**((np.log10(x) - 10.27) / 1.17), c="k", zorder=0.1, label="Thomas+16", alpha=0.7, lw=1)
ax2.legend()
# rb - rsoi
x = np.geomspace(*ax3.get_xlim(), 10)
ax3.plot(x, 10**((np.log10(x) + 0.01) / 0.95), c="k", zorder=0.1, alpha=0.7, lw=1)
'''

plt.colorbar(sm_g, ax=ax3, label=r"$\gamma_0$")
bgs.plotting.savefig("core_relations.pdf", force_ext=True, fig=fig)

plt.colorbar(sm_d, ax=axd[-1], label=r"$M_\bullet/\mathrm{M}_\odot$")
bgs.plotting.savefig("density.pdf", force_ext=True, fig=figd)

plt.close()

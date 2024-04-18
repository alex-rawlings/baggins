import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import yaml

import baggins as bgs

parser = argparse.ArgumentParser(
    description="Compare Graham density fits of different mergers", allow_abbrev=False
)
parser.add_argument(help=".yaml file of stan files to compare", type=str, dest="files")
parser.add_argument(help="family to compare", type=str, dest="fam")
parser.add_argument(
    "-c",
    "--colour-var",
    type=str,
    choices=["rperi", "res", "generic"],
    default="generic",
    dest="cvar",
    help="variable to colour samples by",
)
parser.add_argument(
    "-P", "--Publish", action="store_true", dest="publish", help="use publishing format"
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", console_level=args.verbose)

if args.publish:
    bgs.plotting.set_publishing_style()
    full_figsize = rcParams["figure.figsize"]
    full_figsize[0] *= 2
else:
    full_figsize = None

figname_base = "hierarchical_models/density/compare"
stan_model_file = "stan/density/graham.stan"
models = []
colour_var = []

with open(args.files, "r") as f:
    stan_files = yaml.safe_load(f)
try:
    for i, (k, f) in enumerate(stan_files[f"family-{args.fam}"].items()):
        models.append(
            bgs.analysis.StanModel.load_fit(
                stan_model_file, f["stan"], figname_base=figname_base
            )
        )
        if args.cvar == "rperi":
            colour_var.append(float(k.split("-")[-1]))
        elif args.cvar == "res":
            c = bgs.utils.get_files_in_dir(f["cube"])[0]
            _hmq = bgs.analysis.HMQuantitiesBinaryData.load_from_file(c)
            # TODO give actual values instead of fraction?
            s = k.split("-")[0]
            if len(s) == 1:
                s = f"{s}1000"
            colour_var.append(float(float(s[1:]) / 1000))
        elif args.cvar == "generic":
            colour_var.append(i)
        else:
            # this part of the code should not be reached...
            raise NotImplementedError("Only rperi and res implemented.")
except KeyError:
    SL.exception(
        f"No comparison available for '{args.fam}' in file {args.files}", exc_info=True
    )
    raise
SL.debug(f"Colour values are {colour_var}")

fig, ax = bgs.plotting.create_odd_number_subplots(
    2, 3, fkwargs={"figsize": full_figsize}
)
fig2, ax2 = bgs.plotting.create_odd_number_subplots(
    2, 3, fkwargs={"figsize": full_figsize}
)
ax[3].set_xscale("log")
for axi in ax2:
    axi.set_xlabel("Resolution")

# latent parameters to plot distributions of
latent_qtys = [
    "r_b_posterior",
    "Re_posterior",
    "I_b_posterior",
    "g_posterior",
    "n_posterior",
]
xlabels = [
    r"$r_\mathrm{b}$/kpc",
    r"$R_\mathrm{e}$/kpc",
    r"$\Sigma_\mathrm{b}/(10^{9}$M$_\odot$/kpc$^2)$",
    r"$\gamma$",
    r"$n$",
]
for axi, xl in zip(ax2, xlabels):
    axi.set_ylabel(xl)
    axi.set_xscale("log")

cols = bgs.plotting.mplColours()

for i, (m, cv) in enumerate(zip(models, colour_var)):
    m.plot_generated_quantity_dist(
        latent_qtys, xlabels=xlabels, ax=ax, plot_kwargs={"c": cols[i], "label": cv}
    )
    for j, lq in enumerate(latent_qtys):
        v = m.sample_generated_quantity(lq)
        med = np.nanmedian(v)
        ax2[j].errorbar(
            cv,
            med,
            yerr=np.atleast_2d(
                [med - np.nanquantile(v, 0.25), np.nanquantile(v, 0.75) - med]
            ).T,
            fmt="o",
            c="tab:blue",
            mec="k",
            mew=0.75,
        )
if args.cvar == "rperi":
    legend_title = r"$r_\mathrm{peri}$"
elif args.cvar == "res":
    legend_title = "Resolution"
else:
    legend_title = ""
ax[-1].legend(title=legend_title)

bgs.plotting.savefig(
    os.path.join(bgs.FIGDIR, f"{figname_base}_{args.fam}_gqs_{args.cvar}.png"), fig=fig
)
bgs.plotting.savefig(
    os.path.join(bgs.FIGDIR, f"{figname_base}_{args.fam}_gqs_{args.cvar}_2d.png"),
    fig=fig2,
)

if args.cvar == "rperi":
    # compare latent parameter distributions to observations
    fig, ax = plt.subplots(1, 3)
    sersic_n = bgs.literature.LiteratureTables("sahu_2020")
    sersic_n.scatter("logM*_sph", "n_maj", ax=ax[0])
    sersic_n.scatter("logM*_sph", "Re_maj_kpc", ax=ax[1])
    errbar_kwargs = {"fmt": "o", "ls": "", "elinewidth": 1}
    for i, (m, rp) in enumerate(zip(models, colour_var)):
        n = m.sample_generated_quantity("n_posterior")
        Re = m.sample_generated_quantity("Re_posterior")
        y = np.nanmedian(n)
        yerr = np.atleast_2d(
            [y - np.nanquantile(n, 0.16), np.nanquantile(n, 0.84) - y]
        ).T
        ax[0].errorbar(
            np.log10(3.66e11), np.atleast_2d(y), yerr=yerr, c=cols[i], **errbar_kwargs
        )
        y = np.nanmedian(Re)
        yerr = np.atleast_2d(
            [y - np.nanquantile(Re, 0.16), np.nanquantile(Re, 0.84) - y]
        ).T
        ax[1].errorbar(
            np.log10(3.66e11), np.atleast_2d(y), yerr=yerr, c=cols[i], **errbar_kwargs
        )


plt.show()

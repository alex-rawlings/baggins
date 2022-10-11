import argparse
from cmath import log
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yaml

import cm_functions as cmf


parser = argparse.ArgumentParser(description="Compare Graham density fits of different mergers", allow_abbrev=False)
parser.add_argument(help=".yaml file of stan files to compare", type=str, dest="files")
parser.add_argument(help="family to compare", type=str, dest="fam", choices=["AC", "AD", "AE", "CD", "CE", "DE", "HH"])
parser.add_argument("-c", "--colour-var", type=str, choices=["rperi", "res"], default="rperi", dest="cvar", help="variable to colour samples by")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args = parser.parse_args()

SL = cmf.CustomLogger("script", console_level=args.verbose)

figname_base = f"hierarchical_models/density/compare"
stan_model_file = "stan/graham.stan"
models = []
colour_var = []

with open(args.files, "r") as f:
    stan_files = yaml.safe_load(f)
for i, (k, f) in enumerate(stan_files[f"family-{args.fam}"].items()):
    models.append(cmf.analysis.StanModel.load_fit(stan_model_file, f["stan"], figname_base=figname_base))
    if args.cvar == "rperi":
        colour_var.append(float(k.split("-")[-1]))
    elif args.cvar == "res":
        c = cmf.utils.get_files_in_dir(f["cube"])[0]
        _hmq = cmf.analysis.HMQuantitiesData.load_from_file(c)
        # TODO give actual values instead of fraction?
        s = k.split("-")[0]
        if len(s)==1: s=f"{s}1000"
        colour_var.append(float(float(s[1:])/1000))
    else:
        # this part of the code should not be reached...
        raise NotImplementedError("Only rperi and res implemented.")
SL.logger.debug(f"Colour values are {colour_var}")

fig = plt.figure()
ax = []
gs = GridSpec(2, 6, figure=fig, top=0.95)
for i in range(3):
    ax.append(fig.add_subplot(gs[0, 2*i:2*(i+1)]))
for i in range(2):
    ax.append(fig.add_subplot(gs[1, 2*i+1:2*(i+1)+1]))
ax = np.array(ax)
ax[3].set_xscale("log")

vals = ["r_b_posterior", "Re_posterior", "I_b_posterior", "g_posterior", "n_posterior"]
xlabels = [r"$r_\mathrm{b}$/kpc", r"$R_\mathrm{e}$/kpc", r"$\Sigma_\mathrm{b}/(10^{9}$M$_\odot$/kpc$^2)$", r"$\gamma$", r"$n$"]


cmapper, sm = cmf.plotting.create_normed_colours(min(colour_var)*0.9, max(colour_var)*2.1, normalisation="LogNorm")
for i, (m, cv) in enumerate(zip(models, colour_var)):
    m.plot_generated_quantity_dist(vals, xlabels=xlabels, ax=ax, plot_kwargs={"c":cmapper(cv)})
cbar = plt.colorbar(sm, ax=ax[-1])
if args.cvar == "rperi":
    cbar_label = r"$R_\mathrm{peri}/R_{\mathrm{vir},1}$"
elif args.cvar == "res":
    cbar_label = r"$(M_\bullet/m_\star) / (M_\bullet/m_\star)|_\mathrm{fid}$"
cbar.set_label(cbar_label)
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"{figname_base}_{args.fam}_gqs_{args.cvar}.png"), fig=fig)

if args.cvar == "rperi":
    # compare latent parameter distributions to observations
    fig, ax = plt.subplots(1,3)
    sersic_n = cmf.literature.LiteratureTables("sahu_2020")
    sersic_n.scatter("logM*_sph", "n_maj", ax=ax[0])
    sersic_n.scatter("logM*_sph", "Re_maj_kpc", ax=ax[1])
    errbar_kwargs = {"fmt":"o", "ls":"", "elinewidth":1}
    for i, (m, rp) in enumerate(zip(models, colour_var)):
        n = m.sample_generated_quantity("n_posterior")
        Re = m.sample_generated_quantity("Re_posterior")
        y = np.nanmedian(n)
        yerr = np.atleast_2d([y-np.nanquantile(n, 0.16), np.nanquantile(n, 0.84)-y]).T
        ax[0].errorbar(np.log10(3.66e11), np.atleast_2d(y), yerr=yerr, c=cmapper(rp), **errbar_kwargs)
        y = np.nanmedian(Re)
        yerr = np.atleast_2d([y-np.nanquantile(Re, 0.16), np.nanquantile(Re, 0.84)-y]).T
        ax[1].errorbar(np.log10(3.66e11), np.atleast_2d(y), yerr=yerr, c=cmapper(rp), **errbar_kwargs)


plt.show()
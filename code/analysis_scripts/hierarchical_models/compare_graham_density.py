import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

import cm_functions as cmf


parser = argparse.ArgumentParser(description="Compare Graham density fits of different mergers", allow_abbrev=False)
parser.add_argument(help=".json file of stan files to compare", type=str, dest="files")
parser.add_argument(help="family to compare", type=str, dest="fam", choices=["AC", "AD", "AE", "CD", "CE", "DE"])
args = parser.parse_args()

figname_base = f"hierarchical_models/density/compare"
models = []
#stellar_masses = {}
rperi = []

with open(args.files, "r") as f:
    stan_files = json.load(f)
for i, (k, f) in enumerate(stan_files["models"][f"family-{args.fam}"].items()):
    models.append(cmf.analysis.StanModel.load_fit(f["stan"], figname_base=figname_base))
    rperi.append(float(k.split("-")[-1]))
    #stellar_masses[rperi[i]] = []
    for c in cmf.utils.get_files_in_dir(f["cube"]):
        _hmq = cmf.analysis.HMQuantitiesData.load_from_file(c)
        #stellar_masses[rperi[i]].append(_hmq.masses_in_galaxy_radius["stars"])


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
xlabels = [r"$r_\mathrm{b}$/kpc", r"$R_\mathrm{e}$/kpc", r"$\Sigma_\mathrm{b}/(10^{10}$M$_\odot$/kpc$^2)$", r"$\gamma$", r"$n$"]


cmapper, sm = cmf.plotting.create_normed_colours(min(rperi)*0.9, max(rperi)*2.1, normalisation="LogNorm")
for i, (m, rp) in enumerate(zip(models, rperi)):
    m.plot_generated_quantity_dist(vals, xlabels=xlabels, ax=ax, plot_kwargs={"c":cmapper(rp)})
cbar = plt.colorbar(sm, ax=ax[-1])
cbar.set_label(r"$R_\mathrm{peri}/R_{\mathrm{vir},1}$")
cmf.plotting.savefig(os.path.join(cmf.FIGDIR, f"{figname_base}_{args.fam}_gqs.png"), fig=fig)

# compare latent parameter distributions to observations
fig, ax = plt.subplots(1,3)
sersic_n = cmf.literature.LiteratureTables("sahu_2020")
sersic_n.scatter("logM*_sph", "n_maj", ax=ax[0])
sersic_n.scatter("logM*_sph", "Re_maj_kpc", ax=ax[1])
errbar_kwargs = {"fmt":"o", "ls":"", "elinewidth":1}
for i, (m, rp) in enumerate(zip(models, rperi)):
    n = m.sample_generated_quantity("n_posterior")
    Re = m.sample_generated_quantity("Re_posterior")
    #_sm = stellar_masses[rperi[i]]
    y = np.nanmedian(n)
    yerr = np.atleast_2d([y-np.nanquantile(n, 0.16), np.nanquantile(n, 0.84)-y]).T
    ax[0].errorbar(np.log10(3.66e11), np.atleast_2d(y), yerr=yerr, c=cmapper(rp), **errbar_kwargs)
    y = np.nanmedian(Re)
    yerr = np.atleast_2d([y-np.nanquantile(Re, 0.16), np.nanquantile(Re, 0.84)-y]).T
    ax[1].errorbar(np.log10(3.66e11), np.atleast_2d(y), yerr=yerr, c=cmapper(rp), **errbar_kwargs)


plt.show()
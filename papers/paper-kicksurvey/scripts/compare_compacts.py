import os.path
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import baggins as bgs
import figure_config

bgs.plotting.check_backend()


cols = figure_config.custom_colors_shuffled
fig, ax = plt.subplots(1, 2, sharey="all")
fig.set_figwidth(2 * fig.get_figwidth())
legend_font = 5
rng = np.random.default_rng(42)
vk_cols = figure_config.VkickColourMap()


def scatter_kwargs_maker():
    """
    Generate scatter kwargs

    Yields
    ------
    : dict
        plotting kwargs
    """
    for col, char in zip(
        figure_config.custom_colors_shuffled, bgs.plotting.mplChars()[1:]
    ):
        yield {
            "mec": "k",
            "mew": 0.1,
            "ms": 2,
            "c": col,
            "fmt": char,
            "elinewidth": 0.5,
            "capsize": 0,
        }


def scatter_kwargs_from_prev(p):
    """
    Get consistent plotting kwargs when data plotted over multiple axes

    Parameters
    ----------
    p : pyplot.ErrorbarContainer
        output from pyplot.errorbar()

    Returns
    -------
    : dict
        plotting kwargs
    """
    return {
        "mec": "k",
        "mew": 0.1,
        "ms": 2,
        "c": p[0].get_color(),
        "fmt": p[0].get_marker(),
        "elinewidth": 0.5,
        "capsize": 0,
    }


def make_cluster_mean_and_error(q, xy):
    """
    Determine the mean and error of cluster properties in log space

    Parameters
    ----------
    q : array-like
        quantity
    xy : str
        x or y quantity to plot

    Returns
    -------
    : dict
        axis mean and error for plt.errorbar()
    """
    xy = xy.lower()
    assert xy == "x" or xy == "y"
    qmean = np.nanmean(np.log10(q))
    qstd = np.nanstd(np.log10(q))
    return {
        xy: 10**qmean,
        f"{xy}err": np.atleast_2d(
            [10**qmean - 10 ** (qmean - qstd), 10 ** (qmean + qstd) - 10**qmean]
        ).T,
    }


def load_cluster_data():
    dat_files = bgs.utils.get_files_in_dir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/perfect_obs/",
        ".pickle",
    )
    for f in dat_files:
        vk = float(os.path.splitext(os.path.basename(f))[0].replace("perf_obs_", ""))
        if vk > 1080:
            continue
        cluster_mass = []
        cluster_Re = []
        cluster_vsig = []
        cluster = bgs.utils.load_data(f)["cluster_props"]
        for c in cluster:
            if c["visible"]:
                cluster_mass.append(c["cluster_mass"])
                cluster_Re.append(c["cluster_Re_pc"])
                if c["cluster_vsig"]:
                    cluster_vsig.append(c["cluster_vsig"])
        cluster_mass = np.asarray(cluster_mass)
        cluster_Re = np.asarray(cluster_Re)
        cluster_vsig = np.asarray(cluster_vsig)
        yield cluster_mass, cluster_Re, cluster_vsig, vk_cols.get_colour(vk)


cluster_plot_kwargs = {
    "fmt": "o",
    "markersize": 4,
    "elinewidth": 0.5,
    "capsize": 0,
    "mec": "k",
    "mew": 0.1,
}

# XXX: FIGURE 1 - MASS VS RE
carlsten20 = bgs.literature.LiteratureTables.load_carlsten_2020_data()
misgeld09 = bgs.literature.LiteratureTables.load_misgeld_2009_data()
misgeld11 = bgs.literature.LiteratureTables.load_misgeld_2011_data()
price09 = bgs.literature.LiteratureTables.load_price_2009_data()
mcconnachie12 = bgs.literature.LiteratureTables.load_mcconnachie_2012_data()
siljeg24 = bgs.literature.LiteratureTables.load_siljeg_2024_data()

sk_gen = scatter_kwargs_maker()
cluster_gen = load_cluster_data()
for i, props in enumerate(cluster_gen):
    ax[0].errorbar(
        **make_cluster_mean_and_error(props[0], "x"),
        **make_cluster_mean_and_error(props[1], "y"),
        **cluster_plot_kwargs,
        c=props[3],
        label=r"$\mathrm{BH\; cluster}$" if i == 0 else "",
    )
    ax[1].errorbar(
        **make_cluster_mean_and_error(props[2], "x"),
        **make_cluster_mean_and_error(props[1], "y"),
        **cluster_plot_kwargs,
        c=props[3],
        label=r"$\mathrm{BH\; cluster}$" if i == 0 else "",
    )

misgeld09.scatter(
    "mass",
    "Re_pc",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="Re_err_pc",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
price09.scatter(
    "mass",
    "Re",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="Re_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
# do this data in two parts, as there are a large number of points below Re=10pc
mask = misgeld11.table.loc[:, "Re_pc"] > 10
misgeld11.scatter(
    "mass",
    "Re_pc",
    scatter_kwargs={"fmt": ".", "ms": 1.5, "c": "gray", "zorder": 0.1, "mew": 0},
    ax=ax[0],
    mask=mask,
)
downsampled = np.logical_and(
    ~mask, bernoulli.rvs(0.2, size=misgeld11.num_obs, random_state=rng)
)
misgeld11.scatter(
    "mass",
    "Re_pc",
    scatter_kwargs={"fmt": ".", "ms": 1.5, "c": "gray", "zorder": 0.1, "mew": 0},
    ax=ax[0],
    mask=downsampled,
    use_label=False,
)
_, m12p = mcconnachie12.scatter(
    "mass",
    "rh",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="rh_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
carlsten20.scatter(
    "mass",
    "re_pc",
    xerr=["mass_err_low", "mass_err_up"],
    yerr="re_err_pc",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)
_, s24p = siljeg24.scatter(
    "mass",
    "Re_pc",
    xerr="mass_err",
    yerr="Re_pc_err",
    scatter_kwargs=next(sk_gen),
    ax=ax[0],
)

ax[0].set_xlim(1e3, ax[0].get_xlim()[1])
ax[0].set_ylim(1, ax[0].get_ylim()[1])
ax[0].legend(fontsize=legend_font)
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"$M_\star/\mathrm{M}_\odot$")
ax[0].set_ylabel(r"$R_\mathrm{e}/\mathrm{pc}$")

# XXX: FIGURE 2 - SIGMA VS RE
harris10 = bgs.literature.LiteratureTables.load_harris_2010_data()

harris10.scatter("sig_v", "Re", xerr="sig_v_err", scatter_kwargs=next(sk_gen), ax=ax[1])
mcconnachie12.scatter(
    "vsig",
    "rh",
    xerr="vsig_err",
    yerr="rh_err",
    scatter_kwargs=scatter_kwargs_from_prev(m12p),
    ax=ax[1],
)
siljeg24.scatter(
    "vsig",
    "Re_pc",
    xerr="vsig_err",
    yerr="Re_pc_err",
    scatter_kwargs=scatter_kwargs_from_prev(s24p),
    ax=ax[1],
)

ax[1].legend(fontsize=legend_font)
ax[1].set_xlabel(r"$\sigma_\star/\mathrm{km\,s}^{-1}$")
ax[1].set_ylabel("")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
vk_cols.make_cbar(ax=ax[1])
bgs.plotting.savefig(figure_config.fig_path("compact.pdf"), force_ext=True)

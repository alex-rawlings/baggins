import os.path
import numpy as np
import matplotlib.pyplot as plt
import baggins as bgs

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
        yield cluster_mass, cluster_Re, cluster_vsig, vk


def data_grabber():
    """
    Generator to get the data to plot

    Yields
    ------
    return_dict : dict
        plotting data
    """
    for i, df in enumerate(bgs.utils.get_files_in_dir(os.path.join("/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data", "bound_stars"), ".pickle")):
        data = bgs.utils.load_data(df)
        # XXX temp fix
        vk = float(
            os.path.splitext(os.path.basename(df))[0]
            .replace("kick-vel-", "")
            .replace("-bound", "")
        )
        if i == 0:
            m_bh = data["other"]["mbh"]
        # need to find first pericentre
        return_dict = {
            "vk": vk,
            "bound_a": np.nan,
            "bound_p": np.nan,
            "r_a": np.nan,
            "r_p": np.nan,
            "m_bh": m_bh,
            "rhalf": np.nan
        }
        try:
            return_dict["bound_a"] = data["bound_mass"][0]
            return_dict["bound_p"] = data["bound_mass"][1]
            return_dict["r_a"] = data["r"][0]
            return_dict["r_p"] = data["r"][1]
            return_dict["rhalf"] = data["rhalf"][0]
        except ValueError as err:
            print(f"Skipping: {vk}")
        yield return_dict

def keymaker(s):
    return f"v{int(s):d}"


fig, ax = plt.subplots(1, 2, figsize=(7,4))

grab_data = data_grabber()
dat1M = {}
dat1R = {}
for g in grab_data:
    dat1M[keymaker(g["vk"])] = g["bound_a"]
    dat1R[keymaker(g["vk"])] = g["rhalf"]
obs_data = load_cluster_data()
dat2M = {}
dat2R = {}
for o in obs_data:
    try:
        dat2M[keymaker(o[3])] = o[0][-1]
    except IndexError:
        dat2M[keymaker(o[3])] = o[0]
    try:
        dat2R[keymaker(o[3])] = o[1][-1]
    except IndexError:
        dat2R[keymaker(o[3])] = o[1]

ax[0].set_xlabel("Energy cut mass [Msun]")
ax[0].set_ylabel("Obs.method mass [Msun]")
ax[1].set_xlabel("Energy cut half radius [pc]")
ax[1].set_ylabel("Obs.method half radius [pc]")

cmapper, sm = bgs.plotting.create_normed_colours(300, 1080)

for k in dat1M.keys():
    print(f"Doing key {k}")
    try:
        ax[0].plot(dat1M[k], dat2M[k], marker="o", ls="", c=cmapper(float(k.strip("v"))), mew=0.5, mec="k")
        ax[1].plot(dat1R[k], dat2R[k], marker="o", ls="", c=cmapper(float(k.strip("v"))), mew=0.5, mec="k")
    except KeyError:
        print(f"Skipping {k}, no key in dat2")
    except ValueError as err:
        print(err)
for axi in ax:
    x = np.linspace(*axi.get_xlim(), 10)
    axi.plot(x, x, ls="-", c="k", label="y=x")
    axi.legend()
plt.colorbar(sm, ax=ax[1], label="kick vel")
bgs.plotting.savefig("measure_compare.png")

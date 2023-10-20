import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pygad
import cm_functions as cmf


runs = dict(
    seeds_high = "/scratch/pjohanss/arawling/gadget4-ketju/hardening_convergence/mergers/H_1-000",
    seeds_low = "/scratch/pjohanss/arawling/gadget4-ketju/hardening_convergence/mergers/H_0-100",
    bumps_high = "/scratch/pjohanss/arawling/testing/temp_perturb_ics/H_1-000-a-H_1-000-a-0.05-0.02/temp_perturbs", 
    randomise_bh = "/scratch/pjohanss/arawling/testing/temp_perturb_ics/H_1-000-a-H_1-000-a-0.05-0.02/temp_perturbs_bh"
)

softenings = {"stars":0.005, "dm":0.3, "bh":0.005}

SL = cmf.setup_logger("script", console_level="DEBUG")

cols = cmf.plotting.mplColours()

hamiltonian_dat = {"run":[], "H":[]}

for i, (key, dir) in enumerate(runs.items()):
    subdirs = [d.path for d in os.scandir(dir) if os.path.isdir(d)]
    for j, subdir in enumerate(subdirs):
        SL.debug(f"Reading from subdir {subdir}")
        snapfiles = cmf.utils.get_snapshots_in_dir(subdir)
        if not snapfiles:
            continue
        perturb_idx = cmf.analysis.snap_num_for_time(snapfiles, 10)
        snap = pygad.Snapshot(snapfiles[perturb_idx], physical=True)
        bhsep = pygad.utils.geo.dist(snap.bh["pos"][0,:], snap.bh["pos"][1,:])
        assert bhsep > 10
        bhid = snap.bh["ID"][0]
        bhid_mask = pygad.IDMask(bhid)
        SL.debug(f"Translating by:\n{-snap.bh[bhid_mask]['pos'].flatten()}")
        translation = pygad.Translation(-snap.bh[bhid_mask]["pos"].flatten())
        translation.apply(snap, total=True)
        # TODO is this being affected by some extreme value from DM or a BH?
        hamiltonian_dat["H"] = cmf.analysis.calculate_Hamiltonian(snap)
        SL.debug(f"Hamiltonian:\n{hamiltonian_dat['H'][-1]}")
        hamiltonian_dat["run"].append(key)
        pygad.gc_full_collect()
        del snap
accel_df = pd.DataFrame(hamiltonian_dat)

sns.violinplot(data=hamiltonian_dat, x="run", y=k, hue="run", dodge=False, orient="v", ax=axi)

plt.show()

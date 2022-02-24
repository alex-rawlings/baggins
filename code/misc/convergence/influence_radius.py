import os.path
import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


paramfile_base = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/resolution-convergence/hardening/"

if False:
    data = dict()

    for pfR, n in zip(("DE-030-0005r1.py", "DE-030-0005r2.py", "DE-030-0005r5.py"), ("1", "2", "5")):
        pf = os.path.join(paramfile_base, pfR)
        pfv = cmf.utils.read_parameters(pf)
        data[n] = dict()
        data[n]["child"] = dict()
        data[n]["child"]["time"] = []
        data[n]["child"]["rinfl"] = []
        data[n]["parent"] = dict()
        data[n]["parent"]["time"] = []
        data[n]["parent"]["rinfl"] = []

        for i in range(11):
            if i<10:
                print("Child {}".format(i))
                perturb_idx = "{:03d}".format(i)
                snaplist = cmf.utils.get_snapshots_in_dir(os.path.join(pfv.full_save_location, pfv.perturbSubDir, perturb_idx, "output"))
                use_binary = True
            else:
                print("Parent")
                snaplist = cmf.utils.get_snapshots_in_dir(os.path.join(pfv.full_save_location, "output"))
                use_binary = False
            rinfl_list = []
            time_list = []
            for snapfile in snaplist:
                snap = pygad.Snapshot(snapfile, physical=True)
                time_list.append(cmf.general.convert_gadget_time(snap))
                rinfl_list.append(
                    list(cmf.analysis.influence_radius(snap, binary=use_binary).values())[0].in_units_of("pc")
                )
                snap.delete_blocks()
            if i<10:
                data[n]["child"]["rinfl"].append(rinfl_list)
                data[n]["child"]["time"].append(time_list)
            else:
                data[n]["parent"]["rinfl"].append(rinfl_list)
                data[n]["parent"]["time"].append(time_list)
    cmf.utils.save_data(data, "influence_radius.pickle")
else:
    data = cmf.utils.load_data("influence_radius.pickle")

#set up plot
fig, ax = plt.subplots(1,1)
ax.set_xlabel("Time [Gyr]")
ax.set_ylabel(r"$r_\mathrm{infl}$ [pc]")
for (k, res), c in zip(data.items(), ("tab:blue", "tab:orange", "tab:green")):
    ax.plot(res["parent"]["time"][0], res["parent"]["rinfl"][0], c=c, ls=":")
    for i, (ts, rinfs) in enumerate(zip(res["child"]["time"], res["child"]["rinfl"])):
        ts = np.array(ts)
        ax.plot(ts[:-1]+res["parent"]["time"][-1][-1], rinfs[:-1], c=c, label=(r"$N_\mathrm{{fid}}$/{}".format(k) if i==0 else ""))
plt.legend()
plt.show()
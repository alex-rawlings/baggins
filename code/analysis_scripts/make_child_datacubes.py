import cm_functions as cmf

paramfile = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AC/AC-030-0050.py"
child = "004"
dc = cmf.analysis.ChildSim(paramfile, child)
#dc.get_shell_velocity_stats()

if False:
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    vwl = [v/w for (v,w) in zip(dc.stellar_shell_outflow_velocity, dc.bh_binary_watershed_velocity)]
    vw = -np.concatenate(vwl)
    ts = np.repeat(dc.snapshot_times, [len(l) for l in dc.stellar_shell_outflow_velocity])
    j = cmf.plotting.seaborn_jointplot_cbar(x=ts, y=vw, kind="hist", binwidth=(dc.snapshot_times[1]-dc.snapshot_times[0], 0.05), cbar=True, cbar_kws={"label":"Count"})
    j.refline(y=1)
    j.set_axis_labels("t/Myr", "v/w")
    j.figure.suptitle("Incoming velocity of Stellar Particles through a 30pc Shell")
    plt.show()
    quit()


quit()
#dc.load_all()
dc.print()
dc.make_hdf5("my_cube.hdf5")

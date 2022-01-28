import cm_functions as cmf

paramfile = "/users/arawling/projects/collisionless-merger-sample/parameters/parameters-mergers/main/AC/AC-030-0005.py"
child = "004"
dc = cmf.analysis.ChildDataCube(paramfile, child)
dc.get_shell_velocity_stats()

if True:
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    vwl = [l/w for (l,w) in zip(dc.stellar_shell_outflow_velocity, dc.bh_binary_watershed_velocity)]
    vw = -np.concatenate(vwl)
    ts = np.repeat(dc.snapshot_times, [len(l) for l in dc.stellar_shell_outflow_velocity])
    j = sns.jointplot(x=ts, y=vw)
    j.refline(y=1)
    j.set_axis_labels("t/Myr", "v/w")
    j.fig.suptitle("Incoming velocity of Stellar Particles through a 30pc Shell")
    plt.subplots_adjust(left=0.1, bottom=0.08, top=0.97)
    plt.show()
    quit()


quit()
#dc.load_all()
dc.print()
dc.make_hdf5("my_cube.hdf5")
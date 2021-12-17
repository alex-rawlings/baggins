import numpy as np
import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf


class ShellData:
    def __init__(self, shells):
        self.t = []
        self.n = len(shells)
        self.shells = shells
        # TODO need to account for x,y,z in each shell
        self.xcom_offsets_1 = [[] for i in range(self.n)]
        self.xcom_offsets_2 = [[] for i in range(self.n)]
        self.vcom_offsets_1 = [[] for i in range(self.n)]
        self.vcom_offsets_2 = [[] for i in range(self.n)]
    
    def add_data(self, t, xdat, vdat):
        self.t.append(t)
        for k, xoff, voff in zip(
                                xdat.keys(),
                                (self.xcom_offsets_1, self.xcom_offsets_2),
                                (self.vcom_offsets_1, self.vcom_offsets_2)
        ):
            for i in range(self.n):
                xoff[i].append(xdat[k][i])
                voff[i].append(vdat[k][i])
    
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,2, sharex="all")
        ax[0].plot(self.t, self.xcom_offsets_1, c=self.shells)
        ax[1].plot(self.t, self.vcom_offsets_1, c=self.shells)

if False:
    mainpath = "/scratch/pjohanss/arawling/collisionless_merger/mergers/D-E-3.0-0.005/output/"
    shells = {"start":1e-6, "stop":500, "num":5}

    snaplist = cmf.utils.get_snapshots_in_dir(mainpath)
    shell_data = ShellData(np.geomspace(**shells))

    for i, snapfile in enumerate(snaplist[:3]):
        snap = pygad.Snapshot(snapfile, physical=True)
        snap_time = cmf.general.convert_gadget_time(snap)
        xcoms, vcoms = cmf.analysis.shell_com_motions_each_galaxy(snap, shell_kw=shells)
        shell_data.add_data(snap_time, xcoms, vcoms)

    data_dict = {"shell_data": shell_data}

    cmf.utils.save_data(data_dict, "shell_motion.pickle")
else:
    data_dict = cmf.utils.load_data("shell_motion.pickle")
    shell_data = data_dict["shell_data"]
print(shell_data.xcom_offsets_1)
shell_data.plot()
plt.show()
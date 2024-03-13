import argparse
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib import colors
import pygad
import baggins as bgs


class ShellData:
    def __init__(self, shells):
        self.t = []
        self.r = []
        self.n = len(shells)
        self.shells = shells
        self.xcom_offsets_1 = [[] for i in range(self.n)]
        self.xcom_offsets_2 = [[] for i in range(self.n)]
        self.vcom_offsets_1 = [[] for i in range(self.n)]
        self.vcom_offsets_2 = [[] for i in range(self.n)]

    def add_data(self, t, r, xdat, vdat):
        self.t.append(t)
        self.r.append(r)
        for k, xoff, voff in zip(
            xdat.keys(),
            (self.xcom_offsets_1, self.xcom_offsets_2),
            (self.vcom_offsets_1, self.vcom_offsets_2),
        ):
            for i in range(self.n):
                xoff[i].append(xdat[k][i])
                voff[i].append(vdat[k][i])

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2, sharex="all")
        # TODO mapping t -> r is incorrect
        f_tr = scipy.interpolate.interp1d(self.t, self.r, bounds_error=False)
        ax2_1 = bgs.plotting.twin_axes_plot(ax[0], f_tr)
        ax2_2 = bgs.plotting.twin_axes_plot(ax[1], f_tr)
        norm = colors.LogNorm(min(self.shells), max(self.shells))
        self.t = np.array(self.t)
        self.r = np.array(self.r)
        mask = self.t < 1.6
        for i, (s, xoff, voff) in enumerate(
            zip(self.shells, self.xcom_offsets_1, self.vcom_offsets_1)
        ):
            if i < 2:
                continue
            xoff = np.array(xoff)
            voff = np.array(voff)
            ax[0].semilogy(self.t[mask], xoff[mask], color=plt.cm.viridis(norm(s)))
            ax[1].semilogy(self.t[mask], voff[mask], color=plt.cm.viridis(norm(s)))
        plt.colorbar(
            plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm),
            ax=ax[1],
            label="R/kpc",
        )
        ax[0].set_xlabel("t/Gyr")
        ax[1].set_xlabel("t/Gyr")
        ax[0].set_ylabel(
            "Magnitude of Position Offset between Shell CoM and Global CoM [kpc]"
        )
        ax[1].set_ylabel(
            "Magnitude of Velocity Offset between Shell CoM and Global CoM [km/s]"
        )
        ax2_1.twin_ax.set_xlabel("BH Separation [kpc]")
        ax2_2.twin_ax.set_xlabel("BH Separation [kpc]")
        return ax


parser = argparse.ArgumentParser(
    description="Create figure of CoM vs shell offset", allow_abbrev=False
)
parser.add_argument(
    "-n", "--new", help="analyse a new dataset", dest="new", action="store_true"
)
args = parser.parse_args()

if args.new:
    mainpath = (
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/D-E-3.0-0.005/output/"
    )
    shells = {"start": 1e-4, "stop": 500, "num": 10}

    snaplist = bgs.utils.get_snapshots_in_dir(mainpath)
    bhfile = bgs.utils.get_ketjubhs_in_dir(mainpath)[0]
    bh1, bh2, merged = bgs.analysis.get_bh_particles(bhfile)
    peri_time = bgs.analysis.find_pericentre_time(bh1, bh2)[0][0] / 1e3  # Gyr

    shell_data = ShellData(np.geomspace(**shells))

    for i, snapfile in enumerate(snaplist):
        snap = pygad.Snapshot(snapfile, physical=True)
        snap_time = bgs.general.convert_gadget_time(snap)
        if snap_time > peri_time:
            break
        (
            xcoms,
            vcoms,
            global_xcom,
            global_vcom,
        ) = bgs.analysis.shell_com_motions_each_galaxy(snap, shell_kw=shells)
        xcom_mag = dict()
        vcom_mag = dict()
        for k in xcoms:
            xcom_mag[k] = bgs.mathematics.radial_separation(xcoms[k], global_xcom[k])
            vcom_mag[k] = bgs.mathematics.radial_separation(global_vcom[k], vcoms[k])
        bh_sep = bgs.mathematics.radial_separation(
            snap.bh["pos"][0, :], snap.bh["pos"][1, :]
        )[0]
        shell_data.add_data(snap_time, bh_sep, xcom_mag, vcom_mag)

    data_dict = {"shell_data": shell_data}

    bgs.utils.save_data(data_dict, "shell_motion.pickle")
else:
    data_dict = bgs.utils.load_data("shell_motion.pickle")
    shell_data = data_dict["shell_data"]
shell_data.plot()
plt.show()

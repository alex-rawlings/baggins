import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from seaborn import color_palette
from datetime import datetime
import ketjugw
import baggins as bgs


class Trajectory_Langrangian:
    def _data_init(self, kf, lf):
        bh1, bh2 = ketjugw.load_hdf5(kf).values()
        bh = bh1 if len(bh1) > len(bh2) else bh2
        bh = bh[10000:]
        bh.x /= bgs.general.units.kpc
        bh.t /= bgs.general.units.Myr
        bh.x -= bh.x[0, :]
        # TODO return appropriate Lagrangian data
        lang_data = np.loadtxt(lf, skiprows=1, dtype=float)
        # need the Gadget time conversion factor
        GADGET_FACTOR = 0.978 * 1e3
        return bh, lambda t: np.interp(
            t, lang_data[:, 0] * GADGET_FACTOR, lang_data[:, 4]
        )

    def _init_lists(self):
        self.xdata, self.ydata = [[], []], [[], []]

    def __init__(self, kf, lf, maxN=None, step=100):
        self.step = int(step)
        self.bh, self.lang = self._data_init(kf, lf)
        if maxN is None:
            self.maxN = len(self.bh)
        else:
            self.maxN = min([int(maxN), len(self.bh)])
        self._zoomed = False
        self._rb0 = 0.58

        # set up the figure
        self.fig, self.ax = plt.subplots(1, 2, figsize=(5, 3))
        self.ax[0].set_facecolor("#303030")
        self.ax[0].set_aspect("equal")
        self.ax[0].set_xlim(-22, 22)
        self.ax[0].set_ylim(-18, 18)
        self.ax[0].set_xticklabels([])
        self.ax[0].set_yticklabels([])
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self._sizebar_kw = dict(
            ax=self.ax[0], units="kpc", color="w", fmt=".0f", location="lower left"
        )
        self.sizebar = bgs.plotting.draw_sizebar(length=5, **self._sizebar_kw)
        self.ax[1].set_xlim(0, 1.01 * self.bh.t[self.maxN])
        self.ax[1].set_ylim(1, 2)
        self.ax[1].set_xlabel(r"$t/\mathrm{Myr}$")
        self.ax[1].set_ylabel(r"$r(M_\star=M_\bullet)/\mathrm{kpc}$")

        self.frames = np.arange(0, self.maxN + 1, self.step)
        self._init_lists()

        # start with a blank plot
        self._bh_col = color_palette("icefire", 50).as_hex()[5]
        (self.line1,) = self.ax[0].plot(
            [],
            [],
            c=self._bh_col,
            marker="o",
            markevery=[-1],
            mec=self.ax[0].get_facecolor(),
            mew=0.5,
            lw=2,
        )
        (self.line2,) = self.ax[1].plot([], [], lw=2)

        # add circle to show binary core radius
        core_circle = Circle((0, 0), self._rb0, ec="w", ls=":", lw=1, fill=None)
        self.ax[0].add_artist(core_circle)

        # initiate zoom generator
        self.zoomer = self.zoom_in([-0.9, 0.9], [-0.9, 0.9], 20)
        self._zoom_count = 0

    def zoom_in(self, xlim, ylim, nframes):
        # define how much the window should zoom in each iteration
        xlim_delta = 0.5 * (np.diff(self.ax[0].get_xlim()) - np.diff(xlim)) / nframes
        ylim_delta = 0.5 * (np.diff(self.ax[0].get_ylim()) - np.diff(ylim)) / nframes
        xlim0 = self.ax[0].get_xlim()
        ylim0 = self.ax[0].get_ylim()
        sb_redrawn = False
        for i in range(1, nframes + 1):
            self.ax[0].set_xlim(xlim0[0] + i * xlim_delta, xlim0[1] - i * xlim_delta)
            self.ax[0].set_ylim(ylim0[0] + i * ylim_delta, ylim0[1] - i * ylim_delta)
            if not sb_redrawn and np.abs(np.diff(self.ax[0].get_xlim())) < 10:
                self.sizebar.remove()
                self.sizebar = bgs.plotting.draw_sizebar(length=1, **self._sizebar_kw)
                sb_redrawn = True
            yield self.line1

    def __call__(self, i):
        if i == 0:
            self._init_lists()
        print(f"Generating frame {i:07d} of {self.maxN}...              ", end="\r")
        if i > 9e5 and not self._zoomed:
            # zoom in
            while True:
                try:
                    self._zoom_count += self.step
                    next(self.zoomer)
                    return self.line1, self.line2
                except StopIteration:
                    self._zoomed = True
                    text_angle = 0.20 * np.pi
                    self.ax[0].text(
                        self._rb0 * np.cos(text_angle),
                        self._rb0 * np.cos(text_angle),
                        r"$r_\mathrm{b,0}$",
                        ha="left",
                        va="bottom",
                        c="w",
                    )
                    self.ax[0].scatter(
                        0, 0, c=self._bh_col, marker=(9, 1, 10), zorder=1
                    )
                    break
        else:
            self.fig.suptitle(
                f"$t = {self.bh.t[i-self._zoom_count]:>6.0f}\,\mathrm{{Myr}}$",
                fontsize="x-large",
            )
            self.xdata[0].append(self.bh.x[i - self._zoom_count, 0])
            self.ydata[0].append(self.bh.x[i - self._zoom_count, 2])
            self.xdata[1].append(self.bh.t[i - self._zoom_count])
            self.ydata[1].append(self.lang(self.xdata[1][-1]))
            self.line1.set_data(self.xdata[0], self.ydata[0])
            self.line2.set_data(self.xdata[1], self.ydata[1])
            return self.line1, self.line2


if __name__ == "__main__":
    ketju_file = bgs.utils.get_ketjubhs_in_dir(
        "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0780"
    )[0]
    lagrangian_file = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/lagrangian_files/kick-vel-0780.txt"

    # start timer
    tstart = datetime.now()

    TL = Trajectory_Langrangian(ketju_file, lagrangian_file, maxN=1.8e6, step=2.5e3)

    ani = animation.FuncAnimation(
        TL.fig, TL, frames=TL.frames, repeat=True, interval=20, repeat_delay=500
    )
    ani.save(os.path.join(bgs.DATADIR, "visualisations/traj_lang.gif"))

    print(f"Animation ran in {datetime.now() - tstart}")

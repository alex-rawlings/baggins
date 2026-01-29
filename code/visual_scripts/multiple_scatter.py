import argparse
import os.path
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from datetime import datetime
from ketjugw.units import pc
import baggins as bgs


class MultipleBHTrajectories:
    def _data_init(self, kf):
        self.num_sims = len(kf)
        self.bhsA = []
        self.bhsB = []
        for _kf in kf:
            bh1, bh2, _ = bgs.analysis.get_bh_particles(_kf)
            bh1, bh2 = bgs.analysis.move_to_centre_of_mass(bh1, bh2)
            self.bhsA.append(bh1)
            self.bhsB.append(bh2)

    def _init_lists(self):
        # print(self.num_sims)
        self.xdataA = [[] for _ in range(self.num_sims)]
        self.ydataA = [[] for _ in range(self.num_sims)]
        self.xdataB = [[] for _ in range(self.num_sims)]
        self.ydataB = [[] for _ in range(self.num_sims)]

    def _find_max_peri_idx(self, desired_peri):
        max_peri_idx = -1
        for bh1, bh2 in zip(self.bhsA, self.bhsB):
            _, peri_idxs = bgs.analysis.find_pericentre_time(bh1, bh2)
            if peri_idxs[desired_peri] > max_peri_idx:
                max_peri_idx = peri_idxs[desired_peri]
        return max_peri_idx

    def _convert_bhs_to_pc(self):
        self.axlim = -1
        for bh1, bh2 in zip(self.bhsA, self.bhsB):
            bh1.x /= pc
            bh2.x /= pc
            bh1.t /= bgs.general.units.Myr
            bh2.t /= bgs.general.units.Myr
            _lim = max(
                np.max(np.abs(bh1.x[:, [0, 2]])), np.max(np.abs(bh2.x[:, [0, 2]]))
            )
            if _lim > self.axlim:
                self.axlim = _lim

    def set_frames(self):
        div_idx = self.max_peri_idx - self.zoom_before_peri
        pre_zoom = np.arange(0, div_idx, self.step_big)
        self.step_small = max(1, int(self.step_big / 30))
        post_zoom = np.arange(pre_zoom[-1], self.maxN, self.step_small)
        self.frames = np.concatenate((pre_zoom, post_zoom))

    def __init__(
        self,
        kf,
        max_n_after_peri=5000,
        desired_peri=4,
        step=100,
        zoom_width=80,
        zoom_before_peri=300,
    ):
        self.step_big = step
        self.zoom_before_peri = zoom_before_peri
        self._data_init(kf)
        self._init_lists()
        self.max_peri_idx = self._find_max_peri_idx(desired_peri=desired_peri)
        self.maxN = self.max_peri_idx + max_n_after_peri
        self._convert_bhs_to_pc()
        self.set_frames()

        # set up the figure
        fig, ax = plt.subplots(figsize=(3, 3))
        self.fig = fig
        self.ax = ax
        self.ax.set_xlim(-self.axlim, self.axlim)
        self.ax.set_ylim(-self.axlim, self.axlim)
        self.ax.set_facecolor("#303030")
        self.ax.set_aspect("equal")
        self.kwargs1 = {
            "lw": 2,
            "alpha": 1,
            "markevery": [-1],
            "marker": "o",
            "mec": ax.get_facecolor(),
            "mew": 0.1,
            "zorder": 1,
            "solid_capstyle": "round",
        }
        self.kwargs2 = copy(self.kwargs1)
        self.cmapper1, _ = bgs.plotting.create_normed_colours(
            0, 1.1 * self.num_sims, cmap="flare"
        )
        self.cmapper2, _ = bgs.plotting.create_normed_colours(
            0, 1.1 * self.num_sims, cmap="crest"
        )
        self._sizebar_kw = dict(
            ax=self.ax, units="pc", color="w", fmt=".0f", location="lower left"
        )
        self._size_bar_length = 0.2 * self.axlim
        self.sizebar = bgs.plotting.draw_sizebar(
            length=self._size_bar_length, **self._sizebar_kw
        )
        self.lineA = []
        self.lineB = []
        for pidx in range(self.num_sims):
            lpA = self.ax.plot([], [], c=self.cmapper1(pidx), **self.kwargs1)[-1]
            lpB = self.ax.plot([], [], c=self.cmapper2(pidx), **self.kwargs2)[-1]
            self.lineA.append(lpA)
            self.lineB.append(lpB)

        # initiate zoom generator
        self.zoom_width = zoom_width
        zoom_half_width = zoom_width / 2
        self.zoomer = self.zoom_in(
            [-zoom_half_width, zoom_half_width], [-zoom_half_width, zoom_half_width], 50
        )
        self._zoom_count = 0
        self._zoomed = False

    def zoom_in(self, xlim, ylim, nframes):
        # define how much the window should zoom in each iteration
        xlim_delta = 0.5 * (np.diff(self.ax.get_xlim()) - np.diff(xlim)) / nframes
        ylim_delta = 0.5 * (np.diff(self.ax.get_ylim()) - np.diff(ylim)) / nframes
        xlim0 = self.ax.get_xlim()
        ylim0 = self.ax.get_ylim()
        sb_redrawn = False
        seq = np.concatenate(
            (np.zeros(int(0.5 * nframes)), np.arange(1, nframes + 1, 1))
        )
        for i in seq:
            self.ax.set_xlim(xlim0[0] + i * xlim_delta, xlim0[1] - i * xlim_delta)
            self.ax.set_ylim(ylim0[0] + i * ylim_delta, ylim0[1] - i * ylim_delta)
            if (
                not sb_redrawn
                and np.abs(np.diff(self.ax.get_xlim())) < 2 * self._size_bar_length
            ):
                self.sizebar.remove()
                self.sizebar = bgs.plotting.draw_sizebar(
                    length=0.2 * self.zoom_width, **self._sizebar_kw
                )
                sb_redrawn = True
            yield self.lineA, self.lineB

    def __call__(self, i):
        print(f"Generating frame {i:07d} of {self.maxN:07d}...              ", end="\r")
        tail = min(i, 100)
        if not self._zoomed and i > self.max_peri_idx - self.zoom_before_peri:
            # zoom in
            rect = Rectangle(
                (-self.zoom_width / 2, -self.zoom_width / 2),
                self.zoom_width,
                self.zoom_width,
                ec="w",
                lw=1,
                fill=None,
            )
            self.ax.add_artist(rect)
            self._init_lists()
            for pidx, (bh1, bh2) in enumerate(zip(self.bhsA, self.bhsB)):
                self.lineA[pidx].set_data(bh1.x[i, 0], bh1.x[i, 2])
                self.lineB[pidx].set_data(bh2.x[i, 0], bh2.x[i, 2])
            while True:
                try:
                    self._zoom_count += self.step_small
                    next(self.zoomer)
                    return self.lineA, self.lineB
                except StopIteration:
                    self._zoomed = True
                    break
        else:
            for pidx, (bh1, bh2) in enumerate(zip(self.bhsA, self.bhsB)):
                self.xdataA[pidx].append(bh1.x[i - self._zoom_count, 0])
                self.ydataA[pidx].append(bh1.x[i - self._zoom_count, 2])
                self.xdataB[pidx].append(bh2.x[i - self._zoom_count, 0])
                self.ydataB[pidx].append(bh2.x[i - self._zoom_count, 2])
                self.lineA[pidx].set_data(
                    self.xdataA[pidx][-tail:], self.ydataA[pidx][-tail:]
                )
                self.lineB[pidx].set_data(
                    self.xdataB[pidx][-tail:], self.ydataB[pidx][-tail:]
                )
            return self.lineA, self.lineB


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiple binary trajectory plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(dest="simdir", help="simulation directory", type=str)
    args = parser.parse_args()

    ketju_files = bgs.utils.get_ketjubhs_in_dir(args.simdir)
    # TODO set this throufh command line
    ketju_files = [
        ketju_files[0],
        ketju_files[1],
        ketju_files[6],
        ketju_files[7],
        ketju_files[2],
    ]

    # start timer
    tstart = datetime.now()
    MBHT = MultipleBHTrajectories(ketju_files)
    ani = animation.FuncAnimation(
        MBHT.fig, MBHT, frames=MBHT.frames, repeat=True, interval=20, repeat_delay=500
    )

    ani.save(os.path.join(bgs.DATADIR, "visualisations/multi_traj.gif"))
    print()
    print(f"Animation ran in {datetime.now() - tstart}")

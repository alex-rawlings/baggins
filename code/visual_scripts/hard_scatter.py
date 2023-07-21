import os.path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from seaborn import color_palette
import ketjugw
import cm_functions as cmf


main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/eccentricity_study/e-090/"
fig_prefix = "e90"
label = r"$e_0=0.90$"

data_dirs = [
    os.path.join(main_path, "100K/D_100K_a-D_100K_c-3.720-0.279"),
    os.path.join(main_path, "100K/D_100K_b-D_100K_e-3.720-0.279"),
]


class HardScatter:
    def _data_init(self, i):
        b1, b2, *_ = cmf.analysis.get_bh_particles(cmf.utils.get_ketjubhs_in_dir(data_dirs[i])[0])
        b1, b2 = cmf.analysis.move_to_centre_of_mass(b1, b2)
        b1.x /= ketjugw.units.pc
        b2.x /= ketjugw.units.pc
        b1.t /= cmf.general.units.Myr
        b2.t /= cmf.general.units.Myr
        return b1, b2


    def _init_lists(self):
        self.xdataA, self.ydataA = [[], [], [], []], [[], [], [], []]
        self.xdataB, self.ydataB = [[], []], [[], []]
        self.xdataC, self.ydataC = [[], []], [[], []]


    def _init_frames(self, N):
        split = int(self.split)
        f1 = np.arange(0, split, 500)
        f2 = np.arange(split, N, 4)
        return np.concatenate((f1, f2))


    def __init__(self, max_N, split) -> None:
        # set up data
        self.split = split
        self.background_col = "#303030"
        self.bhA1, self.bhA2 = None, None
        self.bhB1, self.bhB2 = None, None
        self.bhA1, self.bhA2 = self._data_init(0)
        self.bhB1, self.bhB2 = self._data_init(1)
        self.max_N = min(
                    int(max_N),
                    min([len(l) for l in (self.bhA1, self.bhA2, self.bhB1, self.bhB2)])
                    )

        self.fig, self.ax = plt.subplot_mosaic(
                                    """
                                    BAC
                                    """,
                                    figsize=(7,3)
        )
        for k in "ABC":
            self.ax[k].set_aspect("equal")
            self.ax[k].set_xlabel(r"$x/\mathrm{pc}$")
            self.ax[k].set_ylabel(r"$z/\mathrm{pc}$")
            self.ax[k].set_facecolor(self.background_col)
        self.ax["A"].set_xlim(-700, 1200)
        self.ax["A"].set_ylim(-500, 2000)
        for k in "BC":
            self.ax[k].set_xlim(-35, 35)
            self.ax[k].set_ylim(-40, 40)


        self._init_lists()
        self.line1 = dict.fromkeys(self.ax, None)
        self.line2 = dict.fromkeys(self.ax, None)

        col_list = color_palette("icefire", 50).as_hex()

        for k in self.ax.keys():
            self.line1[k], = self.ax[k].plot([], [], lw=2, markevery=[-1], marker="o", c=col_list[5], mec=self.background_col, mew=0.5)
            self.line2[k], = self.ax[k].plot([], [], lw=2, markevery=[-1], marker="o", c=col_list[-5], mec=self.background_col, mew=0.5)
        self.rectangle = Rectangle(
                                (self.ax["B"].get_xlim()[0], self.ax["B"].get_ylim()[0]),
                                np.diff(self.ax["B"].get_xlim())[0],
                                np.diff(self.ax["B"].get_ylim())[0],
                                lw=0.5, ec="w", fc="none", zorder=10
                                )
        self.frames = self._init_frames(self.max_N)


    def __call__(self, i) -> Any:
        if i == 0:
            self._init_lists()
        else:
            self.ax["A"].patches.pop()
        # uncomment to see frame number
        #self.ax["A"].set_title(i)
        self.fig.suptitle(f"$e_0=0.90$, $t={self.bhA1.t[i]:>5.2f} \,\mathrm{{Myr}}$", fontsize="x-large")
        self.ax["B"].set_title("Realisation A")
        self.ax["C"].set_title("Realisation B")
        for j, bh in enumerate((self.bhA1, self.bhA2, self.bhB1, self.bhB2)):
            self.xdataA[j].append(bh.x[i,0])
            self.ydataA[j].append(bh.x[i,2])
            if j%2==0:
                self.line1["A"].set_data(self.xdataA[j], self.ydataA[j])
                if j==0:
                    self.line1["B"].set_data(self.xdataA[j], self.ydataA[j])
                else:
                    self.line1["C"].set_data(self.xdataA[j], self.ydataA[j])
            else:
                self.line2["A"].set_data(self.xdataA[j], self.ydataA[j])
                if j==1:
                    self.line2["B"].set_data(self.xdataA[j], self.ydataA[j])
                else:
                    self.line2["C"].set_data(self.xdataA[j], self.ydataA[j])
        self.ax["A"].add_patch(self.rectangle)
        return self.line1, self.line2


hd = HardScatter(4.75e4, 4.5e4)


ani = animation.FuncAnimation(hd.fig, hd, frames=hd.frames, repeat=True, interval=50, repeat_delay=500)
ani.save(os.path.join(cmf.DATADIR, "hard_scatter.gif"))

#plt.show()
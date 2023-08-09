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
        self.xdata, self.ydata = [[], [], [], []], [[], [], [], []]


    def _init_frames(self, N):
        split = int(self.split)
        f1 = np.arange(0, split, 250)
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
                                    BC
                                    """,
                                    figsize=(5,3)
        )
        self.ax["b"] = self.ax["B"].inset_axes([0.007, 0.565, 0.3, 0.454])
        self.ax["c"] = self.ax["C"].inset_axes([0.007, 0.565, 0.3, 0.454])
        for k in "bcBC":
            self.ax[k].set_aspect("equal")
            self.ax[k].set_facecolor(self.background_col)
            self.ax[k].set_xticklabels([])
            self.ax[k].set_yticklabels([])
            self.ax[k].set_xticks([])
            self.ax[k].set_yticks([])
        for k in "bc":
            for edge in self.ax[k].spines.keys():
                self.ax[k].spines[edge].set_color("w")
            self.ax[k].set_xlim(-700, 800)
            self.ax[k].set_ylim(-500, 1800)
            cmf.plotting.draw_sizebar(self.ax[k], 500, "pc", color="w", fmt=".0f", sep=4, location="upper left")
        for k in "BC":
            self.ax[k].set_xlim(-35, 35)
            self.ax[k].set_ylim(-40, 40)
            cmf.plotting.draw_sizebar(self.ax[k], 10, "pc", color="w", fmt=".0f", sep=4)
        self.ax["B"].set_title("Realisation A")
        self.ax["C"].set_title("Realisation B")
        
        self._init_lists()
        self.line1 = dict.fromkeys(self.ax, None)
        self.line2 = dict.fromkeys(self.ax, None)

        col_list = color_palette("icefire", 50).as_hex()
        plotkwargs = dict.fromkeys(self.ax, {"lw":0.5})
        for k in "BC":
            plotkwargs[k] = {"marker":"o", "markevery":[-1], "mec":self.background_col, "mew":0.5, "lw":2}

        for k in self.ax.keys():
            self.line1[k], = self.ax[k].plot([], [], c=col_list[5], **plotkwargs[k])
            self.line2[k], = self.ax[k].plot([], [], c=col_list[-5], **plotkwargs[k])

        self.frames = self._init_frames(self.max_N)


    def __call__(self, i) -> Any:
        if i == 0:
            self._init_lists()
        print(f"Generating frame {i:05d} of {self.max_N}...              ", end="\r")
        # uncomment to see frame number
        #self.ax["B"].set_title(i)
        self.fig.suptitle(f"$e_0=0.90$, $t={self.bhA1.t[i]:>5.2f} \,\mathrm{{Myr}}$", fontsize="x-large")
        for j, bh in enumerate((self.bhA1, self.bhA2, self.bhB1, self.bhB2)):
            self.xdata[j].append(bh.x[i,0])
            self.ydata[j].append(bh.x[i,2])
            if j%2==0:
                if j==0:
                    self.line1["b"].set_data(self.xdata[j], self.ydata[j])
                    self.line1["B"].set_data(self.xdata[j], self.ydata[j])
                else:
                    self.line1["c"].set_data(self.xdata[j], self.ydata[j])
                    self.line1["C"].set_data(self.xdata[j], self.ydata[j])
            else:
                if j==1:
                    self.line2["b"].set_data(self.xdata[j], self.ydata[j])
                    self.line2["B"].set_data(self.xdata[j], self.ydata[j])
                else:
                    self.line2["c"].set_data(self.xdata[j], self.ydata[j])
                    self.line2["C"].set_data(self.xdata[j], self.ydata[j])
        return self.line1, self.line2


if __name__ == "__main__":
    hd = HardScatter(4.75e4, 4.5e4)


    ani = animation.FuncAnimation(hd.fig, hd, frames=hd.frames, repeat=True, interval=20, repeat_delay=500)
    ani.save(os.path.join(cmf.DATADIR, "hard_scatter.gif"))

#plt.show()

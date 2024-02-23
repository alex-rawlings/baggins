from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pygad
import ketjugw
import cm_functions as cmf


__all__ = ["OverviewAnimation", "SMBHtrajectory"]


class SMBHAnimation:
    def __init__(
        self, ax, show_axis_labels=True, stepping={"start": 110000, "step": 500}
    ) -> None:
        # TODO "parent" animation that takes a series of potential trajectories to animate across potentially multiple plotting axes
        # TODO take most of the methodology from SMBHtrajectory, and wrap those attributes into this class
        # TODO this class should construct the step_gen function and the __call__ function
        self.ax = ax
        if isinstance(self.ax, np.ndarray):
            self.ax = np.concatenate(self.ax).flatten()
        self.show_axis_labels = show_axis_labels
        self.stepping = stepping


class SMBHtrajectory:
    def __init__(
        self,
        bhdata,
        ax,
        centre=1,
        axes=[0, 2],
        axis_offset=1,
        trails=5000,
        show_axis_labels=True,
        stepping={"start": 110000, "step": 500},
        only_bound=False,
        fix_centre=None,
    ):
        """
        Create an animation of the BH motions

        Parameters
        ----------
        bhdata : str
            path to the ketju_bhs.hdf5 to read in
        ax : matplotlib.axes.Axes
            axis to plot to
        centre : int, optional
            which BH to centre on, by default 1
        axes : list, optional
            axes to plot (0:x, 1:y, 2:z), by default [0,2]
        axis_offset : float, optional
            figure will have centre +/- this value in the window, by default 1
        trails : int, optional
            draw a trail of this length behind the BH, by default 5000
        show_axis_labels : bool, optional
            show the axis labels , by default True
        stepping : dict, optional
            TODO change to times instead of index
            frame to start with and the stepping between frames
            by default {"start":110000, "step":500}
        only_bound : bool, optional
            animation only for when the BHs are bound?, by default False
        """
        kpc = ketjugw.units.pc * 1e3
        myr = ketjugw.units.yr * 1e6
        if only_bound:
            bh1, bh2, merged = cmf.analysis.get_bound_binary(bhdata)
        else:
            bh1, bh2, merged = cmf.analysis.get_bh_particles(bhdata)
        self.com = (bh1.m[:, np.newaxis] * bh1.x + bh2.m[:, np.newaxis] * bh2.x) / (
            bh1.m + bh2.m
        )[:, np.newaxis]
        self.com /= kpc
        self.length = len(bh1.t)
        self.stepping = stepping
        self.save_count = int((self.length - stepping["start"]) / stepping["step"]) - 1
        self.ax = ax
        self.centre = centre
        self.bh1x = bh1.x / kpc - self.com
        self.bh2x = bh2.x / kpc - self.com
        self.time = bh1.t / myr
        self.axes = axes
        if show_axis_labels:
            axlabels = ["x/kpc", "y/kpc", "z/kpc"]
            ax.set_xlabel(axlabels[axes[0]])
            ax.set_ylabel(axlabels[axes[1]])
        self.axis_offset = axis_offset
        self.trails = trails
        self.fix_centre = fix_centre
        (self.traj1,) = self.ax.plot([], [], markevery=[-1], marker="o")
        (self.traj2,) = self.ax.plot([], [], markevery=[-1], marker="o")

    def __call__(self, i):
        """Update the figure with the ith frame number"""
        centre_idx = i if self.fix_centre is None else self.fix_centre
        if self.centre == -1:
            centre = self.com[centre_idx, self.axes]
        elif self.centre == 1:
            centre = self.bh1x[centre_idx, self.axes]
        else:
            centre = self.bh2x[centre_idx, self.axes]
        self.ax.set_xlim(centre[0] - self.axis_offset, centre[0] + self.axis_offset)
        self.ax.set_ylim(centre[1] - self.axis_offset, centre[1] + self.axis_offset)
        self.ax.set_title(f"{self.time[i]:.3f} Myr")
        self.ax.figure.canvas.draw()  # update the changes to the canvas
        if i == 0:
            # initialise the plot
            self.traj1.set_data([], [])
            self.traj2.set_data([], [])
            return self.traj1, self.traj2
        line_end = max(0, i - self.trails)
        self.traj1.set_data(
            self.bh1x[line_end:i, self.axes[0]],
            self.bh1x[line_end:i, self.axes[1]],
        )
        self.traj2.set_data(
            self.bh2x[line_end:i, self.axes[0]],
            self.bh2x[line_end:i, self.axes[1]],
        )
        return self.traj1, self.traj2

    def step_gen(self):
        """generate the stepping between frames"""
        for cnt in itertools.count(
            start=self.stepping["start"], step=self.stepping["step"]
        ):
            print(
                f"Creating image sequence: {cnt/(self.length-1)*100:.3f}%                         ",
                end="\r",
            )
            yield cnt


class OverviewAnimation:
    def __init__(
        self, snaplist, fig, ax, centre=None, axis_offsets={"stars": 500, "dm": 1000}
    ):
        """
        Create an animation of pygad snapshots

        Parameters
        ----------
        snaplist : list
            list of ordered snapshots to plot (note no reordering is done)
        fig : matplotlib.figure.Figure
            figure object
        ax : matplotlib.axes.Axes
            axis to plot to
        centre : str, optional
            centring of animation, options are:
                - None: centre at [0,0,0] origin
                - big: centre at more massive BH position (for isolated runs
                       this will be just the BH position)
                -small: centre at less massive BH position,
            by default None
        axis_offsets : dict, optional
            window extent about centre for each particle family, by default
            {"stars":500, "dm":1000}
        """
        self.snaplist = snaplist
        self.fig = fig
        self.ax = ax
        self.image = None
        self.length = len(snaplist)
        self.savecount = self.length - 1
        self.vlims = {"stars": [100, -100], "dm": [100, -100]}
        self.centre = centre
        self.axis_offset = axis_offsets
        self.extent = None
        self.get_vlims()

    def step_gen(self, start=0):
        """generate the stepping between frames"""
        for cnt in itertools.count(start=start):
            print(
                f"Creating image sequence: {cnt/(self.length-1)*100:.3f}%                         ",
                end="\r",
            )
            yield cnt

    def get_extent(self):
        self.extent = {}
        if self.centre is None:
            origin = [0, 0, 0]
        elif self.centre == "big":
            origin = self.snap.bh["pos"][np.argmax(self.snap.bh["ID"]), :]
        elif self.centre == "small":
            origin = self.snap.bh["pos"][np.argmin(self.snap.bh["ID"]), :]
        else:
            raise ValueError("centre must be None, 'big', or 'small'")
        for family, offset in self.axis_offset.items():
            self.extent[family] = {}
            for projname, proj in zip(("xz", "xy"), ([0, 2], [0, 1])):
                self.extent[family][projname] = [
                    [origin[proj[0]] - offset, origin[proj[0]] + offset],
                    [origin[proj[1]] - offset, origin[proj[1]] + offset],
                ]

    def get_vlims(self):
        """
        Loop through all images to ensure consistent colour scheme
        throughout animation
        """
        # set up a temporary axis, as we don't want to save any plots here
        tempfig, tempax = plt.subplots(2, 2)
        for i in range(self.length):
            print(
                f"Getting vlims: {i/(self.length-1)*100:.3f}%                         ",
                end="\r",
            )
            self.snap = pygad.Snapshot(self.snaplist[i], physical=True)
            # determine the extents of the axis
            self.get_extent()
            _, _, imtemp = cmf.plotting.plot_galaxies_with_pygad(
                self.snap, return_ims=True, figax=[tempfig, tempax], extent=self.extent
            )
            for ind, family in enumerate(self.vlims.keys()):
                self.vlims[family][0] = np.min(
                    [
                        self.vlims[family][0],
                        np.min(imtemp[0 + ind].get_array()),
                        np.min(imtemp[2 + ind].get_array()),
                    ]
                )
                self.vlims[family][1] = np.max(
                    [
                        self.vlims[family][1],
                        np.max(imtemp[0 + ind].get_array()),
                        np.max(imtemp[2 + ind].get_array()),
                    ]
                )
        for family in self.vlims.keys():
            self.vlims[family][0] = floor(self.vlims[family][0])
            self.vlims[family][1] = ceil(self.vlims[family][1])
        print("vlims have been set...                         ")

    # TODO allow for a zoom in mid-animation

    def __call__(self, i):
        """Update the figure with the ith frame number"""
        for ax in np.concatenate(self.ax).flat:
            ax.clear()
        self.snap = pygad.Snapshot(self.snaplist[i], physical=True)
        # determine the extents of the axis
        self.get_extent()
        # plot
        if i == 0:
            _, _, self.image = cmf.plotting.plot_galaxies_with_pygad(
                self.snap,
                return_ims=True,
                figax=[self.fig, self.ax],
                extent=self.extent,
            )
        else:
            _, _, self.image = cmf.plotting.plot_galaxies_with_pygad(
                self.snap,
                return_ims=True,
                figax=[self.fig, self.ax],
                extent=self.extent,
                kwargs={"showcbar": False},
                append_kwargs=True,
            )
        for ax in np.concatenate(self.ax).flat:
            ax.figure.canvas.draw()  # update the changes to the canvas
        return self.image

from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pygad
import ketjugw
import cm_functions as cmf


__all__ = ["OverviewAnimation", "SMBHtrajectory"]


class SMBHtrajectory:
    """
    Create an animation of the BH motions
    """
    def __init__(self, bhdata, ax, centre=1, axes=[0,2], axis_offset=1, trails=5000, show_axis_labels=True, stepping={"start":110000, "step":500}):
        """
        Initialisation

        Parameters
        ----------
        bhdata: path to the ketju_bhs.hdf5 to read in
        ax: pyplot axis to plot the figure on (must be created externally)
        centre: [1,2] which BH to centre on
        axes: list, the axes to plot (0:x, 1:y, 2:z)
        axis_offset: figure will have centre +/- this value in the window
        trails: draw a trail of this length behind the BH
        show_axis_labels: show the axis labels 
        stepping: dict, with the frame to start with and the stepping between
                  frames
        """
        kpc = ketjugw.units.pc * 1e3
        myr = ketjugw.units.yr * 1e6
        bhs = ketjugw.data_input.load_hdf5(bhdata)
        bh1, bh2 = bhs.values()
        self.length = len(bh1.t)
        self.stepping = stepping
        self.save_count = int((self.length - stepping["start"])/stepping["step"])-1
        self.ax = ax
        self.centre = centre
        self.bh1x = bh1.x/kpc
        self.bh2x = bh2.x/kpc
        self.time = bh1.t/myr
        self.axes = axes
        if show_axis_labels:
            axlabels = ["x/kpc", "y/kpc", "z/kpc"]
            ax.set_xlabel(axlabels[axes[0]])
            ax.set_ylabel(axlabels[axes[1]])
        self.axis_offset = axis_offset
        self.trails = trails
        self.traj1, = self.ax.plot([], [], markevery=[-1], marker="o")
        self.traj2, = self.ax.plot([], [], markevery=[-1], marker="o")
    
    def __call__(self, i):
        """Update the figure with the ith frame number"""
        if self.centre == 1:
            centre = self.bh1x[i, self.axes]
        else:
            centre = self.bh2x[i, self.axes]
        self.ax.set_xlim(
            centre[0] - self.axis_offset, 
            centre[0] + self.axis_offset
        )
        self.ax.set_ylim(
            centre[1] - self.axis_offset, 
            centre[1] + self.axis_offset
        )
        self.ax.set_title("{:.3f} Myr".format(self.time[i]))
        self.ax.figure.canvas.draw() #update the changes to the canvas
        if i == 0:
            #initialise the plot
            self.traj1.set_data([], [])
            self.traj2.set_data([], [])
            return self.traj1, self.traj2
        line_end = max(0, i-self.trails)
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
        for cnt in itertools.count(start=self.stepping["start"], step=self.stepping["step"]):
            print("Creating image sequence: {:.3f}%                         ".format(cnt/(self.length-1)*100), end="\r")
            yield cnt


class OverviewAnimation:
    """
    Create an animation of pygad snapshots
    """
    def __init__(self, snaplist, fig, ax, centre=None, axis_offsets={"stars":500, "dm":1000}):
        """
        Initialisation

        Parameters
        ----------
        snaplist: list of ordered snapshots to plot (note no reordering
                  is done)
        fig: pyplot figure object
        ax: pyplot axis object, must be an ndarray of shape=(2,2)
        centre: centring of animation, options are:
                - None: centre at [0,0,0] origin
                - big: centre at more massive BH position (for isolated runs
                       this will be just the BH position)
                -small: centre at less massive BH position
        
        """
        self.snaplist = snaplist
        self.fig = fig
        self.ax = ax
        self.image = None
        self.length = len(snaplist)
        self.savecount = self.length - 1
        self.vlims = {"stars": [100, -100], "dm":[100, -100]}
        self.centre = centre
        self.axis_offset = axis_offsets
        self.extent = None
        self.get_vlims()

    def step_gen(self, start=0):
        """generate the stepping between frames"""
        for cnt in itertools.count(start=start):
            print("Creating image sequence: {:.3f}%                         ".format(cnt/(self.length-1)*100), end="\r")
            yield cnt
    
    def get_extent(self):
        self.extent = {}
        if self.centre is None:
            origin = [0,0,0]
        elif self.centre == "big":
            origin = self.snap.bh["pos"][np.argmax(self.snap.bh["ID"]),:]
        elif self.centre == "small":
            origin = self.snap.bh["pos"][np.argmin(self.snap.bh["ID"]),:]
        else:
            raise ValueError("centre must be None, 'big', or 'small'")
        for family, offset in self.axis_offset.items():
            self.extent[family] = {}
            for projname, proj in zip(("xz", "xy"), ([0,2], [0,1])):
                self.extent[family][projname] = [
                    [origin[proj[0]]-offset, origin[proj[0]]+offset], 
                    [origin[proj[1]]-offset, origin[proj[1]]+offset]
                    ]

    def get_vlims(self):
        """
        Loop through all images to ensure consistent colour scheme
        throughout animation
        """
        #set up a temporary axis, as we don't want to save any plots here
        tempfig, tempax = plt.subplots(2,2)
        for i in range(self.length):
            print("Getting vlims: {:.3f}%                         ".format(i/(self.length-1)*100), end="\r")
            self.snap = pygad.Snapshot(self.snaplist[i], physical=True)
            #determine the extents of the axis
            self.get_extent()
            _,_,imtemp = cmf.plotting.plot_galaxies_with_pygad(self.snap, return_ims=True, figax=[tempfig,tempax], extent=self.extent)
            for ind, family in enumerate(self.vlims.keys()):
                self.vlims[family][0] = np.min([
                    self.vlims[family][0],
                    np.min(imtemp[0+ind].get_array()),
                    np.min(imtemp[2+ind].get_array())
                ])
                self.vlims[family][1] = np.max([
                    self.vlims[family][1],
                    np.max(imtemp[0+ind].get_array()),
                    np.max(imtemp[2+ind].get_array())
                ])
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
        #determine the extents of the axis
        self.get_extent()
        #plot
        if i == 0:
            _,_,self.image = cmf.plotting.plot_galaxies_with_pygad(self.snap, return_ims=True, figax=[self.fig, self.ax], extent=self.extent)
        else:
            _,_,self.image = cmf.plotting.plot_galaxies_with_pygad(self.snap, return_ims=True, figax=[self.fig, self.ax], extent=self.extent, kwargs={"showcbar":False}, append_kwargs=True)
        for ax in np.concatenate(self.ax).flat:
            ax.figure.canvas.draw() #update the changes to the canvas
        return self.image

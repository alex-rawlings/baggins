import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import itertools
import pygad
import ketjugw
import cm_functions as cmf


__all__ = ["OverviewAnimation", "overview_animation", "SMBHtrajectory"]


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
            centre = self.bh1x[i, self.axes]
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
    def __init__(self, snaplist, fig, ax, orientate=None):
        self.snaplist = snaplist
        self.fig = fig
        self.ax = ax
        self.image = None
        self.orientate = orientate
        self.counter = 0
    
    def __call__(self):
        print("Reading: {}".format(self.snaplist[self.counter]))
        snap = pygad.Snapshot(self.snaplist[self.counter])
        snap.to_physical_units()
        figax = [self.fig, self.ax]
        _,_,self.image,*_ = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=self.orientate, figax=figax)
        self.counter += 1
        yield self.image


#def generate_snapshot_image_sequence(snaplist):



def overview_animation(snaplist, figax, orientate=None):
    """
    Generate an animation based off a list of ordered snapshots. The plot will
    consist of four panels: 
        - top left: stars (x-z)
        - bottom left: stars (x-y)
        - top right: dm (x-z)
        - bottom right: dm (x-y)
    """
    images = []
    for ind, snapfile in enumerate(snaplist):
        print("Reading: {}".format(snapfile))
        snap = pygad.Snapshot(snapfile)
        snap.to_physical_units()
        _,_,image = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=orientate, figax=figax)
        if ind == 0:
            cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=orientate, figax=figax)
        """time = cmf.general.convert_gadget_time(snap, new_unit="Myr")
        if ind == 0:
            fig, ax, image = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=orientate, figax=None)
        else:
            figax = [fig, ax]
            _,_,image = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=orientate, figax=figax)"""
        
        images.append(image)
    ani = matplotlib.animation.ArtistAnimation(figax[0], images, interval=50, blit=False)
    plt.show()


def generate_images(gal):
    """
    Generate the top and side views of the galaxy. This code is largely based
    off Matias' implementation.

    Parameters
    ----------


    Returns
    -------
    """
    #initialise lists
    imgs_top = []
    imgs_side = []
    times = []
    #plot the snapshot


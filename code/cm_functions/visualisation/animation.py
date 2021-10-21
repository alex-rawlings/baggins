import matplotlib.pyplot as plt
import matplotlib.animation
import pygad
import cm_functions as cmf


__all__ = ["OverviewAnimation", "overview_animation"]


class OverviewAnimation:
    def __init__(self, snaplist, fig, ax, orientate=None):
        self.snaplist = snaplist
        self.fig = fig
        self.ax = ax
        self.image = []
        self.orientate = orientate
        self.counter = 0
    
    def __call__(self):
        print("Reading: {}".format(self.snaplist[self.counter]))
        snap = pygad.Snapshot(self.snaplist[self.counter])
        snap.to_physical_units()
        if self.counter == 0:
            figax = None
        else:
            figax = [self.fig, self.ax]
        _,_,self.image = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=self.orientate, figax=figax)
        self.counter += 1
        yield self.image


#def generate_snapshot_image_sequence(snaplist):



def overview_animation(snaplist, orientate=None):
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
        time = cmf.general.convert_gadget_time(snap, new_unit="Myr")
        if ind == 0:
            fig, ax, image = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=orientate, figax=None)
        else:
            figax = [fig, ax]
            _,_,image = cmf.plotting.plot_galaxies_with_pygad(snap, return_ims=True, orientate=orientate, figax=figax)
        
        images.append(image)
    ani = matplotlib.animation.ArtistAnimation(fig, images, interval=50, blit=True)
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



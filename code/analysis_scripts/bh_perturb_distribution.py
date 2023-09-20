import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pygad
import cm_functions as cmf


#set up command line arguments
parser = argparse.ArgumentParser(description="Determine the deviation of the BH from the CoM", allow_abbrev=False)
parser.add_argument(type=str, help="path to snapshot directory or previous dataset", dest="path")
parser.add_argument("-p", "--path2", help="path to second .pickle file for plotting purposes", type=str, dest="path2", action="append")
parser.add_argument("-t", "--time", type=float, help="plot times after this [Gyr]", dest="time", default=-1)
# TODO better handling of all snaps
parser.add_argument("-l", "--lastsnap", help="last snap number to analyse", type=int, dest="last_snap", default=9999)
parser.add_argument("-v", "--verbose", help="verbose printing", dest="verbose", action="store_true")
args = parser.parse_args()

new_data = False if args.path.endswith(".pickle") else True


# simple class to hold data
class Brownian:
    def __init__(self, bhid, n, verbose=False) -> None:
        self.bhid = bhid
        self.n = n
        self.count = 0
        self.verbose = verbose
        self.times = np.full(self.n, np.nan, dtype=float)
        self.x_offset = np.full((self.n, 3), np.nan, dtype=float)
        self.v_offset = np.full_like(self.x_offset, np.nan, dtype=float)
        self.id_mask_bh = pygad.IDMask(self.bhid)
        self.x_offset_mag = np.full_like(self.times, np.nan, dtype=float)
        self.v_offset_mag = np.full_like(self.times, np.nan, dtype=float)
        self.bh_sep = np.full_like(self.times, np.nan, dtype=float)
    
    @property
    def count(self):
        return self._count
    
    @count.setter
    def count(self, v):
        assert 0 <= v <= self.n
        self._count  = v

    def add_data(self, snap, xcom=None, vcom=None):
        if self.verbose: print(f"{self.count}: {self.bhid}")
        assert snap.phys_units_requested
        self.times[self.count] = cmf.general.convert_gadget_time(snap)
        id_masks_stars = cmf.analysis.get_all_id_masks(snap)
        if xcom is None:
            xcom = cmf.analysis.get_com_of_each_galaxy(snap, method="ss", masks=id_masks_stars, family="stars", verbose=self.verbose)
        if vcom is None:
            vcom = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcom, masks=id_masks_stars, verbose=self.verbose)
        self.x_offset[self.count, :] = xcom[self.bhid] - snap.bh[self.id_mask_bh]["pos"]
        self.v_offset[self.count, :] = vcom[self.bhid] - snap.bh[self.id_mask_bh]["vel"]
        self.bh_sep[self.count] = pygad.utils.geo.dist(snap.bh["pos"][0,:], snap.bh["pos"][1,:])
        self.count += 1
        if self.count == self.n:
            self.compute_offset_magnitude()
        return xcom, vcom
    
    def compute_offset_magnitude(self):
        self.x_offset_mag = cmf.mathematics.radial_separation(self.x_offset)
        self.v_offset_mag = cmf.mathematics.radial_separation(self.v_offset)
    
    def save(self, fname=None):
        data_dict = dict(
            times = self.times,
            x_offset = self.x_offset,
            x_offset_mag = self.x_offset_mag,
            v_offset = self.v_offset,
            v_offset_mag = self.v_offset_mag,
            bh_sep = self.bh_sep
        )
        if fname is None:
            fname = f"my_galaxy_{self.bhid}.pickle"
        else:
            fpre, fext = os.path.splitext(fname)
            assert fext == ".pickle"
            fname = fpre + f"_{self.bhid}" + fext
        savepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"pickle/bh_perturb_merger/{fname}")
        cmf.utils.save_data(data_dict, savepath)
    
    @classmethod
    def load(cls, fname):
        c = cls(None, 0)
        data = cmf.utils.load_data(fname)
        for k, v in data.items():
            setattr(c, k, v)
        return c

    def plot(self, ax=None, xval="time", **kwargs):
        if xval == "time":
            xlabel = "Time / Gyr"
            xs = self.times
        elif xval=="sep":
            xlabel = "Separation / kpc"
            xs = self.bh_sep
        else:
            raise ValueError("xval must be 'times' or 'sep'")
        sc_kw = {"linewidth":0.5, "edgecolor":"k"}
        if ax is None:
            newax = True
            fig, ax = plt.subplots(1,3, figsize=(10,4))
            ax[0].set_xlabel(xlabel)
            ax[0].set_ylabel("BH position offset [kpc]")
            ax[1].set_xlabel(xlabel)
            ax[1].set_ylabel("BH velocity offset [km/s]")
            ax[2].set_xlabel("BH position offset [kpc]")
            ax[2].set_ylabel("BH velocity offset [km/s]")
            fig.suptitle("BH Brownian Motion")
        else:
            newax = False
        ax[0].scatter(xs, self.x_offset_mag, **sc_kw, **kwargs)
        ax[1].scatter(xs, self.v_offset_mag, **sc_kw, **kwargs)
        ax[2].scatter(self.x_offset_mag, self.v_offset_mag, **sc_kw, **kwargs)
        if newax:
            return ax, fig
        else:
            return ax


brownian_objects = []

if new_data:
    snapfiles = cmf.utils.get_snapshots_in_dir(args.path)

    for ind, snapfile in enumerate(snapfiles):
        if ind <= args.last_snap:
            snap = pygad.Snapshot(snapfile, physical=True)
            if ind==0:
                for bhid in snap.bh["ID"]:
                    brownian_objects.append(Brownian(bhid, len(snapfiles)))
            for ind2, b in enumerate(brownian_objects):
                if ind2==0:
                    xcom, vcom = b.add_data(snap)
                else:
                    b.add_data(snap, xcom=xcom, vcom=vcom)
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
    for b in brownian_objects:
        b.compute_offset_magnitude()
        b.save()

else:
    print("Previous data set being read!")
    brownian_objects.append(Brownian.load(args.path))
    if args.path2 is not None:
        for p in args.path2:
            brownian_objects.append(Brownian.load(p))
    for b in brownian_objects:
        b.compute_offset_magnitude()


for i, b in enumerate(brownian_objects):
    if i==0:
        ax, *_ = b.plot()
    else:
        ax = b.plot(ax=ax)
for i in range(2):
    ax[i].axvline(1.71071, c="k", alpha=0.7, ls=":", label=r"$t_\mathrm{peri}$")
ax[0].legend()

plt.show()

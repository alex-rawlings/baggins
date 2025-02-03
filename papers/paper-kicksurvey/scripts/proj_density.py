import argparse
import os.path
from datetime import datetime
import numpy as np
from scipy.ndimage import uniform_filter, generic_filter
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import pygad
import h5py
import baggins as bgs
import dask
import figure_config

bgs.plotting.check_backend()

parser = argparse.ArgumentParser(
    description="Plot projected density image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-d",
    "--data",
    dest="data",
    type=str,
    help="(list of) snapshot(s) to aanlyse or plot",
    default="/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output/snap_007.hdf5",
)
parser.add_argument(
    "--prominence-only",
    help="only calculate prominences",
    action="store_true",
    dest="prom_only",
)
parser.add_argument("-z", dest="redshift", help="redshift", type=float, default=0.3)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    default="INFO",
    choices=bgs.VERBOSITY,
    dest="verbosity",
    help="set verbosity level",
)
args = parser.parse_args()

SL = bgs.setup_logger("script", args.verbosity)


# class to facilitate different use cases
class ProjectedDensityObject:
    def __init__(self, redshift, plot=False):
        self._redshift = redshift
        self._plot = plot
        self._snapfiles = None
        self.ang_scale = bgs.cosmology.angular_scale(args.redshift)
        SL.info(f"Angular scale is {self.ang_scale:.4f} kpc/arcsec")
        self.extent = 40
        self.filter_code = "Euclid/NISP.Y"
        self.galaxy_metallicity = 0.012
        self.galaxy_star_age = 7.93e9  # yr
        self._save_location = None
        self.max_bh_dist = 30

        # plotting attributes
        self.ax = None
        self.xaxis = 0
        self.yaxis = 2
        self._fontsize = 12

    @property
    def single_snapshot(self):
        return self._single_snapshot

    @property
    def save_location(self):
        return self._save_location

    @save_location.setter
    def save_location(self, v):
        self._save_location = v

    @classmethod
    def load_single_snapshot(cls, snapfile, redshift):
        C = cls(redshift=redshift)
        C._plot = True
        C._snapfiles = [snapfile]
        C.setup_plot()
        C._single_snapshot = True
        return C

    @classmethod
    def load_snapshot_list(cls, snapdir, redshift, saveloc):
        C = cls(redshift=redshift)
        C._snapfiles = bgs.utils.get_snapshots_in_dir(snapdir)
        assert os.path.splitext(saveloc)[1] == ".pickle"
        C.save_location = saveloc
        C._single_snapshot = False
        return C

    def get_num_pixels(self, angscale, spatial_res=0.101, extent=None):
        """
        Determine the pixel bin width

        Parameters
        ----------
        angscale : float
            angular scale
        spatial_res : float, optional
            spatial resolution of detector in kpc/", by default 0.101
        extent : int, optional
            spatial extent of image in kpc, by default 40

        Returns
        -------
        : float
            pixel bin wodth
        """
        if extent is None:
            extent = self.extent
        return int(extent / (spatial_res * angscale))

    def setup_plot(self):
        fig, self.ax = plt.subplots(1, 3)
        fig.set_figwidth(3 * fig.get_figwidth())

    def add_extras(self, ax, pos):
        # mark BH position
        ax.annotate(
            r"$\mathrm{bound\;cluster}$",
            (pos[0, 0] + 0.25, pos[0, 2] - 0.25),
            (pos[0, 0] + 12, pos[0, 2] - 8),
            color="w",
            arrowprops={"fc": "w", "ec": "w", "arrowstyle": "wedge"},
            ha="right",
            va="bottom",
            fontsize=self._fontsize,
        )
        ax.set_facecolor("k")

    def run(self):
        data_to_save = {}
        prev_bh_dist = -99

        # calculations that needn't be done each iteration
        density_mask = pygad.ExprMask("abs(pos[:,0]) <= 25") & pygad.ExprMask(
            "abs(pos[:,2]) <= 25"
        )
        synth_grid, synth_SED = bgs.analysis.get_spectrum_ssp(
            age=self.galaxy_star_age, metallicity=self.galaxy_metallicity
        )
        euclid_filters = bgs.analysis.get_euclid_filter_collection(synth_grid)

        for snapnum, snapfile in enumerate(self._snapfiles):
            # load and centre the snapshot
            snap = self.load_and_centre_snap(snapfile=snapfile)

            if len(snap.bh) > 1:
                SL.warning("BHs have not yet merged! Skipping this snapshot")
                # clean memory
                snap.delete_blocks()
                del snap
                pygad.gc_full_collect()
                continue

            # check to see if we have reached apocentre: if so, break
            current_bh_dist = pygad.utils.geo.dist(snap.bh["pos"][0, :])
            if current_bh_dist < prev_bh_dist or current_bh_dist > self.max_bh_dist:
                SL.warning("We have reached apocentre! Stopping")
                bgs.utils.save_data(data_to_save, self.save_location)
                break
            else:
                prev_bh_dist = current_bh_dist

            if self.single_snapshot:
                t0 = None
                _snapfiles = bgs.utils.get_snapshots_in_dir(os.path.dirname(snapfile))
                i = 0
                while t0 is None:
                    SL.debug(f"Finding t0: iteration {i}")
                    with h5py.File(_snapfiles[i], "r") as f:
                        if len(f["/PartType5/Masses"]) < 2:
                            _t0snap = pygad.Snapshot(_snapfiles[i], physical=True)
                            t0 = bgs.general.convert_gadget_time(_t0snap)
                            del _t0snap
                    i += 1
                SL.info(
                    f"Snapshot is at time {bgs.general.convert_gadget_time(snap)-t0:.3f} Gyr"
                )

            if self._plot:
                self.plot_surface_mass_density(snap=snap, density_mask=density_mask)

            # convert to magnitudes
            bgs.analysis.set_luminosity(snap=snap, sed=synth_SED, z=self._redshift)

            # now we need to bin the galaxy in the 2D plane,
            # and determine the magnitude for each pixel
            start_time = datetime.now()
            star_count, xedges, yedges = np.histogram2d(
                x=snap.stars[density_mask]["pos"][:, self.xaxis],
                y=snap.stars[density_mask]["pos"][:, self.yaxis],
                bins=self.get_num_pixels(self.ang_scale),
            )
            SL.debug(f"There are {(len(xedges)-1)**2:.2e} bins")

            # helper function to parallelise magnitude calculation
            @dask.delayed
            def parallel_mag_helper(num_stars_in_bin):
                nonlocal snap
                return bgs.analysis.get_magnitudes(
                    sed=synth_SED,
                    stellar_mass=num_stars_in_bin * float(snap.stars["mass"][0]),
                    filters_collection=euclid_filters,
                    filter_code=self.filter_code,
                    z=self._redshift,
                )["app_mag"]

            res = []
            for sc in star_count.flat:
                res.append(parallel_mag_helper(sc))
            res = dask.compute(*res)
            mag_map = np.array(res).reshape(star_count.shape).T
            SL.info(f"Apparent magnitude calculated in {datetime.now()-start_time}")

            if self._plot:
                ax_mag = self.ax[1]
            else:
                # define a dummy axis
                fig, ax_mag = plt.subplots()
            im_mag = ax_mag.imshow(
                mag_map,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap="mako_r",
            )

            if self._plot:
                self.plot_app_magnitude(im_mag=im_mag, snap=snap)
            else:
                plt.close()
                k = f"snap_{snapnum:03d}"
                data_to_save[k] = {}
                data_to_save[k]["snap"] = snapfile
                data_to_save[k]["im_mag"] = im_mag

            # determine a local "prominence"
            self.calculate_prominence(im_mag, snap)

            # clean memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()
        else:
            if not self._plot:
                bgs.utils.save_data(data_to_save, self.save_location)

    def load_and_centre_snap(self, snapfile):
        SL.info(f"Reading snapshot {snapfile}")
        snap = pygad.Snapshot(snapfile, physical=True)
        # move to CoM frame
        pre_ball_mask = pygad.BallMask(5)
        centre = pygad.analysis.shrinking_sphere(
            snap.stars,
            pygad.analysis.center_of_mass(snap.stars),
            30,
        )
        vcom = pygad.analysis.mass_weighted_mean(snap.stars[pre_ball_mask], "vel")
        pygad.Translation(-centre).apply(snap, total=True)
        pygad.Boost(-vcom).apply(snap, total=True)
        return snap

    def calculate_prominence(self, im_mag, snap):
        if isinstance(snap, str):
            snap = self.load_and_centre_snap(snap)

        galaxy_stellar_mass = np.sum(snap.stars[pygad.BallMask(30)]["mass"])
        bh_mass = 10 ** bgs.literature.Sahu19(np.log10(galaxy_stellar_mass))
        rinfl = bgs.literature.Merritt09(bh_mass)
        SL.debug(f"Influence radius is {rinfl:.1e} kpc")
        filter_window = max(
            2
            * min(
                self.get_num_pixels(self.ang_scale, extent=5 * rinfl),
                int(min(im_mag.get_array().shape) / 10),
            ),
            3,
        )
        SL.info(f"Filter window size is {filter_window}")

        # determine the prominence within some aperture
        # prom = np.abs(im_mag.get_array()) / np.abs(uniform_filter(im_mag.get_array(), size=filter_window, mode="nearest"))
        rng = np.random.default_rng()

        def custom_mean(x):
            N = 50
            mean_arr = np.full(N, np.nan)
            for i in range(N):
                mean_arr[i] = np.nanmean(rng.choice(x, x.shape, replace=True))
            return np.nanmedian(mean_arr)

        def custom_std(x):
            N = 50
            std_arr = np.full(N, np.nan)
            for i in range(N):
                std_arr[i] = np.nanstd(rng.choice(x, x.shape, replace=True))
            return np.nanmedian(std_arr)

        filter_kwargs = {"size": filter_window, "mode": "nearest"}
        # prom = (im_mag.get_array() - uniform_filter(im_mag.get_array(), size=filter_window, mode="nearest")) / np.std(im_mag.get_array())
        # prom = (im_mag.get_array() - generic_filter(im_mag.get_array(), custom_mean, **filter_kwargs)) / generic_filter(im_mag.get_array(), custom_std, **filter_kwargs)
        prom = (
            im_mag.get_array() - uniform_filter(im_mag.get_array(), **filter_kwargs)
        ) / generic_filter(im_mag.get_array(), np.std, **filter_kwargs)
        # prom = im_mag.get_array() - median_filter(im_mag.get_array(), **filter_kwargs)
        # prom = (im_mag.get_array() - np.mean(im_mag.get_array())) / np.std(im_mag.get_array())

        if self._plot:
            ax_prom = self.ax[2]
        else:
            # define a dummy axis
            fig, ax_prom = plt.subplots()
        im_SN = ax_prom.imshow(
            prom,
            origin="lower",
            extent=im_mag.get_extent(),
            cmap="vlag",
            norm=CenteredNorm(vcenter=0),
        )
        if self._plot:
            self.plot_prominence_map(im_SN=im_SN, snap=snap)
            bgs.plotting.savefig(
                figure_config.fig_path("density_map.pdf"),
                fig=self.ax[0].get_figure(),
                force_ext=True,
            )
            plt.close()

        signal_prom = bgs.mathematics.get_pixel_value_in_image(
            snap.bh["pos"][0, 0], snap.bh["pos"][0, 2], im_SN
        )[0]
        SL.info(f"S/N at BH position is {signal_prom:.2e}")
        fac = np.sign(signal_prom + 1e-14)
        ecdf = bgs.mathematics.empirical_cdf(fac * im_SN.get_array(), fac * signal_prom)
        x = np.sort(fac * im_SN.get_array().flatten())
        x = x[~np.isnan(x)]
        SL.info(f"This corresponds to approx. the {ecdf:.2f} quantile of total pixels")

    def plot_surface_mass_density(self, snap, density_mask, add_extras=True):
        # figure 1: easy, surface mass density
        SL.debug("Plotting surface mass density")
        _, _, im, _ = pygad.plotting.image(
            snap.stars[density_mask],
            qty="mass",
            surface_dens=True,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            cbartitle=r"$\log_{10}\left(\Sigma/\left(\mathrm{M}_\odot\,\mathrm{kpc}^{-2}\right)\right)$",
            fontsize=self._fontsize,
            outline=None,
            cmap="rocket",
            ax=self.ax[0],
            Npx=self.get_num_pixels(self.ang_scale),
            extent=self.extent,
        )
        if add_extras:
            self.add_extras(ax=self.ax[0], pos=snap.bh["pos"])

    def plot_app_magnitude(self, im_mag, snap, add_extras=False):
        self.ax[1].set_facecolor("k")
        im_mag_array = im_mag.get_array()
        im_mag_array = im_mag_array[np.isfinite(im_mag_array)]
        pygad.plotting.add_cbar(
            ax=self.ax[1],
            cbartitle=r"$\mathrm{App. Mag.}$",
            clim=np.percentile(im_mag_array, [0.1, 99.9]),
            cmap=im_mag.get_cmap(),
            fontcolor="w",
            fontsize=self._fontsize,
        )
        pygad.plotting.make_scale_indicators(
            ax=self.ax[1],
            extent=pygad.UnitArr(
                im_mag.get_extent(), units=snap["pos"].units, subs=snap
            ).reshape((2, 2)),
            fontcolor="w",
            fontsize=self._fontsize,
        )
        if add_extras:
            self.add_extras(ax=self.ax[1], pos=snap.bh["pos"])

    def plot_prominence_map(self, im_SN, snap):
        SL.debug("Plotting S/N map")
        self.ax[2].set_facecolor("k")
        pygad.plotting.make_scale_indicators(
            ax=self.ax[2],
            extent=pygad.UnitArr(
                im_SN.get_extent(), units=snap["pos"].units, subs=snap
            ).reshape((2, 2)),
            fontcolor="k",
            fontsize=self._fontsize,
        )
        pygad.plotting.add_cbar(
            self.ax[2],
            cbartitle=r"$S/N$",
            clim=im_SN.get_clim(),
            cmap=im_SN.get_cmap(),
            fontcolor="k",
            fontsize=self._fontsize,
        )


# define the different use cases
if os.path.isfile(args.data):
    # create images for a single snapshot
    proj_dens = ProjectedDensityObject.load_single_snapshot(
        snapfile=args.data, redshift=args.redshift
    )
    proj_dens.run()
else:
    data_file = os.path.join(
        figure_config.reduced_data_dir,
        "mag-maps",
        f"{args.data.rstrip('/').split('/')[-2]}-magnitude-maps.pickle",
    )
    proj_dens = ProjectedDensityObject.load_snapshot_list(
        snapdir=args.data, redshift=args.redshift, saveloc=data_file
    )
    if args.prom_only:
        data = bgs.utils.load_data(data_file)
        for v in data.values():
            proj_dens.calculate_prominence(**v)
    else:
        proj_dens.run()

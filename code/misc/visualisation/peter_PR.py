import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import dask
import pygad
import baggins as bgs


# class to facilitate different use cases
class PlotObject:
    def __init__(self, snapfile, redshift, logger):
        self._snapfile = snapfile
        self._redshift = redshift
        self._logger = logger
        self.instrument = bgs.analysis.Euclid_VIS(z=redshift)
        self.instrument.max_extent = 40
        self.filter_code = "Euclid/VIS.vis"
        self.galaxy_metallicity = 0.03396487304923489
        self.galaxy_star_age = 3.645e9  # yr
        self.binary_core_radius = 0.58
        self._save_location = None
        self.max_bh_dist = 30
        self._cluster_prom = None

        # plotting attributes
        self.ax = None
        self.xaxis = 0
        self.yaxis = 2
        self._fontsize = 12
        self.im_mag = None

    def make_density_plot(self, ax):
        # calculations that needn't be done each iteration
        density_mask = pygad.ExprMask("abs(pos[:,0]) <= 25") & pygad.ExprMask(
            "abs(pos[:,2]) <= 25"
        )
        synth_grid, synth_SED = bgs.analysis.get_spectrum_ssp(
            age=self.galaxy_star_age, metallicity=self.galaxy_metallicity
        )
        euclid_filters = bgs.analysis.get_euclid_filter_collection(synth_grid)

        snap = self.load_and_centre_snap(snapfile=self._snapfile)

        # convert to magnitudes
        bgs.analysis.set_luminosity(snap=snap, sed=synth_SED, z=self._redshift)

        # now we need to bin the galaxy in the 2D plane,
        # and determine the magnitude for each pixel
        start_time = datetime.now()
        star_count, xedges, yedges = np.histogram2d(
            x=snap.stars[density_mask]["pos"][:, self.xaxis],
            y=snap.stars[density_mask]["pos"][:, self.yaxis],
            bins=self.instrument.number_pixels,
        )
        self._logger.debug(f"There are {(len(xedges)-1)**2:.2e} bins")

        # helper function to parallelise magnitude calculation
        @dask.delayed
        def parallel_mag_helper(num_stars_in_bin):
            nonlocal snap
            return bgs.analysis.get_surface_brightness(
                sed=synth_SED,
                stellar_mass=num_stars_in_bin * float(snap.stars["mass"][0]),
                filters_collection=euclid_filters,
                filter_code=self.filter_code,
                z=self._redshift,
                pixel_size=xedges[1] - xedges[0],
            )["app_mag"]

        res = []
        for sc in star_count.flat:
            res.append(parallel_mag_helper(sc))
        res = dask.compute(*res)
        mag_map = np.array(res).reshape(star_count.shape).T
        self._logger.info(
            f"Apparent magnitude calculated in {datetime.now()-start_time}"
        )
        im_mag = ax.imshow(
            mag_map,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="mako_r",
        )
        self.im_mag = im_mag

        ax.set_facecolor("k")
        ax.text(
            0.05,
            0.9,
            f"$z={self._redshift:.1f}$",
            color="w",
            transform=ax.transAxes,
            fontsize=self._fontsize,
        )
        im_mag_array = im_mag.get_array()
        im_mag_array = im_mag_array[np.isfinite(im_mag_array)]
        ax.contour(
            im_mag.get_array(),
            10,
            colors="k",
            linewidths=0.5,
            ls="-",
            extent=im_mag.get_extent(),
            zorder=0.2,
        )
        pygad.plotting.add_cbar(
            ax=ax,
            cbartitle=r"$\mathrm{mag}\,\mathrm{arcsec}^{-2}$",
            clim=np.percentile(im_mag_array, [0.1, 99.9]),
            cmap=im_mag.get_cmap(),
            fontcolor="w",
            fontsize=self._fontsize,
        )
        pygad.plotting.make_scale_indicators(
            ax=ax,
            extent=pygad.UnitArr(
                im_mag.get_extent(), units=snap["pos"].units, subs=snap
            ).reshape((2, 2)),
            fontcolor="w",
            fontsize=self._fontsize,
        )

        # clean memory
        snap.delete_blocks()
        del snap
        pygad.gc_full_collect()
        return im_mag

    def make_ifu_plot(self, ax):
        snap = self.load_and_centre_snap(snapfile=self._snapfile)
        muse_nfm = bgs.analysis.MUSE_NFM()
        muse_nfm.redshift = self._redshift
        seeing_muse = {"num": 25, "sigma": muse_nfm.resolution_kpc.value}
        ifu_mask = muse_nfm.get_fov_mask(0, 2)
        voronoi = bgs.analysis.VoronoiKinematics(
            x=snap.stars[ifu_mask]["pos"][:, 0],
            y=snap.stars[ifu_mask]["pos"][:, 2],
            V=snap.stars[ifu_mask]["vel"][:, 1],
            m=snap.stars[ifu_mask]["mass"],
            Npx=muse_nfm.number_pixels,
            seeing=seeing_muse,
        )
        voronoi.make_grid(part_per_bin=int(800**2))
        voronoi.binned_LOSV_statistics()
        voronoi.plot_kinematic_maps(
                ax=ax,
                moments="2",
                cbar="inset",
                fontsize=self._fontsize,
                cbar_kwargs={"ha":"left"}
            )
        pygad.plotting.make_scale_indicators(
                ax=ax,
                extent=pygad.UnitArr(
                    self.im_mag.get_extent(), units=snap["pos"].units, subs=snap
                ).reshape((2,2)),
                fontcolor="k",
                fontsize=self._fontsize,
            )


    def load_and_centre_snap(self, snapfile):
        self._logger.info(f"Reading snapshot {snapfile}")
        snap = pygad.Snapshot(snapfile, physical=True)
        # move to CoM frame
        bgs.analysis.basic_snapshot_centring(snap)
        return snap


if __name__ == "__main__":
    snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0540/output/snap_009.hdf5"
    SL = bgs.setup_logger("script", "INFO")
    REDSHIFT = 0.6

    fig, ax = plt.subplots(2, 1)
    fig.set_figheight(1.6 * fig.get_figheight())
    po = PlotObject(
        snapfile = snapfile,
        redshift=REDSHIFT,
        logger=SL
    )
    po.make_density_plot(ax=ax[0])
    po.make_ifu_plot(ax=ax[1])


    bgs.plotting.savefig("peter.png", save_kwargs={"dpi":600})
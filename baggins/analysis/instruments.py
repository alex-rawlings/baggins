import os
from abc import ABC, abstractmethod
from functools import lru_cache
import numpy as np
from scipy.stats import binned_statistic, gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.units import Unit
from astropy.cosmology import Planck18
from pygad import ExprMask
from unyt import arcsecond, angstrom, Msun, kpc, yr
import synthesizer.particle
from synthesizer.photometry import PhotometryCollection
from synthesizer.emission_models import BimodalPacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.kernel_functions import Kernel
from synthesizer.instruments.photometric_imager import PhotometricImager
from baggins.analysis.obs_helper import (
    get_synthesizer_grid,
    get_filter_collection,
    get_filter_lam_range,
    EUCLID_FILTER_CODES,
    HST_FILTER_CODES,
    JWST_MIRI_FILTER_CODES,
    JWST_NIRCam_FILTER_CODES,
    VLT_FORS2_FILTER_CODES,
)
from baggins.analysis.voronoi import VoronoiKinematics
from baggins.env_config import _cmlogger
from baggins.cosmology import angular_scale
from baggins.mathematics import (
    get_histogram_bin_centres,
    equal_count_bins,
    next_square_root,
)
from baggins.plotting import draw_sizebar

__all__ = [
    "MUSE_NFM",
    "MUSE_WFM",
    "HARMONI_SENSITIVE",
    "HARMONI_BALANCED",
    "HARMONI_SPATIAL",
    "Euclid_VIS",
    "HSTWFC3",
    "JWST_MIRI",
    "JWST_NIRCam",
    "ERIS_IFU",
    "JWST_IFU",
    "MICADO_WFM_LSS",
    "MICADO_NFM_LSS",
    "VLT_FORS2_LSS",
    "ERIS_NIX_NFM_LSS",
    "JWST_LSS",
]

_logger = _cmlogger.getChild(__name__)

# Cap on threads handed to synthesizer's particle spectra extraction, to
# avoid oversubscribing shared/large-core-count nodes when nthreads=-1
# ("all available") is requested.
_MAX_SPECTRA_THREADS = 8


def _resolve_nthreads():
    """Number of threads to use for spectra generation, capped at
    `_MAX_SPECTRA_THREADS`."""
    return min(os.cpu_count() or 1, _MAX_SPECTRA_THREADS)


# If fewer than this fraction of particles have a unique age, deduplicating
# spectra generation (see PhotometricInstrument._generate_particle_photometry_deduped)
# pays off enough over the per-particle approach to bother with the extra
# bookkeeping.
_DEDUP_UNIQUE_AGE_FRACTION = 0.5


def _filter_tailored_lam(lam_lo, lam_hi, n_fine=100, n_coarse=30):
    """
    Build a wavelength grid that is finely sampled across [lam_lo, lam_hi]
    (where an instrument's filters actually have transmission) and coarsely
    sampled outside that range (for the continuum/dust-curve shape). This
    keeps per-particle spectra generation cheap without under-resolving the
    photometry that's actually computed downstream.

    Parameters
    ----------
    lam_lo, lam_hi : float
        wavelength bounds in Angstrom to sample finely
    n_fine : int, optional
        number of linearly-spaced points across [lam_lo, lam_hi], by default 100
    n_coarse : int, optional
        number of log-spaced points on either side of [lam_lo, lam_hi], by
        default 30

    Returns
    -------
    : unyt_array
        wavelength grid, in Angstrom
    """
    fine = np.linspace(lam_lo, lam_hi, n_fine)
    coarse_lo = np.geomspace(100, lam_lo, n_coarse, endpoint=False)
    coarse_hi = np.geomspace(lam_hi, 1e5, n_coarse)[1:]
    return np.sort(np.concatenate([coarse_lo, fine, coarse_hi])) * angstrom


@lru_cache(maxsize=8)
def _cached_synthesizer_grid(grid_name, new_lam_key):
    """Memoized `get_synthesizer_grid`, keyed on (grid_name, new_lam), so
    repeated calls (e.g. across galaxies/instruments sharing the same grid
    and wavelength sampling) don't re-read and re-resample the same grid
    from disk every time."""
    new_lam = np.asarray(new_lam_key) * angstrom
    return get_synthesizer_grid(grid_name=grid_name, new_lam=new_lam)


class BasicInstrument(ABC):
    def __init__(self, fov, sampling, res=None, z=None, max_extent=40):
        """
        Template class for defining basic observation instrument properties

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        sampling : float
            spatial sampling of instrument in arcsec/pixel
        res : float, optional
            angular resolution in arcsec, by default None
        z : float, optional
            redshift of observations, by default None
        max_extent : float, optional
            maximum spatial extent [kpc], by default 40
        """
        self.field_of_view = fov * Unit("arcsec")
        self.sampling = sampling * Unit("arcsec")
        self.angular_resolution = res * Unit("arcsec")
        self._ang_scale = None
        self.max_extent = max_extent * Unit("kpc")
        if z is not None:
            self.redshift = z

    def _param_check(self):
        """Ensure the instrument's redshift (and hence angular scale) has
        been set before any angular/physical unit conversion is used."""
        try:
            assert self._ang_scale is not None
        except AssertionError:
            _logger.exception("Redshift must be set first!", exc_info=True)
            raise RuntimeError

    @property
    def redshift(self):
        """Redshift of the observation."""
        return self._redshift

    @redshift.setter
    def redshift(self, z):
        """
        Set the observation redshift, updating the cached angular scale.

        Parameters
        ----------
        z : float
            redshift to set; values below 1e-3 are clamped to 1e-3 to
            avoid cosmology methods failing at z=0
        """
        if z < 1e-3:
            # protect against cosmology methods failing at z=0
            z = 1e-3
        self._redshift = z
        self._ang_scale = angular_scale(z)

    @property
    def max_extent(self):
        """Maximum spatial extent of the instrument's field of view, in kpc."""
        return self._max_extent

    @max_extent.setter
    def max_extent(self, R):
        """
        Set the maximum spatial extent, attaching kpc units if a bare
        number is given.

        Parameters
        ----------
        R : float, astropy.units.Quantity
            maximum spatial extent; assumed to be in kpc if unitless
        """
        self._max_extent = R
        try:
            self._max_extent.value
        except AttributeError:
            self._max_extent = self._max_extent * Unit("kpc")

    @property
    def ang_scale(self):
        """Angular scale of the observation, in kpc/arcsec."""
        return self._ang_scale

    @property
    def pixel_width(self):
        """Width of a single detector pixel, in kpc."""
        self._param_check()
        return self.sampling * self._ang_scale

    @property
    def resolution_kpc(self):
        """Angular resolution (e.g. PSF FWHM) of the instrument, in kpc."""
        self._param_check()
        return self.angular_resolution * self._ang_scale

    @property
    def extent(self):
        """Spatial extent of the field of view, in kpc, clipped to `max_extent`."""
        self._param_check()
        return min(self.ang_scale * self.field_of_view, self.max_extent)

    @property
    def number_pixels(self):
        """Number of pixels spanning `extent` at this instrument's pixel width."""
        return int(self.extent / self.pixel_width)

    @property
    def name(self):
        """Class name of this instrument."""
        return type(self).__name__

    def __repr__(self):
        """Human-readable summary of this instrument's geometry."""
        return f"{self.name}:\n FoV: {self.field_of_view}\n sampling: {self.sampling}/pix\n angular resolution: {self.angular_resolution}\n pixel width: {self.pixel_width:.3e}\n # pixels: {self.number_pixels}\n extent: {self.extent:.3e}"

    def get_fov_mask(self, xaxis, yaxis):
        """
        Get the field of view mask for the instrument, where x=0, y=1, z=2 as axis coordinates.

        Parameters
        ----------
        xaxis : int
            integer representation of x coordinate
        yaxis : int
            integer representation of y coordinate

        Returns
        -------
        mask : pygad.ExprMask
            mask to select subregion of snapshot
        """
        mask = ExprMask(f"abs(pos[:,{xaxis}]) <= {0.5 * self.extent.value}") & ExprMask(
            f"abs(pos[:,{yaxis}]) <= {0.5 * self.extent.value}"
        )
        return mask

    def _get_LOS_axis(self, xaxis, yaxis):
        """
        Get the LOS axis for an observation, orthogonal to spatial axes.

        Parameters
        ----------
        xaxis : int, str
            spatial x axis of observation
        yaxis : int, str
            spatial y axis of observation

        Returns
        -------
        : int
            LOS axis
        """
        valid_str = "xyz"
        if isinstance(xaxis, str):
            try:
                assert xaxis in valid_str
                xaxis = valid_str.find(xaxis)
            except AssertionError:
                _logger.exception("x axis must 'x', 'y', or 'z'.", exc_info=True)
                raise
        if isinstance(yaxis, str):
            try:
                assert yaxis in valid_str
                yaxis = valid_str.find(yaxis)
            except AssertionError:
                _logger.exception("y axis must 'x', 'y', or 'z'.", exc_info=True)
                raise
        try:
            assert xaxis != yaxis
        except AssertionError:
            _logger.exception("xaxis and yaxis must be different!", exc_info=True)
            raise
        return list(set({0, 1, 2}).difference({xaxis, yaxis}))[0]


# ------------------------------------------------------------------
# Photometric instruments
# ------------------------------------------------------------------


def _robust_asinh_norm(arr):
    """
    Build a percentile-clipped arcsinh colour norm, robust to images that
    are mostly zero-valued (as SPH-kernel-smoothed particle images are
    outside the light distribution) -- a plain linear or log norm over
    such an image either saturates almost all of the signal or floors at
    an arbitrary epsilon. Operating directly on `arr` (rather than a
    pre-transformed display array) means the norm can be attached to the
    image's `imshow` call and read straight off by a colourbar, so the
    colourbar reflects real (flux) values rather than stretched ones.

    Parameters
    ----------
    arr : np.array
        image array (in physical units, e.g. flux) to normalize

    Returns
    -------
    : matplotlib.colors.AsinhNorm
        colour norm with a MAD-based linear width around zero and
        1st/99.5th percentile display bounds
    """
    arr = np.asarray(arr, dtype=float)
    mad = np.nanmedian(np.abs(arr - np.nanmedian(arr)))
    scale = 1.4826 * mad if mad > 0 else (np.nanstd(arr) or 1.0)
    vmin, vmax = np.nanpercentile(arr, [1, 99.5])
    return AsinhNorm(linear_width=scale, vmin=vmin, vmax=vmax)


class PhotometricInstrument(BasicInstrument):
    def __init__(self, fov, sampling, label, res=None, z=None, max_extent=40):
        """
        Base class for instruments that generate photometric (imaging)
        observations via synthesizer.

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        sampling : float
            spatial sampling of instrument in arcsec/pixel
        label : str
            instrument label, used by the underlying synthesizer
            PhotometricImager
        res : float, optional
            angular resolution in arcsec, by default None
        z : float, optional
            redshift of observations, by default None
        max_extent : float, optional
            maximum spatial extent [kpc], by default 40
        """
        super().__init__(fov, sampling, res, z, max_extent)
        self.galaxy = None
        self._filters = None
        self.label = label
        self._instr = None

    @property
    def filters(self):
        """The synthesizer FilterCollection built for this instrument."""
        return self._filters

    @property
    @abstractmethod
    def filter_codes(self):
        """SVO filter codes for this instrument, independent of any grid --
        used to tailor the SSP grid's wavelength sampling to this
        instrument's filters before the grid itself is built."""
        raise NotImplementedError

    @abstractmethod
    def get_filters(self, grid):
        """
        Build this instrument's filter collection.

        Parameters
        ----------
        grid : synthesizer.grid.Grid
            grid to sample filter transmission curves onto

        Returns
        -------
        : synthesizer.filters.FilterCollection
            this instrument's filters
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Geometry -> Synthesizer's unyt-based resolution/fov
    # ------------------------------------------------------------------
    def synthesizer_resolution(self, angular=False):
        """This instrument's pixel size, as a unyt quantity."""
        if angular:
            return self.sampling.to("arcsec").value * arcsecond
        self._param_check()
        return self.pixel_width.to("kpc").value * kpc

    def synthesizer_fov(self, angular=False):
        """This instrument's field of view (post max_extent clip), as a unyt quantity."""
        if angular:
            return self.field_of_view.to("arcsec").value * arcsecond
        self._param_check()
        return self.extent.to("kpc").value * kpc

    def _project_and_cut(self, pos, xaxis, yaxis):
        """Same FoV + LOS-depth cut used by the from-scratch Instrument
        class, kept here so both approaches select an identical subset
        of particles for a fair comparison."""
        self._param_check()
        valid_str = "xyz"
        xi = valid_str.find(xaxis) if isinstance(xaxis, str) else xaxis
        yi = valid_str.find(yaxis) if isinstance(yaxis, str) else yaxis
        los = self._get_LOS_axis(xaxis, yaxis)

        half_extent = 0.5 * self.extent.to("kpc").value
        half_depth = 0.5 * self.max_extent.to("kpc").value

        x, y, zl = pos[:, xi], pos[:, yi], pos[:, los]
        keep = (
            (np.abs(x) <= half_extent)
            & (np.abs(y) <= half_extent)
            & (np.abs(zl) <= half_depth)
        )
        return x[keep], y[keep], keep

    def _to_instrument_units(self, x_kpc, y_kpc, angular=False):
        """Convert centered (x, y) in kpc to the units the ImageCollection
        was built with (kpc for physical, arcsec for angular)."""
        if not angular:
            return np.column_stack([x_kpc, y_kpc]) * kpc
        scale = self.ang_scale.to("kpc/arcsec").value  # kpc per arcsec
        return np.column_stack([x_kpc / scale, y_kpc / scale]) * arcsecond

    def load_and_project_galaxy(
        self, snap, xaxis=0, yaxis=2, ages=None, metallicity=None, softening=None
    ):
        """
        Build a synthesizer Galaxy from a pygad snapshot, restricted to
        this instrument's field of view.

        Parameters
        ----------
        snap : pygad.Snapshot
            snapshot to analyse
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2
        ages : float, int, np.array, unyt_array, optional
            stellar ages; a scalar is broadcast to all particles, by
            default None (uses the snapshot's own per-particle "age"
            block)
        metallicity : float, int, np.array, optional
            stellar metallicities; a scalar is broadcast to all
            particles, by default None (uses the snapshot's own
            per-particle "metallicity" block)
        softening : float, int, np.array, unyt_array, optional
            per-particle smoothing length in kpc, by default None (falls
            back to half the instrument's resolution element)
        """
        mask = self.get_fov_mask(xaxis, yaxis)
        snap = snap[mask]

        if ages is None:
            # TODO check units
            ages = np.asarray(snap.stars["age"]) * yr
        elif isinstance(ages, (float, int)):
            ages = np.full(len(snap.stars), ages) * yr
        if metallicity is None:
            metallicity = snap.stars["metallicity"]
        elif isinstance(metallicity, (float, int)):
            metallicity = np.full(len(snap.stars), metallicity)
        if softening is None:
            # no physical smoothing length available (e.g. collisionless
            # snapshots without an hsml block) -- fall back to a fraction
            # of the instrument's resolution element rather than leaving
            # particles unsmoothed
            softening = (
                np.full(len(snap.stars), 0.5 * self.resolution_kpc.to("kpc").value)
                * kpc
            )
        elif isinstance(softening, (float, int)):
            softening = np.full(len(snap.stars), softening * kpc)
        stars = synthesizer.particle.Stars(
            initial_masses=np.asarray(snap.stars["mass"]) * Msun,
            ages=ages,
            metallicities=metallicity,
            coordinates=np.asarray(snap.stars["pos"]) * kpc,
            centre=np.zeros(3) * kpc,
            smoothing_lengths=softening,
            redshift=self.redshift,
        )
        self.galaxy = synthesizer.particle.Galaxy(stars=stars, redshift=self.redshift)

    def generate_particle_spectra(
        self, grid_name="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c23.01-sps.hdf5"
    ):
        """
        Generate per-particle spectra and photometry for the loaded galaxy.

        Builds an SSP grid whose wavelength sampling is tailored to this
        instrument's own filters (see `_filter_tailored_lam`), then
        generates dust-attenuated per-particle photometry through this
        instrument's filters via a BimodalPacmanEmission model. If most
        particles share their age with another particle (few unique ages
        relative to the particle count -- see `_DEDUP_UNIQUE_AGE_FRACTION`),
        this is done by deduplicating on (age, metallicity) rather than
        running every particle individually (see
        `_generate_particle_photometry_deduped`); otherwise every particle
        is run through the model directly (see
        `_generate_particle_photometry_full`).

        Parameters
        ----------
        grid_name : str, optional
            SSP grid to query, by default
            "bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c23.01-sps.hdf5"

        Returns
        -------
        grid : synthesizer.grid.Grid
            grid used to generate the spectra
        """
        lam_lo, lam_hi = get_filter_lam_range(self.filter_codes)
        new_lam = _filter_tailored_lam(lam_lo, lam_hi)
        grid = _cached_synthesizer_grid(grid_name, tuple(new_lam.to("angstrom").value))
        self._filters = self.get_filters(grid)
        model = BimodalPacmanEmission(
            grid=grid,
            tau_v_ism=1.0,
            tau_v_birth=0.7,
            dust_curve_ism=PowerLaw(slope=-1.3),
            dust_curve_birth=PowerLaw(slope=-0.7),
            fesc=0.1,
            fesc_ly_alpha=0.9,
            label="total",
            per_particle=True,
        )

        n_particles = len(self.galaxy.stars.ages)
        n_unique_ages = len(np.unique(self.galaxy.stars.ages.to("yr").value))
        if n_unique_ages / n_particles < _DEDUP_UNIQUE_AGE_FRACTION:
            self._generate_particle_photometry_deduped(model)
        else:
            self._generate_particle_photometry_full(model)
        return grid

    def _generate_particle_photometry_full(self, model):
        """
        Generate per-particle spectra and photometry by running every
        particle through the emission model directly.

        Parameters
        ----------
        model : synthesizer.emission_models.BimodalPacmanEmission
            emission model to generate spectra with
        """
        self.galaxy.stars.get_spectra(model, nthreads=_resolve_nthreads())
        self.galaxy.get_observed_spectra(Planck18)
        self.galaxy.get_photo_fnu(self._filters)

    def _generate_particle_photometry_deduped(self, model):
        """
        Generate per-particle photometry by exploiting the finite SSP
        grid: particles sharing the same (age, metallicity) necessarily
        produce byte-identical spectra before mass scaling, so only
        unique (age, metallicity) pairs need to be run through the
        emission model. Results are expanded back out to the full
        particle array by index and rescaled by each particle's own
        mass, so this loses no fidelity relative to
        `_generate_particle_photometry_full` -- it only avoids repeating
        identical grid interpolations.

        Only `particle_photo_fnu["attenuated"]` (what `observe()` reads
        via `get_images_flux`) is reconstructed for the full particle
        array; per-particle spectra are not, since nothing here consumes
        them.

        Parameters
        ----------
        model : synthesizer.emission_models.BimodalPacmanEmission
            emission model to generate spectra with
        """
        stars = self.galaxy.stars
        ages = stars.ages.to("yr").value
        metallicities = np.asarray(stars.metallicities)
        masses = stars.initial_masses.to("Msun").value

        pairs = np.column_stack([ages, metallicities])
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        n_unique = len(unique_pairs)
        if n_unique == 1:
            # sidestep a synthesizer bug triggered by a size-1 emitter:
            # a dimensionless masked attribute (e.g. log10age vs the
            # model's age_pivot) gets stripped to a bare ndarray via
            # `.ndview` in Particles.get_mask, and a later `attr.value`
            # call then assumes it's still a unyt array once
            # attr.size == 1. Padding to 2 identical particles avoids
            # that branch entirely and is otherwise inert, since
            # `inverse` (all zeros here) still indexes the duplicated
            # row 0.
            unique_pairs = np.repeat(unique_pairs, 2, axis=0)
            n_unique = 2
        _logger.info(
            f"Deduplicating spectra generation: {n_unique} unique "
            f"(age, metallicity) pairs for {len(ages)} particles."
        )

        reduced_stars = synthesizer.particle.Stars(
            initial_masses=np.ones(n_unique) * Msun,
            ages=unique_pairs[:, 0] * yr,
            metallicities=unique_pairs[:, 1],
            coordinates=np.zeros((n_unique, 3)) * kpc,
            centre=np.zeros(3) * kpc,
            smoothing_lengths=np.ones(n_unique) * kpc,
            redshift=self.redshift,
        )
        reduced_galaxy = synthesizer.particle.Galaxy(
            stars=reduced_stars, redshift=self.redshift
        )
        reduced_galaxy.stars.get_spectra(model, nthreads=_resolve_nthreads())
        reduced_galaxy.get_observed_spectra(Planck18)
        reduced_galaxy.get_photo_fnu(self._filters)

        # rescale each unique pair's (1 Msun) photometry by every
        # particle's real mass, expanding back out to the full array
        reduced_phot = reduced_galaxy.stars.particle_photo_fnu["attenuated"]
        expanded_fnu = reduced_phot.photo_fnu[:, inverse] * masses[np.newaxis, :]
        stars.particle_photo_fnu["attenuated"] = PhotometryCollection(
            filters=self._filters, photometry=expanded_fnu
        )

        # the model's routing metadata (which labels are directly
        # generated vs combined from others) depends only on model
        # structure, not particle data, so it's safe to copy over from
        # the reduced run
        stars.model_param_cache.update(reduced_galaxy.stars.model_param_cache)
        self.galaxy.model_param_cache.update(reduced_galaxy.model_param_cache)

    def _gaussian_psf_kernel(self, size=25):
        """Build a normalized Gaussian PSF kernel, in pixel units, sized
        from this instrument's angular resolution (PSF FWHM).

        Parameters
        ----------
        size : int, optional
            side length of the (square) kernel array in pixels, by default 25

        Returns
        -------
        : np.array
            normalized 2D PSF kernel
        """
        fwhm_pix = (self.resolution_kpc / self.pixel_width).to("").value
        sigma = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
        y, x = np.mgrid[0:size, 0:size]
        c = size // 2
        psf = np.exp(-0.5 * ((x - c) ** 2 + (y - c) ** 2) / sigma**2)
        return psf / psf.sum()

    def build_instrument(self, angular=False, apply_psf=True, psf_size=25):
        """
        Build the underlying synthesizer PhotometricImager instrument.

        Parameters
        ----------
        angular : bool, optional
            use angular (arcsec) units instead of physical (kpc), by default False
        apply_psf : bool, optional
            convolve images with a Gaussian PSF matched to this instrument's
            angular resolution, by default True
        psf_size : int, optional
            side length of the PSF kernel array in pixels, by default 25
        """
        psfs = None
        if apply_psf:
            kernel = self._gaussian_psf_kernel(size=psf_size)
            psfs = {fcode: kernel for fcode in self._filters.filter_codes}
        self._instr = PhotometricImager(
            label=self.label,
            resolution=self.synthesizer_resolution(angular=angular),
            filters=self._filters,
            psfs=psfs,
        )

    def _generate_flux_images(self, angular=False):
        """
        Generate the attenuated flux image collection for this instrument.

        Falls back to unsmoothed 'hist' imaging if smoothed imaging fails,
        and applies the instrument's PSFs when present.

        Parameters
        ----------
        angular : bool, optional
            use angular (arcsec) units instead of physical (kpc), by
            default False

        Returns
        -------
        imgs : synthesizer.imaging.ImageCollection
            one image per filter, keyed by filter code
        """
        kwargs = dict(
            instrument=self._instr,
            fov=self.synthesizer_fov(angular=angular),
            img_type="smoothed",
            kernel=Kernel().get_kernel(),
            cosmo=Planck18,
        )
        try:
            imgs = self.galaxy.get_images_flux("attenuated", **kwargs)
        except Exception as e:
            _logger.warning(
                f"Smoothed imaging failed ({e}); falling back to unsmoothed "
                "'hist' imaging. Images will look noticeably worse."
            )
            kwargs["img_type"] = "hist"
            imgs = self.galaxy.get_images_flux("attenuated", **kwargs)

        if self._instr.psfs is not None:
            imgs = self._instr.apply_psfs(imgs)

        return imgs

    def make_rgb_image(self, r, g, b, ax=None, angular=False):
        """
        Create and plot an RGB composite image from three of this
        instrument's filters.

        The three filters are mapped to the red, green and blue channels,
        combined with synthesizer's ``make_rgb_image``, and normalised with
        a percentile stretch before display.

        Parameters
        ----------
        r : str
            filter code to map to the red channel
        g : str
            filter code to map to the green channel
        b : str
            filter code to map to the blue channel
        ax : matplotlib.axes.Axes, optional
            axis to plot the composite into, by default None (creates a new
            figure and axis)
        angular : bool, optional
            use angular (arcsec) units instead of physical (kpc), by
            default False

        Returns
        -------
        ax : matplotlib.axes.Axes
            axis the RGB image was plotted onto

        Raises
        ------
        ValueError
            if any of the requested filter codes is not one of this
            instrument's filters
        """
        rgb_filters = {"R": r, "G": g, "B": b}
        valid = set(self.filters.filter_codes)
        for channel, fcode in rgb_filters.items():
            if fcode not in valid:
                raise ValueError(
                    f"filter '{fcode}' requested for the {channel} channel is "
                    f"not one of this instrument's filters: {sorted(valid)}"
                )

        imgs = self._generate_flux_images(angular=angular)
        rgb_img = imgs.make_rgb_image(
            rgb_filters={channel: [fcode] for channel, fcode in rgb_filters.items()}
        )

        vmin = -np.percentile(rgb_img, 32)
        vmax = np.percentile(rgb_img, 99.9)
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgb_img = norm(rgb_img)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        rgb_fmt = lambda x: os.path.splitext(x)[-1].lstrip(".")
        fig.suptitle(
            rf"{self.label} (r:{rgb_fmt(r)}, g:{rgb_fmt(g)}, b:{rgb_fmt(b)}) $z={self.redshift:.3f}$"
        )

        fov_width = self.synthesizer_fov(angular=angular)
        half_width = 0.5 * fov_width.value
        extent = [-half_width, half_width, -half_width, half_width]
        sizebar_units = "arcsec" if angular else "kpc"
        sizebar_length = 0.2 * half_width

        ax.imshow(
            rgb_img.swapaxes(0, 1),
            origin="lower",
            interpolation="nearest",
            extent=extent,
        )
        ax.set_facecolor("k")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        draw_sizebar(ax, sizebar_length, sizebar_units, color="w", fmt=".0f")
        return ax

    def observe(self, ax=None, angular=False):
        """
        Generate and plot flux images for this instrument's filters.

        Each panel is drawn with an arcsinh stretch, an inset scale bar,
        and an inset colourbar labelled with the image's true flux units;
        grid lines and axis ticks/labels are always switched off.

        Parameters
        ----------
        ax : list of matplotlib.axes.Axes, optional
            axes to plot each filter's image into, one per filter, by
            default None (creates a new figure with one axis per filter)
        angular : bool, optional
            use angular (arcsec) units instead of physical (kpc), by
            default False

        Returns
        -------
        ax : list of matplotlib.axes.Axes
            axes the images were plotted onto
        """
        imgs = self._generate_flux_images(angular=angular)

        if ax is None:
            nax = len(self.filters)
            fig, ax = plt.subplots(ncols=nax, sharex="all", sharey="all", squeeze=False)
            if nax > 3:
                nr = next_square_root(nax)
                fig, ax = plt.subplots(
                    nrows=nr,
                    ncols=nr,
                    sharex="all",
                    sharey="all",
                    figsize=(2 * nr, 2 * nr),
                )
            else:
                fig, ax = plt.subplots(
                    ncols=nax, sharex="all", sharey="all", squeeze=False
                )
            if nax == 1:
                ax = np.array([ax])
        else:
            try:
                fig = ax.get_figure()
            except AttributeError:
                fig = ax[0].get_figure()
        fig.suptitle(rf"$z={self.redshift:.3f}$")
        fov_width = self.synthesizer_fov(angular=angular)
        half_width = 0.5 * fov_width.value
        extent = [-half_width, half_width, -half_width, half_width]
        sizebar_units = "arcsec" if angular else "kpc"
        sizebar_length = 0.2 * half_width

        used_axes = set()
        for axi, fcode in zip(ax.flat, self.filters.filter_codes):
            used_axes.add(axi)
            img = imgs[fcode]
            norm = _robust_asinh_norm(img.arr)
            im = axi.imshow(
                img.arr.T, cmap="bone", norm=norm, extent=extent, origin="lower"
            )
            axi.set_title(fcode)
            axi.set_facecolor("k")

            # a) grid lines always off
            axi.grid(False)
            # b) no x/y ticks or labels
            axi.set_xticks([])
            axi.set_yticks([])
            axi.set_xlabel("")
            axi.set_ylabel("")
            # c) inset scale bar
            draw_sizebar(axi, sizebar_length, sizebar_units, color="w", fmt=".0f")
            # d) inset colourbar, labelled with the image's true flux units
            cax = inset_axes(
                axi,
                width="5%",
                height="50%",
                loc="upper left",
                borderpad=1,
            )
            cbar = axi.figure.colorbar(im, cax=cax)
            cbar.set_label(f"[{img.units}]", color="w")
            cbar.ax.yaxis.set_tick_params(color="w", labelcolor="w")
            cbar.outline.set_edgecolor("w")

        for axi in ax.flat:
            if axi not in used_axes:
                fig.delaxes(axi)
        return ax


class Euclid_VIS(PhotometricInstrument):
    """Euclid VIS imaging channel (broad optical band, ~550-900 nm)."""

    def __init__(self, z=None):
        """
        Euclid VIS imaging channel.

        Parameters
        ----------
        z : float, optional
            redshift of observations, by default None
        """
        super().__init__(
            fov=0.787 * 3600,  # ~0.787 deg per detector -> arcsec (illustrative)
            sampling=0.101,  # arcsec/pixel
            res=0.16,  # PSF FWHM ~0.16"
            z=z,
            label="EuclidVIS",
        )

    @property
    def filter_codes(self):
        """SVO filter codes for Euclid VIS."""
        return EUCLID_FILTER_CODES

    def get_filters(self, grid):
        """
        Build the Euclid VIS filter collection.

        Parameters
        ----------
        grid : synthesizer.grid.Grid
            grid to sample filter transmission curves onto

        Returns
        -------
        : synthesizer.filters.FilterCollection
            Euclid VIS filter collection
        """
        return get_filter_collection(grid, "euclid")


class HSTWFC3(PhotometricInstrument):
    """HST WFC3/UVIS, F606W-like broad V band."""

    def __init__(self, z=None):
        """
        HST WFC3/UVIS, F606W-like broad V band.

        Parameters
        ----------
        z : float, optional
            redshift of observations, by default None
        """
        super().__init__(
            fov=162.0,  # arcsec, UVIS field of view
            sampling=0.04,  # arcsec/pixel
            res=0.07,  # diffraction-limited PSF FWHM, approximate
            z=z,
            label="HSTWFC3",
        )

    @property
    def filter_codes(self):
        """SVO filter codes for HST WFC3/UVIS."""
        return HST_FILTER_CODES

    def get_filters(self, grid):
        """
        Build the HST WFC3/UVIS filter collection.

        Parameters
        ----------
        grid : synthesizer.grid.Grid
            grid to sample filter transmission curves onto

        Returns
        -------
        : synthesizer.filters.FilterCollection
            HST WFC3/UVIS filter collection
        """
        return get_filter_collection(grid, "hst")


class JWST_MIRI(PhotometricInstrument):
    """JWST MIRI"""

    def __init__(self, z=None):
        """
        JWST MIRI

        Parameters
        ----------
        z : float, optional
            redshift of observations, by default None
        """
        super().__init__(
            fov=74,  # arcsec
            sampling=0.11,  # arcsec/pixel
            res=0.22,  # diffraction-limited PSF FWHM at F560W; grows with wavelength
            z=z,
            label="JWST-MIRI",
        )

    @property
    def filter_codes(self):
        """SVO filter codes for JWST-MIRI."""
        return JWST_MIRI_FILTER_CODES

    def get_filters(self, grid):
        """
        Build the JWST-MIRI filter collection.

        Parameters
        ----------
        grid : synthesizer.grid.Grid
            grid to sample filter transmission curves onto

        Returns
        -------
        : synthesizer.filters.FilterCollection
            JWST-MIRI filter collection
        """
        return get_filter_collection(grid, "jwst_miri")


class JWST_NIRCam(PhotometricInstrument):
    """JWST NIRCam"""

    def __init__(self, z=None):
        """
        JWST NIRCam

        Parameters
        ----------
        z : float, optional
            redshift of observations, by default None
        """
        super().__init__(
            fov=132,  # arcsec
            sampling=0.031,  # arcsec/pixel
            res=0.07,  # diffraction-limited PSF FWHM, approximate
            z=z,
            label="JWST-NIRCam",
        )

    @property
    def filter_codes(self):
        """SVO filter codes for JWST-NIRCam."""
        return JWST_NIRCam_FILTER_CODES

    def get_filters(self, grid):
        """
        Build the JWST-NIRCam filter collection.

        Parameters
        ----------
        grid : synthesizer.grid.Grid
            grid to sample filter transmission curves onto

        Returns
        -------
        : synthesizer.filters.FilterCollection
            JWST-NIRCam filter collection
        """
        return get_filter_collection(grid, "jwst_nircam")


class VLT_FORS2(PhotometricInstrument):
    """VLT FORS2 Imager"""

    def __init__(self, z=None):
        """
        VLT FORS2 Imager

        Parameters
        ----------
        z : float, optional
            redshift of observations, by default None
        """
        super().__init__(
            fov=6.8 * 60,  # arcsec (6.8' FoV, standard-resolution collimator)
            sampling=0.25,  # arcsec/pixel (SR collimator, default 2x2 binning)
            res=0.5,  # seeing-limited PSF FWHM (ground-based; Paranal ~0.5-0.8")
            z=z,
            label="FORS2",
        )

    @property
    def filter_codes(self):
        """SVO filter codes for Paranal VLT FORS2"""
        return VLT_FORS2_FILTER_CODES

    def get_filters(self, grid):
        """
        Build the Paranal VLT FORS2 filter collection.

        Parameters
        ----------
        grid : synthesizer.grid.Grid
            grid to sample filter transmission curves onto

        Returns
        -------
        : synthesizer.filters.FilterCollection
            Paranal VLT FORS2 filter collection
        """
        return get_filter_collection(grid, "vlt_fors2")


# ------------------------------------------------------------------
# IFU instruments
# ------------------------------------------------------------------


class IFUInstrument(BasicInstrument):
    @abstractmethod
    def __init__(self, fov, sampling, res=None, z=None, pseudo_particle_split=None):
        """
        Base class for IFU observations. The class has an attribute 'voronoi' which gives access to the underlying baggins.analysis.voronoi.VoronoiKinematics object, and all its methods (including plotting routines).

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        sampling : float
            spatial sampling of instrument in arcsec/pixel
        res : float, optional
            angular resolution in arcsec, by default None
        z : float, optional
            redshift of observations, by default None
        pseudo_particle_split : int, optional
            number of pseudoparticles to generate for each star to mimic seeing, by default None
        """
        super().__init__(fov, sampling, res, z)
        if pseudo_particle_split is None:
            pseudo_particle_split = 25
        self.pseudo_particle_split = pseudo_particle_split
        self.voronoi = None

    def make_observation(
        self, snap, xaxis=0, yaxis=2, signal_noise=1000, moment=None, rng=None
    ):
        """
        Convenience method that wraps the instrument properties into the VoronoiKinematics interface.

        Parameters
        ----------
        snap : pygad.Snapshot
            snapshot to analyse
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2
        signal_noise : int, float, optional
            Poisson S/N per bin, by default 1000
        moment : int, optional
            Gauss Hermite order to fit to, by default None
        rng : np.random.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        if rng is None:
            rng = np.random.default_rng()
        mask = self.get_fov_mask(xaxis=xaxis, yaxis=yaxis)
        LOS_axis = self._get_LOS_axis(xaxis=xaxis, yaxis=yaxis)
        self.voronoi = VoronoiKinematics(
            x=snap[mask]["pos"][:, xaxis],
            y=snap[mask]["pos"][:, yaxis],
            V=snap[mask]["vel"][:, LOS_axis],
            m=snap[mask]["mass"],
            Npx=self.number_pixels,
            seeing={
                "num": self.pseudo_particle_split,
                "sigma": self.resolution_kpc.value,
                "rng": rng,
            },
        )
        self.voronoi.make_grid(part_per_bin=int(signal_noise**2))
        kwargs = {}
        if moment is not None:
            kwargs["p"] = moment
        self.voronoi.binned_LOSV_statistics(**kwargs)

    def overlay_isophotes_on_maps(
        self, snap, axes, xaxis=0, yaxis=2, quantiles=None, **kwargs
    ):
        """
        Overlay isophotal contours containing certain quantiles of mass on the IFU plots.

        Parameters
        ----------
        snap : pygad.Snapshot
            snapshot to analyse
        axes : np.array
            array of plotting axes
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2
        quantiles : list, optional
            quantiles to plot, by default None
        kwargs : dict, optional
            other kyword arguments parsed to plt.contour()
        """
        if quantiles is None:
            quantiles = [0.5, 0.8]
        mask = self.get_fov_mask(xaxis, yaxis)
        x = snap[mask]["pos"][:, xaxis].view(np.ndarray)
        y = snap[mask]["pos"][:, yaxis].view(np.ndarray)
        weights = snap[mask]["mass"].view(np.ndarray)

        # bin particles onto a grid instead of feeding raw points to KDE
        x_edges = np.linspace(x.min() - 1, x.max() + 1, self.number_pixels + 1)
        y_edges = np.linspace(y.min() - 1, y.max() + 1, self.number_pixels + 1)
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=weights)

        # KDE now operates on grid
        xi = get_histogram_bin_centres(x_edges)
        yi = get_histogram_bin_centres(y_edges)
        Xi, Yi = np.meshgrid(xi, yi)
        # only pass non-empty bins to the KDE
        nonempty = H.T > 0  # histogram2d is (x,y)-indexed; transpose to (row=y, col=x)
        kde = gaussian_kde(
            np.vstack([Xi[nonempty], Yi[nonempty]]),
            weights=H.T[nonempty],
        )
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        # sort bins by density descending, walk cumulative mass
        total_mass = weights.sum()
        flat_density = Zi.ravel()
        flat_mass = H.T.ravel()  # mass in each bin, matched to Zi layout
        order = np.argsort(flat_density)[::-1]
        cumulative = np.cumsum(flat_mass[order]) / total_mass
        levels = []
        for q in quantiles:
            idx = np.searchsorted(cumulative, q)
            levels.append(flat_density[order[idx]])

        kwargs.setdefault("linewidths", 0.5)
        for ax in axes.flat:
            ax.contour(Xi, Yi, Zi, levels=sorted(levels), colors="k", **kwargs)


class MUSE_NFM(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        MUSE narrow field mode instrument. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
        """
        super().__init__(
            fov=7.42,
            sampling=0.025,
            res=0.055,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{MUSE}$"


class MUSE_WFM(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        MUSE wide field mode instrument. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/muse/overview.html
        """
        super().__init__(
            fov=60,
            sampling=0.2,
            res=0.4,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{MUSE}$"


class HARMONI_SENSITIVE(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        HARMONI optimised for sensitivity. Parameters taken from:
        https://elt.eso.org/instrument/HARMONI/
        """
        super().__init__(
            fov=3.04,
            sampling=20e-3,
            res=20e-3,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{HARMONI}$"


class HARMONI_BALANCED(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        HARMONI balanced for sensitivity and spatial. Parameters taken from:
        https://elt.eso.org/instrument/HARMONI/
        """
        super().__init__(
            fov=1.52,
            sampling=10e-3,
            res=20e-3,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{HARMONI}$"


class HARMONI_SPATIAL(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        HARMONI optimised for spatial. Parameters taken from:
        https://elt.eso.org/instrument/HARMONI/
        """
        super().__init__(
            fov=0.61,
            sampling=4e-3,
            res=10e-3,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{HARMONI}$"


class ERIS_IFU(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        Eris IFU. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/eris/doc/ERIS_User_Manual_v116.0.pdf
        """
        super().__init__(
            fov=0.8,
            sampling=25e-3,
            res=0.1,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{ERIS}$"


class JWST_IFU(IFUInstrument):
    def __init__(self, z=None, pseudo_particle_split=None):
        """
        JWST IFU. Parameters taken from:
        https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0
        """
        super().__init__(
            fov=3,
            sampling=0.1,
            res=100e-3,
            z=z,
            pseudo_particle_split=pseudo_particle_split,
        )
        self.label = r"$\mathrm{JWST}$"


# ------------------------------------------------------------------
# Long slit spectroscopy instruments
# ------------------------------------------------------------------


class LongSlitInstrument(BasicInstrument):
    @abstractmethod
    def __init__(
        self, fov, sampling, slit_width, slit_length, res=None, z=None, rng=None
    ):
        """
        Base class for long slit spectroscopy instruments.

        Parameters
        ----------
        fov : float
            field of view in arcsecs
        sampling : float
            spatial sampling of instrument in arcsec/pixel
        slit_width : float
            width of slit in arcsecs
        slit_length : float
            length of slit in arcsecs
        res : float, optional
            angular resolution in arcsec, by default None
        z : float, optional
            redshift of observations, by default None
        rng : np.random.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        super().__init__(fov, sampling, res, z=z)
        self.slit_width = slit_width * Unit("arcsec")
        self.slit_length = slit_length * Unit("arcsec")
        if rng is None:
            self._rng = np.random.default_rng()
        self._slit_length_kpc = None

    @property
    def slit_length_kpc(self):
        """Slit length in kpc, truncated to the instrument's maximum extent."""
        if self._slit_length_kpc is None:
            sl = self.slit_length * self._ang_scale
            if sl > self.extent:
                _logger.warning(
                    f"Truncating {self.name} slit length ({sl:.1e}) to maximum extent ({self.extent:.1e})!"
                )
                sl = self.extent
            self._slit_length_kpc = sl
        return self._slit_length_kpc

    @property
    def slit_width_kpc(self):
        """Slit width in kpc."""
        return self.slit_width * self._ang_scale

    def get_slit_mask(self, xaxis=0, yaxis=2):
        """
        Mask for those particles in the slit.

        Parameters
        ----------
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2

        Returns
        -------
        mask : ExprMask
            pygad mask to select those particles within the slit
        """
        mask = ExprMask(
            f"abs(pos[:,{xaxis}]) <= {0.5 * self.slit_length_kpc.value}"
        ) & ExprMask(f"abs(pos[:,{yaxis}]) <= {0.5 * self.slit_width_kpc.value}")
        return mask

    def get_LOS_velocity_dispersion_profile(
        self, snap, N=100, xaxis=0, yaxis=2, N_per_bin=100
    ):
        """
        Calculate a 1D velocity dispersion profile using the slit. Note that no
        centring is done.

        Parameters
        ----------
        snap : pygad.Snapshot
            snapshot to analyse
        N : int, optional
            number of pseudoparticles per particle to generate, by default 100
        xaxis : int, optional
            spatial x axis, by default 0
        yaxis : int, optional
            spatial y axis, by default 2

        Returns
        -------
        : np.array
            centres of pixels that define the slit
        vel_disp : np.array
            velocity dispersion along the long side of the slit
        """
        LOS_axis = self._get_LOS_axis(xaxis=xaxis, yaxis=yaxis)
        mask = self.get_slit_mask(xaxis=xaxis, yaxis=yaxis)
        _logger.debug(f"Slit length is {self.slit_length_kpc:.2e}")
        x = np.array(
            [
                xx + self._rng.normal(0, self.resolution_kpc.value, size=N)
                for xx in snap.stars[mask]["pos"][:, xaxis]
            ]
        ).flatten()
        V = np.repeat(snap.stars[mask]["vel"][:, LOS_axis], N).flatten()

        # ensure pseudo-particles are not generated outside slit
        pseudo_mask = np.abs(x) < 0.5 * self.slit_length_kpc.value
        x = x[pseudo_mask]
        V = V[pseudo_mask]

        bins = equal_count_bins(x, N_per_bin)
        _logger.debug(f"Starting with {len(bins)} bins for LSS")
        # if bin difference is less than instrument sampling, join bins
        if np.any(np.diff(bins) < self.pixel_width.value):
            # which bins are smaller than the pixel width
            _bins = np.full_like(bins, np.nan)
            _bins[0] = bins[0]
            offset = 0
            for i, b in enumerate(bins[1:]):
                if b - _bins[i - offset] > self.pixel_width.value:
                    _bins[i - offset + 1] = b
                else:
                    offset += 1
            bins = _bins[~np.isnan(_bins)]
        try:
            assert np.all(np.diff(bins) >= self.pixel_width.value)
        except AssertionError:
            _logger.exception(
                "Some slit bins are narrower than the instrument resolution!",
                exc_info=True,
            )
            raise
        _logger.debug(
            f"There are {len(bins)} bins for LSS. Minimum bin width is {np.min(np.diff(bins))} (pixel width is {self.pixel_width})."
        )
        vel_disp, *_ = binned_statistic(x, V, bins=bins, statistic="std")
        return get_histogram_bin_centres(bins), vel_disp


class MICADO_WFM_LSS(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        MICADO for ELT
        https://elt.eso.org/instrument/MICADO/
        """
        super().__init__(
            fov=50.5,
            sampling=4e-3,
            res=10e-3,
            slit_width=16e-3,
            slit_length=3,
            rng=rng,
            z=z,
        )
        self.label = r"$\mathrm{MICADO}$"


class MICADO_NFM_LSS(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        MICADO for ELT
        https://elt.eso.org/instrument/MICADO/
        """
        super().__init__(
            fov=18,
            sampling=1.5e-3,
            res=20e-3,
            slit_width=16e-3,
            slit_length=3,
            rng=rng,
            z=z,
        )
        self.label = r"$\mathrm{MICADO}$"


class VLT_FORS2_LSS(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        VLT FORS2 instrument for long slit spectroscopy
        https://www.eso.org/sci/facilities/paranal/instruments/fors/doc/VLT-MAN-ESO-13100-1543_P116.2.pdf
        """
        super().__init__(
            fov=7.1 * 60,
            sampling=0.125,
            slit_width=0.28,
            slit_length=6.8 * 60,
            res=0.25,
            rng=rng,
            z=z,
        )
        self.label = r"$\mathrm{FORS2}$"


class ERIS_NIX_NFM_LSS(LongSlitInstrument):
    def __init__(self, z=None, rng=None):
        """
        Eris long slit. Parameters taken from:
        https://www.eso.org/sci/facilities/paranal/instruments/eris/doc/ERIS_User_Manual_v116.0.pdf
        """
        super().__init__(
            fov=26,
            sampling=13e-3,
            slit_width=68e-3,
            slit_length=12,
            res=0.1,
            z=z,
            rng=rng,
        )
        self.label = r"$\mathrm{ERIS}$"


class JWST_LSS(LongSlitInstrument):
    def __init__(self, z=None, rng=None):
        """
        JWST long slit. Parameters taken from:
        https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph#gsc.tab=0
        """
        super().__init__(
            fov=3.4 * 60,
            sampling=0.1,
            res=100e-3,
            slit_width=0.2,
            slit_length=3.2,
            z=z,
            rng=rng,
        )
        self.label = r"$\mathrm{JWST}$"

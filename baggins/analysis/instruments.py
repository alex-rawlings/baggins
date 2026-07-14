from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import binned_statistic, gaussian_kde
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from astropy.units import Unit
from astropy.cosmology import Planck18
from astropy.constants import L_sun
from pygad import ExprMask
from unyt import arcsecond, erg, s, Hz, angstrom, Msun, kpc, yr
import synthesizer.particle
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.filters import Filter, FilterCollection
from synthesizer.emission_models import BimodalPacmanEmission
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.kernel_functions import Kernel
from synthesizer.instruments.photometric_imager import PhotometricImager
from baggins.analysis.obs_helper import (
    get_synthesizer_grid,
    get_euclid_filter_collection,
    get_hst_filter_collection,
)
from baggins.analysis.voronoi import VoronoiKinematics
from baggins.env_config import _cmlogger
from baggins.cosmology import angular_scale
from baggins.mathematics import get_histogram_bin_centres, equal_count_bins
from baggins.plotting import NormedColours

__all__ = [
    "MUSE_NFM",
    "MUSE_WFM",
    "HARMONI_SENSITIVE",
    "HARMONI_BALANCED",
    "HARMONI_SPATIAL",
    "Euclid_VIS",
    "HSTWFC3",
    "ERIS_IFU",
    "JWST_IFU",
    "MICADO_WFM",
    "MICADO_NFM",
    "VLT_FORS2",
    "ERIS_NIX_NFM",
    "JWST_LSS",
]

_logger = _cmlogger.getChild(__name__)


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
        try:
            assert self._ang_scale is not None
        except AssertionError:
            _logger.exception("Redshift must be set first!", exc_info=True)
            raise RuntimeError

    @property
    def redshift(self):
        return self._redshift

    @redshift.setter
    def redshift(self, z):
        if z < 1e-3:
            # protect against cosmology methods failing at z=0
            z = 1e-3
        self._redshift = z
        self._ang_scale = angular_scale(z)

    @property
    def max_extent(self):
        return self._max_extent

    @max_extent.setter
    def max_extent(self, R):
        self._max_extent = R
        try:
            self._max_extent.value
        except AttributeError:
            self._max_extent = self._max_extent * Unit("kpc")

    @property
    def ang_scale(self):
        # in kpc/arcsec
        return self._ang_scale

    @property
    def pixel_width(self):
        # in kpc
        self._param_check()
        return self.sampling * self._ang_scale

    @property
    def resolution_kpc(self):
        self._param_check()
        return self.angular_resolution * self._ang_scale

    @property
    def extent(self):
        # in kpc
        self._param_check()
        return min(self.ang_scale * self.field_of_view, self.max_extent)

    @property
    def number_pixels(self):
        return int(self.extent / self.pixel_width)

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
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


class _PhotometricInstrument(BasicInstrument):
    def __init__(
        self,
        fov,
        sampling,
        res=None,
        z=None,
        psf_fwhm=None,
        psf_type="gaussian",
        moffat_beta=2.5,
        read_noise=0.0,
        dark_current=0.0,
        sky_background=0.0,
        zeropoint=25.0,
        flux_zeropoint=None,
        gain=1.0,
        full_well=None,
        exposure_time=1.0,
    ):
        """
        Parameters
        ----------
        fov, sampling, res, z : see BasicInstrument
        psf_fwhm : float, optional
            PSF FWHM in arcsec. Defaults to `angular_resolution` if not given.
        psf_type : {"gaussian", "moffat"}
            Functional form of the PSF.
        moffat_beta : float
            Moffat profile beta parameter (only used if psf_type="moffat").
        read_noise : float
            Detector read noise, in electrons (e-) per pixel.
        dark_current : float
            Dark current, in e-/s/pixel.
        sky_background : float
            Sky background, in e-/s/pixel (already matched to `sampling`).
        zeropoint : float
            AB magnitude corresponding to 1 e-/s in this band.
        flux_zeropoint : float, optional
            Flux, in erg/s/cm^2, corresponding to 1 e-/s in this band.
            Used to convert particle luminosities to count rate in
            `luminosity_to_rate` / `image_from_particles`. If your
            luminosities are already band-specific and monochromatic-flux
            calibrated, set this from the instrument's throughput; otherwise
            treat it as a tunable calibration constant.
        gain : float
            Detector gain, in e-/ADU. Use 1.0 to keep everything in electrons.
        full_well : float, optional
            Full well depth in e-, for saturation. None disables saturation.
        exposure_time : float
            Default exposure time in seconds.
        """
        super().__init__(fov, sampling, res=res, z=z)

        self.psf_fwhm = (
            psf_fwhm if psf_fwhm is not None else self.angular_resolution.value
        )
        self.psf_type = psf_type
        self.moffat_beta = moffat_beta

        self.read_noise = read_noise
        self.dark_current = dark_current
        self.sky_background = sky_background
        self.zeropoint = zeropoint
        self.flux_zeropoint = flux_zeropoint
        self.gain = gain
        self.full_well = full_well
        self.exposure_time = exposure_time

    # ------------------------------------------------------------------
    # PSF
    # ------------------------------------------------------------------
    @property
    def psf_sigma_pix(self):
        """PSF Gaussian sigma, in pixels, from FWHM."""
        fwhm_pix = self.psf_fwhm / self.sampling.value
        return fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def psf_kernel(self, size=None):
        """
        Build a normalized 2D PSF kernel in pixel units.

        Parameters
        ----------
        size : int, optional
            Kernel side length in pixels (odd). Defaults to ~8x sigma,
            clipped to be odd and at least 7 pixels.

        Returns
        -------
        kernel : ndarray
            Normalized (sum=1) 2D PSF kernel.
        """
        sigma = self.psf_sigma_pix
        if size is None:
            size = max(7, int(np.ceil(8 * sigma)) | 1)  # force odd
        elif size % 2 == 0:
            size += 1

        y, x = np.mgrid[0:size, 0:size]
        cy = cx = size // 2
        r2 = (x - cx) ** 2 + (y - cy) ** 2

        if self.psf_type == "gaussian":
            kernel = np.exp(-0.5 * r2 / sigma**2)
        elif self.psf_type == "moffat":
            # alpha related to FWHM and beta for a Moffat profile
            fwhm_pix = self.psf_fwhm / self.sampling.value
            alpha = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / self.moffat_beta) - 1.0))
            kernel = (1.0 + r2 / alpha**2) ** (-self.moffat_beta)
        else:
            _logger.exception(f"Unknown psf_type '{self.psf_type}'", exc_info=True)
            raise ValueError

        return kernel / kernel.sum()

    def convolve_with_psf(self, image):
        """Convolve a 2D image (counts or flux) with the instrument PSF."""
        kernel = self.psf_kernel()
        return fftconvolve(image, kernel, mode="same")

    # ------------------------------------------------------------------
    # Photometric calibration
    # ------------------------------------------------------------------
    def mag_to_rate(self, mag):
        """AB magnitude -> count rate in e-/s, using the instrument zeropoint."""
        return 10.0 ** (-0.4 * (mag - self.zeropoint))

    def rate_to_mag(self, rate):
        """Count rate in e-/s -> AB magnitude."""
        rate = np.clip(rate, 1e-12, None)
        return self.zeropoint - 2.5 * np.log10(rate)

    # ------------------------------------------------------------------
    # Scene generation (simple synthetic sources -> flux-rate image)
    # ------------------------------------------------------------------
    def render_scene(self, catalog):
        """
        Render a flux-rate image (e-/s/pixel, pre-PSF) from a source catalog.

        Parameters
        ----------
        catalog : list of dict
            Each entry needs: 'x', 'y' (pixel coords), 'mag' (AB mag).
            Optional 'r_eff' (pixels) and 'n' (Sersic index) for extended
            sources; point sources are used if 'r_eff' is omitted.

        Returns
        -------
        image : ndarray
            2D array of count rate (e-/s/pixel), shape (npix, npix).
        """
        npix = self.number_pixels
        image = np.zeros((npix, npix))

        yy, xx = np.mgrid[0:npix, 0:npix]
        for src in catalog:
            rate = self.mag_to_rate(src["mag"])
            x0, y0 = src["x"], src["y"]

            if "r_eff" in src and src["r_eff"] > 0:
                r_eff = src["r_eff"]
                n = src.get("n", 1.0)
                bn = 1.9992 * n - 0.3271  # standard approximation
                r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
                profile = np.exp(-bn * ((r / r_eff) ** (1.0 / n) - 1.0))
                profile /= profile.sum()
                image += rate * profile
            else:
                # point source: deposit into nearest pixel
                ix, iy = int(round(x0)), int(round(y0))
                if 0 <= ix < npix and 0 <= iy < npix:
                    image[iy, ix] += rate

        return image

    def random_catalog(self, n_sources=50, mag_range=(20, 26), seed=None):
        """Generate a random point-source + extended-source catalog for testing."""
        rng = np.random.default_rng(seed)
        npix = self.number_pixels
        catalog = []
        for _ in range(n_sources):
            entry = {
                "x": rng.uniform(0, npix),
                "y": rng.uniform(0, npix),
                "mag": rng.uniform(*mag_range),
            }
            if rng.random() < 0.5:
                entry["r_eff"] = rng.uniform(1.5, 6.0)
                entry["n"] = rng.choice([1.0, 4.0])
            catalog.append(entry)
        return catalog

    # ------------------------------------------------------------------
    # Noise / detector effects
    # ------------------------------------------------------------------
    def add_noise(self, counts_image, exposure_time=None):
        """
        Apply sky, dark current, Poisson, and read noise to a noiseless
        counts image (in e-, already integrated over exposure_time).
        """
        t = exposure_time if exposure_time is not None else self.exposure_time

        sky = self.sky_background * t
        dark = self.dark_current * t

        total_e = np.clip(counts_image + sky + dark, 0, None)
        noisy_e = np.random.poisson(total_e).astype(float)
        noisy_e += np.random.normal(0.0, self.read_noise, size=counts_image.shape)

        if self.full_well is not None:
            noisy_e = np.clip(noisy_e, None, self.full_well)

        return noisy_e / self.gain  # convert to ADU if gain != 1

    # ------------------------------------------------------------------
    # Full observation pipeline
    # ------------------------------------------------------------------
    def observe(self, scene_rate_image, exposure_time=None):
        """
        Run the full pipeline on a noiseless count-rate image (e-/s/pixel):
        PSF convolution -> integrate over exposure time -> add noise.

        Parameters
        ----------
        scene_rate_image : ndarray
            2D count-rate image (e-/s/pixel), e.g. from `render_scene`.
        exposure_time : float, optional
            Overrides `self.exposure_time` if given.

        Returns
        -------
        image_adu : ndarray
            Simulated detector frame, in ADU (or e- if gain=1).
        """
        t = exposure_time if exposure_time is not None else self.exposure_time
        blurred_rate = self.convolve_with_psf(scene_rate_image)
        counts = blurred_rate * t
        return self.add_noise(counts, exposure_time=t)

    def mock_observation(
        self,
        catalog=None,
        n_sources=50,
        mag_range=(20, 26),
        seed=None,
        exposure_time=None,
    ):
        """Convenience: build a random scene (or use given catalog) and observe it."""
        if catalog is None:
            catalog = self.random_catalog(
                n_sources=n_sources, mag_range=mag_range, seed=seed
            )
        scene = self.render_scene(catalog)
        return self.observe(scene, exposure_time=exposure_time)

    # ------------------------------------------------------------------
    # Photometric calibration from luminosity
    # ------------------------------------------------------------------
    def flux_from_luminosity(self, luminosity, z=None, cosmology=None):
        """
        Convert luminosity (Lsun) to observed flux (erg/s/cm^2) via the
        luminosity distance at the instrument's redshift.

        Parameters
        ----------
        luminosity : array_like
            Per-particle (or per-pixel) luminosity in solar luminosities.
            For band-correct photometry this should already be luminosity
            *in the instrument's bandpass* -- this does not apply a
            K-correction or SED integration.
        z : float, optional
            Redshift to use; defaults to `self.redshift` if already set.
        cosmology : astropy.cosmology instance, optional
            Defaults to astropy.cosmology.Planck18.

        Returns
        -------
        flux : ndarray
            Flux in erg/s/cm^2.
        """

        if z is None:
            z = self.redshift
        cosmo = cosmology if cosmology is not None else Planck18

        d_L = cosmo.luminosity_distance(z).to("cm").value
        L_erg_s = np.asarray(luminosity) * L_sun.to("erg/s").value
        return L_erg_s / (4.0 * np.pi * d_L**2)

    def luminosity_to_rate(
        self, luminosity, z=None, flux_zeropoint=None, cosmology=None
    ):
        """
        Convert luminosity (Lsun) directly to a detector count rate (e-/s),
        using `self.flux_zeropoint` (flux in erg/s/cm^2 for 1 e-/s) unless
        overridden here.
        """
        zp = flux_zeropoint if flux_zeropoint is not None else self.flux_zeropoint
        if zp is None:
            _logger.exception(
                "flux_zeropoint must be set on the instrument (flux in erg/s/cm^2 "
                "corresponding to 1 e-/s) before converting luminosity to count rate.",
                exc_info=True,
            )
            raise RuntimeError
        flux = self.flux_from_luminosity(luminosity, z=z, cosmology=cosmology)
        return flux / zp

    # ------------------------------------------------------------------
    # Building images directly from particle data
    # ------------------------------------------------------------------
    def project_particles(self, pos, xaxis="x", yaxis="y"):
        """
        Select particles inside the instrument FoV and LOS depth, and
        convert their in-plane positions into pixel coordinates.

        Parameters
        ----------
        pos : ndarray, shape (N, 3)
            Particle positions in kpc, centered on the object of interest.
        xaxis, yaxis : int or str
            Spatial axes to project onto (see `BasicInstrument._get_LOS_axis`).

        Returns
        -------
        px, py : ndarray
            Pixel coordinates of the selected particles (float, unbinned).
        keep : ndarray (bool)
            Mask into the original `pos` array marking retained particles.
        """
        self._param_check()
        valid_str = "xyz"
        xi = valid_str.find(xaxis) if isinstance(xaxis, str) else xaxis
        yi = valid_str.find(yaxis) if isinstance(yaxis, str) else yaxis
        los = self._get_LOS_axis(xaxis, yaxis)

        half_extent = 0.5 * self.extent.value  # kpc, in-plane
        half_depth = 0.5 * self.max_extent.to("kpc").value  # kpc, along LOS

        x, y, zlos = pos[:, xi], pos[:, yi], pos[:, los]
        keep = (
            (np.abs(x) <= half_extent)
            & (np.abs(y) <= half_extent)
            & (np.abs(zlos) <= half_depth)
        )

        npix = self.number_pixels
        pixel_width = self.pixel_width.to("kpc").value  # kpc/pixel

        px = (x[keep] + half_extent) / pixel_width
        py = (y[keep] + half_extent) / pixel_width
        px = np.clip(px, 0, npix - 1e-6)
        py = np.clip(py, 0, npix - 1e-6)

        return px, py, keep

    def bin_particles(self, px, py, weights):
        """Bin projected particle positions + per-particle weights into a 2D image."""
        npix = self.number_pixels
        image, _, _ = np.histogram2d(
            py, px, bins=npix, range=[[0, npix], [0, npix]], weights=weights
        )
        return image

    def image_from_particles(
        self,
        pos,
        weights,
        xaxis="x",
        yaxis="y",
        weight_type="rate",
        z=None,
        flux_zeropoint=None,
        exposure_time=None,
        apply_noise=True,
    ):
        """
        Build a mock observation directly from particle position + weight
        arrays (mass, velocities aren't needed for imaging itself, but
        `pos` is expected to already be centered on the object of interest;
        velocities are typically used upstream for e.g. kinematic cuts).

        Parameters
        ----------
        pos : ndarray, shape (N, 3)
            Particle positions in kpc.
        weights : ndarray, shape (N,)
            Per-particle quantity to bin; interpretation set by `weight_type`.
        xaxis, yaxis : int or str
            Projection axes.
        weight_type : {"rate", "luminosity"}
            "rate": weights are already a detector count rate (e-/s) per
                particle -- use this if you've already done SED/bandpass
                integration per particle.
            "luminosity": weights are luminosity in Lsun; converted to
                count rate via `luminosity_to_rate` (needs `self.redshift`
                and `self.flux_zeropoint`, or override with `z`/`flux_zeropoint`).
        z : float, optional
            Redshift override, only used if weight_type="luminosity".
        flux_zeropoint : float, optional
            Flux (erg/s/cm^2) for 1 e-/s; only used if weight_type="luminosity".
        exposure_time : float, optional
            Overrides `self.exposure_time`.
        apply_noise : bool
            If False, returns the noiseless PSF-convolved rate image
            (useful for checking geometry/projection before adding noise).

        Returns
        -------
        image : ndarray
            Simulated detector frame (ADU or e-), or noiseless PSF-convolved
            rate image if apply_noise=False.
        """
        pos = np.asarray(pos)
        weights = np.asarray(weights)

        px, py, keep = self.project_particles(pos, xaxis=xaxis, yaxis=yaxis)
        w = weights[keep]

        if weight_type == "luminosity":
            w = self.luminosity_to_rate(w, z=z, flux_zeropoint=flux_zeropoint)
        elif weight_type != "rate":
            _logger.exception(f"Unknown weight_type '{weight_type}'", exc_info=True)
            raise ValueError

        rate_image = self.bin_particles(px, py, w)

        if not apply_noise:
            return self.convolve_with_psf(rate_image)

        return self.observe(rate_image, exposure_time=exposure_time)

    def image_from_snapshot(
        self,
        snap,
        xaxis="x",
        yaxis="y",
        pos_key="pos",
        luminosity_key="lum",
        mass_key="mass",
        weight_type="luminosity",
        z=None,
        flux_zeropoint=None,
        exposure_time=None,
        apply_noise=False,
    ):
        """
        Convenience wrapper around `image_from_particles` for a pygad-style
        snapshot object (dict-like field access), so you can go straight
        from a loaded snapshot to a mock image.

        Parameters
        ----------
        snap : dict-like
            Must support `snap[pos_key]` and either `snap[luminosity_key]`
            (weight_type="luminosity"/"rate") or `snap[mass_key]`
            (weight_type="mass"). Apply any pygad ExprMask / sub-snapshot
            selection (e.g. `self.get_fov_mask`, star/gas cuts) to `snap`
            *before* passing it in here -- this function does the spatial
            FoV + LOS-depth cut itself via `project_particles`, but any
            physical selection (particle type, SF state, etc.) is on you.
        pos_key, luminosity_key, mass_key : str
            Field names to look up on `snap`.
        weight_type : {"luminosity", "rate", "mass"}
            "mass" bins raw mass and returns a noiseless, uncalibrated mass
            map (no PSF/noise applied) -- useful as a sanity check on
            projection/geometry, not a photometric image.
        (remaining parameters as in `image_from_particles`)

        Returns
        -------
        image : ndarray
        """
        pos = np.asarray(snap[pos_key])

        if weight_type == "mass":
            weights = np.asarray(snap[mass_key])
            px, py, keep = self.project_particles(pos, xaxis=xaxis, yaxis=yaxis)
            return self.bin_particles(px, py, weights[keep])

        weights = np.asarray(snap[luminosity_key])
        return self.image_from_particles(
            pos,
            weights,
            xaxis=xaxis,
            yaxis=yaxis,
            weight_type="luminosity" if weight_type == "luminosity" else "rate",
            z=z,
            flux_zeropoint=flux_zeropoint,
            exposure_time=exposure_time,
            apply_noise=apply_noise,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_image(self, image, ax=None, stretch="asinh", cmap="bone", title=None):
        """
        Display a simulated image with an astronomy-style stretch.

        Parameters
        ----------
        image : ndarray
            2D image to display (e.g. output of `observe`/`mock_observation`).
        ax : matplotlib Axes, optional
            Existing axes to plot into; a new figure is created if None.
        stretch : {"asinh", "linear", "log"}
            Intensity stretch to apply before display.
        cmap : str
            Matplotlib colormap name.
        title : str, optional
            Plot title; defaults to instrument name.

        Returns
        -------
        fig, ax : matplotlib Figure, Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        if self.full_well is not None:
            sat_frac = np.nanmean(image >= self.full_well)
            if sat_frac > 0.01:
                _logger.warning(
                    f"{sat_frac:.1%} of pixels are at or above full_well "
                    f"({self.full_well} e-). The image is saturated over a "
                    "significant area -- check flux_zeropoint, exposure_time, "
                    "or source luminosities. A saturated plateau this large "
                    "will dominate percentile-based stretches."
                )

        data = image - np.nanmedian(image)
        if stretch == "asinh":
            # Use a median-absolute-deviation scale rather than a fixed
            # percentile: a percentile can land inside a saturated plateau
            # when a sizeable fraction of pixels are clipped to full_well,
            # which crushes all unsaturated (background/source) contrast
            # to ~0. MAD stays anchored to the bulk of the distribution
            # as long as saturated pixels are a minority (<50%).
            mad = np.nanmedian(np.abs(data - np.nanmedian(data)))
            scale = 1.4826 * mad if mad > 0 else (np.nanstd(data) or 1.0)
            disp = np.arcsinh(data / scale)
        elif stretch == "log":
            disp = np.log10(np.clip(data - data.min() + 1.0, 1, None))
        else:
            disp = data

        vmin, vmax = np.nanpercentile(disp, [1, 99.5])
        ax.imshow(disp, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title or f"{self.name} mock observation")
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
        fig.tight_layout()
        return fig, ax


class _PhotometricInstrument2(BasicInstrument):
    def __init__(self, fov, sampling, res=None, z=None, label=None):
        super().__init__(fov, sampling, res=res, z=z)
        self.label = label or type(self).__name__
        self._filters = None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    def set_filters(self, filters):
        """
        Attach a filter set to this instrument.

        Parameters
        ----------
        filters : FilterCollection, list of Filter, or list of str
            - a `synthesizer.filters.FilterCollection` instance, or
            - a list of `synthesizer.filters.Filter` objects (generic
              top-hat filters, built with lam_min/lam_max -- no network
              access required), or
            - a list of SVO filter codes, e.g. "JWST/NIRCam.F200W"
              (fetched from the SVO filter service -- requires network
              access).
        """
        if isinstance(filters, FilterCollection):
            self._filters = filters
        elif filters and isinstance(filters[0], Filter):
            self._filters = FilterCollection(filters=list(filters))
        else:
            self._filters = FilterCollection(filter_codes=list(filters))

    @property
    def filters(self):
        if self._filters is None:
            _logger.exception(
                "Call set_filters(...) before requesting images.",
                exc_info=True,
            )
            raise RuntimeError
        return self._filters

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

    # ------------------------------------------------------------------
    # Low-level: image directly from a precomputed per-particle signal
    # ------------------------------------------------------------------
    def image_from_particles(
        self,
        pos,
        signal,
        xaxis="x",
        yaxis="y",
        filter_code="custom_band",
        img_type="hist",
        smoothing_lengths=None,
        kernel=None,
        kernel_threshold=1,
        angular=False,
        signal_units=erg / s / Hz,
    ):
        """
        Bin an already-computed per-particle signal (e.g. a luminosity
        you calculated yourself, as in `Instrument.image_from_particles`
        in instrument.py) into a Synthesizer ImageCollection, using this
        instrument's FoV/pixel geometry. Delegates the actual binning
        (and, for img_type="smoothed", SPH-kernel smoothing) to
        Synthesizer's own C-extension image generators.

        Parameters
        ----------
        pos : ndarray, shape (N, 3)
            Particle positions in kpc, centered on the object of interest.
        signal : ndarray, shape (N,)
            Per-particle signal to bin (e.g. luminosity in erg/s/Hz).
        xaxis, yaxis : int or str
            Projection axes.
        filter_code : str
            Label for the resulting image within the returned
            ImageCollection. Does not need to be a real filter for this
            manual path -- it's just a dict key.
        img_type : {"hist", "smoothed"}
            "hist": plain 2D histogram, equivalent to
                `Instrument.bin_particles` in instrument.py.
            "smoothed": SPH-kernel-smoothed image; requires
                `smoothing_lengths`, `kernel`, and `self.filters` to be set
                (Synthesizer's smoothed path wraps signal in a
                PhotometryCollection, which is filter-aware).
        smoothing_lengths : ndarray, shape (N,), optional
            Per-particle smoothing lengths in kpc. Required if
            img_type="smoothed".
        kernel : np.ndarray, optional
            SPH kernel lookup table, e.g. from
            `synthesizer.kernel_functions.Kernel().get_kernel()`.
            Required if img_type="smoothed".
        angular : bool
            If True, build the image in this instrument's angular
            (arcsec) geometry instead of physical (kpc) geometry.
        signal_units : unyt unit
            Units to attach to `signal` (default erg/s/Hz, a luminosity
            density; use erg/s/cm**2/Hz for a flux).

        Returns
        -------
        ImageCollection
        """
        pos = np.asarray(pos)
        signal = np.asarray(signal)

        x, y, keep = self._project_and_cut(pos, xaxis, yaxis)
        coords = self._to_instrument_units(x, y, angular=angular)
        sig = signal[keep] * signal_units

        resolution = self.synthesizer_resolution(angular=angular)
        fov = self.synthesizer_fov(angular=angular)
        imgcol = ImageCollection(resolution=resolution, fov=fov)

        if img_type == "hist":
            return imgcol.generate_imgs_hist(
                photometry={filter_code: sig}, coordinates=coords
            )

        if img_type == "smoothed":
            if smoothing_lengths is None or kernel is None:
                _logger.exception(
                    "smoothing_lengths and kernel are required for "
                    "img_type='smoothed'.",
                    exc_info=True,
                )
                raise ValueError

            sl = np.asarray(smoothing_lengths)[keep]
            if not angular:
                sl_q = sl * kpc
            else:
                scale = self.ang_scale.to("kpc/arcsec").value
                sl_q = (sl / scale) * arcsecond

            from synthesizer.photometry import PhotometryCollection

            # PhotometryCollection expects one row per filter; wrap our
            # single custom signal as a (1, N) array against self.filters.
            phot = PhotometryCollection(
                filters=self.filters, photometry=sig.reshape(1, -1)
            )
            return imgcol.generate_imgs_smoothed(
                photometry=phot,
                coordinates=coords,
                smoothing_lengths=sl_q,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
            )

        _logger.exception(f"Unknown img_type '{img_type}'", exc_info=True)
        raise ValueError

    # ------------------------------------------------------------------
    # Instrument object for PSF / noise post-processing
    # ------------------------------------------------------------------
    def build_imager(
        self, psfs=None, depth=None, snrs=None, noise_maps=None, angular=False
    ):
        """
        Build a Synthesizer PhotometricImager carrying this instrument's
        resolution plus optional PSFs/depth/SNR/noise maps, for use with
        `imager.apply_psfs(image_collection)` /
        `imager.apply_noises(image_collection)` on an ImageCollection
        (e.g. one returned by `image_from_particles`).
        """
        resolution = self.synthesizer_resolution(angular=angular)
        return PhotometricImager(
            label=self.label,
            filters=self.filters,
            resolution=resolution,
            psfs=psfs,
            depth=depth,
            snrs=snrs,
            noise_maps=noise_maps,
        )

    # ------------------------------------------------------------------
    # High-level: full SED-based imaging from a Synthesizer Galaxy
    # ------------------------------------------------------------------
    def image_from_galaxy(
        self,
        gal,
        spectra_type,
        img_type="smoothed",
        kernel=None,
        kernel_threshold=1,
        cosmo=None,
        angular=False,
        psfs=None,
        depth=None,
        snrs=None,
    ):
        """
        Full Synthesizer workflow: image a `synthesizer.particle.Galaxy`
        object that already has photometry computed (via an
        EmissionModel + `gal.get_photo_lnu(self.filters)`), using this
        instrument's geometry. This is the "real physics" path --
        luminosities come from actual stellar population synthesis (age,
        metallicity, initial mass -> SED -> filter convolution), not a
        luminosity array you supply yourself; use `image_from_particles`
        for that.

        Parameters
        ----------
        gal : synthesizer.particle.galaxy.Galaxy
            Galaxy object with photometry already computed for
            `spectra_type`.
        spectra_type : str or list of str
            Which computed spectra/photometry to image (e.g.
            "attenuated", "intrinsic", "incident").
        img_type : {"hist", "smoothed"}
        kernel, kernel_threshold : SPH kernel lookup + threshold; required
            for img_type="smoothed".
        cosmo : astropy.cosmology instance, optional
            Required if `angular=True` (converts particle coordinates to
            angular units using this instrument's redshift).
        angular : bool
            Use angular (arcsec) geometry rather than physical (kpc).
        psfs, depth, snrs : optional PSF/noise configuration; applied
            after image generation if given.

        Returns
        -------
        ImageCollection, or dict of ImageCollection if spectra_type is
        a list of labels.
        """
        imager = self.build_imager(psfs=psfs, depth=depth, snrs=snrs, angular=angular)
        fov = self.synthesizer_fov(angular=angular)

        imgs = gal.get_images_luminosity(
            spectra_type,
            instrument=imager,
            fov=fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            cosmo=cosmo,
        )

        if psfs is not None:
            imgs = imager.apply_psfs(imgs)
        if depth is not None or snrs is not None:
            imgs = imager.apply_noises(imgs)

        return imgs


class PhotometricInstrument(BasicInstrument):
    def __init__(self, fov, sampling, label, res=None, z=None, max_extent=40):
        super().__init__(fov, sampling, res, z, max_extent)
        self.galaxy = None
        self._filters = None
        self.label = label
        self._instr = None

    @property
    def filters(self):
        return self._filters

    @abstractmethod
    def get_filters(self, grid):
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
        if ages is None:
            # TODO check units
            ages = np.asarray(snap.stars["age"]) * yr
        elif isinstance(ages, (float, int)):
            ages = np.full(len(snap.stars), ages) * yr
        if metallicity is None:
            metallicity = snap.stars["metallicity"]
        elif isinstance(metallicity, (float, int)):
            metallicity = np.full(len(snap.stars), metallicity)
        if isinstance(softening, float):
            softening = np.full(len(snap.stars), softening * kpc)
        mask = self.get_fov_mask(xaxis, yaxis)
        snap = snap[mask]
        stars = synthesizer.particle.Stars(
            initial_masses=np.asarray(snap.stars["mass"]) * Msun,
            ages=ages,
            metallicities=metallicity,
            coordinates=np.asarray(snap.stars["pos"]) * kpc,
            centre=np.zeros(3) * kpc,
            softening_lengths=softening,
            redshift=self.redshift,
        )
        self.galaxy = synthesizer.particle.Galaxy(stars=stars, redshift=self.redshift)

    def generate_particle_spectra(
        self, grid_name="bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy-c23.01-sps.hdf5"
    ):
        grid = get_synthesizer_grid(
            grid_name=grid_name, new_lam=np.logspace(2, 5, 50) * angstrom
        )
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
        self.galaxy.stars.get_spectra(model)
        self.galaxy.get_observed_spectra(Planck18)
        self.galaxy.get_photo_lnu(self._filters)
        return grid

    def build_instrument(self, angular=False):
        self._instr = PhotometricImager(
            label=self.label,
            resolution=self.synthesizer_resolution(angular=angular),
            filters=self._filters,
        )

    def observe(self, ax=None, angular=False):
        kwargs = dict(
            instrument=self._instr,
            fov=self.synthesizer_fov(angular=angular),
            img_type="smoothed",
            kernel=Kernel().get_kernel(),
            cosmo=Planck18,
        )
        try:
            imgs = self.galaxy.get_images_luminosity("attenuated", **kwargs)
        except Exception as e:
            _logger.error(e)
            kwargs["img_type"] = "hist"
            imgs = self.galaxy.get_images_luminosity("attenuated", **kwargs)

        """if psfs is not None:
            imgs = imager.apply_psfs(imgs)
        if depth is not None or snrs is not None:
            imgs = imager.apply_noises(imgs)"""

        if ax is None:
            nax = len(self.filters)
            fig, ax = plt.subplots(ncols=nax, sharex="all", sharey="all")
            if nax == 1:
                ax = [ax]
        cmapper = NormedColours.from_array_list(
            [v.arr for v in imgs.values()], norm="LogNorm", cmap="bone"
        )
        for axi, fcode in zip(ax, self.filters.filter_codes):
            axi.imshow(imgs[fcode].arr, norm=cmapper.norm, cmap=cmapper.cmap)
            axi.set_title(fcode)
            axi.set_facecolor("k")
        return ax


class Euclid_VIS(PhotometricInstrument):
    """Euclid VIS imaging channel (broad optical band, ~550-900 nm)."""

    def __init__(self, z=None):
        super().__init__(
            fov=0.787 * 3600,  # ~0.787 deg per detector -> arcsec (illustrative)
            sampling=0.101,  # arcsec/pixel
            res=0.16,  # PSF FWHM ~0.16"
            z=z,
            label="EuclidVIS",
            # psf_fwhm=0.16,
            # psf_type="gaussian",
            # read_noise=4.5,  # e-
            # dark_current=0.001,  # e-/s/pix
            # sky_background=0.0015,  # e-/s/pix (approximate, zodiacal-dominated)
            # zeropoint=24.7,  # AB mag for 1 e-/s (approximate)
            # gain=1.0,
            # full_well=200000,
            # exposure_time=565.0,  # s, single VIS exposure
        )
        # self.label = r"$\mathrm{Euclid-VIS}$"

    def get_filters(self, grid):
        return get_euclid_filter_collection(grid)


class HSTWFC3(PhotometricInstrument):
    """HST WFC3/UVIS, F606W-like broad V band."""

    def __init__(self, z=None):
        super().__init__(
            fov=162.0,  # arcsec, UVIS field of view
            sampling=0.04,  # arcsec/pixel
            res=0.07,  # diffraction-limited PSF FWHM, approximate
            z=z,
            label="HSTWFC3",
            # psf_fwhm=0.07,
            # psf_type="moffat",
            # moffat_beta=3.0,
            # read_noise=3.1,  # e-
            # dark_current=0.0153,  # e-/s/pix
            # sky_background=0.03,  # e-/s/pix, approximate
            # zeropoint=26.5,  # AB mag for 1 e-/s (approximate, filter-dependent)
            # gain=1.5,
            # full_well=63000,
            # exposure_time=1200.0,  # s
        )

    def get_filters(self, grid):
        return get_hst_filter_collection(grid)


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
            res=0.2,
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
            res=20e-3,
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
            res=68e-3,
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


class MICADO_WFM(LongSlitInstrument):
    def __init__(self, rng=None, z=None):
        """
        MICADO for ELT
        https://elt.eso.org/instrument/MICADO/
        """
        super().__init__(
            fov=50.5,
            sampling=4e-3,
            res=50e-6,
            slit_width=16e-3,
            slit_length=3,
            rng=rng,
            z=z,
        )
        self.label = r"$\mathrm{MICADO}$"


class MICADO_NFM(LongSlitInstrument):
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


class VLT_FORS2(LongSlitInstrument):
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


class ERIS_NIX_NFM(LongSlitInstrument):
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
            res=68e-3,
            slit_width=0.2,
            slit_length=3.2,
            z=z,
            rng=rng,
        )
        self.label = r"$\mathrm{JWST}$"

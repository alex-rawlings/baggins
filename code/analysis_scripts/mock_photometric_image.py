import pygad
import baggins as bgs


def original_driver(snapfile, redshift):
    snap = pygad.Snapshot(snapfile, physical=True)
    sed = bgs.analysis.get_spectrum_ssp(1e9, 0.03396)[1]
    bgs.analysis.set_luminosity(snap, sed=sed, z=redshift)

    inst = bgs.analysis.HSTWFC3(z=redshift)

    inst.flux_zeropoint = 1e-2  # erg/s/cm^2 for 1 e-/s -- calibrate for real use
    mask = inst.get_fov_mask(0, 1)
    stars = snap.stars[mask]
    img = inst.image_from_snapshot(
        stars,
        xaxis="x",
        yaxis="y",
        weight_type="luminosity",
    )
    fig, ax = inst.plot_image(img, title="Snapshot mock observation", stretch="log")
    bgs.plotting.savefig("test.png")


"""
run_mock_image.py

Driver script for generating mock photometric images using
SynthesizerInstrument (see synthesizer_instrument.py).

Two workflows are shown:

  1. quick_demo() -- runs immediately, no SPS grid download needed.
     Bins an already-computed per-particle luminosity directly into an
     image via `image_from_particles`, then applies a PSF and noise.
     This is the workflow to reach for if you already have per-particle
     luminosities (as in instrument.py's from-scratch Instrument class).

  2. full_pipeline_demo() -- the complete SED-based workflow. Requires
     a downloaded SPS grid and your real snapshot data. Shown as a
     template to adapt, not meant to run as-is.
"""

import numpy as np
import matplotlib.pyplot as plt
from unyt import erg, s, Hz, angstrom

from synthesizer.filters import Filter
from synthesizer.kernel_functions import Kernel


def _gaussian_psf(fwhm_pix, size):
    """Build a small normalized Gaussian PSF kernel for demo purposes."""
    sigma = fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    y, x = np.mgrid[0:size, 0:size]
    c = size // 2
    psf = np.exp(-0.5 * ((x - c) ** 2 + (y - c) ** 2) / sigma**2)
    return psf / psf.sum()


def plot_image(image, title=""):
    """Robust asinh display (see instrument.py's plot_image for why the
    MAD-based scale matters when a chunk of the image can be saturated
    or zero)."""
    data = image - np.nanmedian(image)
    mad = np.nanmedian(np.abs(data - np.nanmedian(data)))
    scale = 1.4826 * mad if mad > 0 else (np.nanstd(data) or 1.0)
    disp = np.arcsinh(data / scale)
    vmin, vmax = np.nanpercentile(disp, [1, 99.5])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(disp, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    fig.tight_layout()
    return fig, ax


def quick_demo():
    """End-to-end with synthetic particle data -- no grid/network needed.
    Good for testing the instrument/imaging plumbing before committing
    to a full SPS run."""
    rng = np.random.default_rng(42)

    inst = bgs.analysis.Euclid_VIS(z=0.3)

    # Generic top-hat filter standing in for the VIS band -- no SVO
    # lookup required. Swap for a real SVO filter code (e.g.
    # "Euclid/VIS.vis") once you have network access to fetch it.
    vis_filter = Filter(
        "VIS",
        lam_min=5500 * angstrom,
        lam_max=9000 * angstrom,
        new_lam=np.linspace(3000, 12000, 300) * angstrom,
    )
    inst.set_filters([vis_filter])

    # Fake particle distribution: a mild disk-like clump, thin along
    # the line of sight, with a range of smoothing lengths.
    N = 2000
    pos = np.zeros((N, 3))
    pos[:, 0] = rng.normal(0, 6, N)
    pos[:, 1] = rng.normal(0, 6, N)
    pos[:, 2] = rng.normal(0, 1.5, N)
    smoothing_lengths = rng.uniform(0.3, 1.2, N)
    luminosity = rng.uniform(1e39, 1e42, N)  # erg/s/Hz, stand-in values

    # SPH-kernel-smoothed image (the physically correct rendering of
    # finite particle resolution -- see the previous conversation's
    # comparison of this against a naive point-binned histogram).
    kernel_data = Kernel().get_kernel()
    imgs = inst.image_from_particles(
        pos,
        luminosity,
        xaxis="x",
        yaxis="y",
        filter_code="VIS",
        img_type="smoothed",
        smoothing_lengths=smoothing_lengths,
        kernel=kernel_data,
    )

    # PSF + noise via the same instrument object.
    # NOTE: `depth` must carry the same kind of units as the image
    # itself (a luminosity density here, since `signal_units` above was
    # erg/s/Hz) -- not an AB magnitude. For flux-calibrated images
    # (signal_units=erg/s/cm**2/Hz) depth would be in those units instead.
    psf = _gaussian_psf(fwhm_pix=4, size=25)
    imager = inst.build_imager(
        psfs={"VIS": psf},
        depth={"VIS": 5e39 * erg / s / Hz},
        snrs={"VIS": 5.0},
    )
    imgs = imager.apply_psfs(imgs)
    imgs = imager.apply_noises(imgs)

    plot_image(imgs["VIS"].arr, title="Synthesizer mock image (quick demo)")
    plt.show()
    return imgs


def full_pipeline_demo(snapshot_path, z=0.3):
    """
    The complete SED-based workflow: real stellar population synthesis
    driving the photometry, rather than a luminosity you supply
    yourself. Requires:
      - a downloaded SPS grid (see Synthesizer's grid-download docs)
      - network access to the SVO filter service for real filter codes
      - your snapshot's star particle ages/metallicities/initial masses,
        not just mass/position/velocity

    This is a template to adapt to your own snapshot reader (see
    instrument.py's `image_from_snapshot` for the pygad-only
    equivalent) -- it is not meant to run unmodified.
    """
    from astropy.cosmology import Planck18 as cosmo
    from unyt import Msun, Myr, kpc
    from synthesizer.grid import Grid
    from synthesizer.particle.stars import Stars
    from synthesizer.particle.galaxy import Galaxy
    from synthesizer.emission_models import BimodalPacmanEmission
    from synthesizer.emission_models.attenuation import PowerLaw

    # --- 1. load your snapshot's star particle data ---
    # Replace this block with your own pygad-based reader; Stars()
    # needs initial_masses, ages, metallicities (and, for smoothed
    # imaging, coordinates + smoothing_lengths).
    import pygad

    snap = pygad.Snapshot(snapshot_path, physical=True)
    stars_sel = snap.stars

    stars = Stars(
        initial_masses=np.asarray(stars_sel["mass"]) * Msun,
        ages=np.asarray(stars_sel["age"]) * Myr,
        metallicities=np.asarray(stars_sel["metallicity"]),
        coordinates=np.asarray(stars_sel["pos"]) * kpc,
        smoothing_lengths=np.asarray(stars_sel["hsml"]) * kpc,
        redshift=z,
    )
    gal = Galaxy(stars=stars, redshift=z)

    # --- 2. generate spectra + photometry ---
    grid = Grid("test_grid")  # replace with your downloaded grid name
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
    gal.stars.get_spectra(model)
    gal.get_observed_spectra(cosmo)

    inst = bgs.analysis.Euclid_VIS(z=z)
    inst.set_filters(["Euclid/VIS.vis"])  # SVO code, needs network access
    gal.get_photo_lnu(inst.filters)

    # --- 3. image ---
    kernel_data = Kernel().get_kernel()
    # NOTE: depth must be in flux-density units matching the image
    # (erg/s/cm**2/Hz for observed/flux photometry) -- not an AB magnitude.
    from unyt import erg as _erg, s as _s, cm as _cm, Hz as _Hz

    imgs = inst.image_from_galaxy(
        gal,
        "attenuated",
        img_type="smoothed",
        kernel=kernel_data,
        cosmo=cosmo,
        psfs={"Euclid/VIS.vis": _gaussian_psf(4, 25)},
        depth={"Euclid/VIS.vis": 1e-30 * _erg / _s / _cm**2 / _Hz},
        snrs={"Euclid/VIS.vis": 5.0},
    )

    plot_image(
        imgs["Euclid/VIS.vis"].arr, title="Synthesizer mock image (full SED pipeline)"
    )
    plt.show()
    return imgs


def my_pipeline(snapfile, z):
    snap = pygad.Snapshot(snapfile, physical=True)
    bgs.analysis.basic_snapshot_centring(snap)
    snap = snap[pygad.BallMask(2)]
    instr = bgs.analysis.HSTWFC3(z=z)
    instr.load_and_project_galaxy(snap, ages=1e9, metallicity=0.03396, softening=5e-3)
    instr.generate_particle_spectra()
    instr.build_instrument()
    instr.observe()
    bgs.plotting.savefig("test.png")


if __name__ == "__main__":
    # quick_demo()

    # Requires a downloaded SPS grid + your real snapshot -- see the
    # docstring above before uncommenting.
    snapfile = "/orion/ptmp/arawling/recoil-sims/k0540_snap_009.hdf5"
    redshift = 0.6
    # full_pipeline_demo(snapfile)
    my_pipeline(snapfile, redshift)

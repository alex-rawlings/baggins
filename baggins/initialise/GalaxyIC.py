import os.path
from datetime import datetime
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import merger_ic_generator as mg
import pygad
from baggins.env_config import _cmlogger, date_format
from baggins.analysis import projected_quantities
from baggins.mathematics import (
    uniform_sample_sphere,
    EmpiricalCDF,
    get_histogram_bin_centres,
)
from baggins.plotting import make_wide_figure, savefig
from baggins.utils import read_parameters, write_calculated_parameters
from baggins.literature import (
    LiteratureTables,
    Sahu19,
    Moster10,
    Girelli20,
    Behroozi19,
    Duffy08,
    zlochower_cold_spins,
    zlochower_dry_spins,
    zlochower_hot_spins,
)
from baggins.initialise.ic_helpers import ensure_reasonable_particle_counts

__all__ = ["GalaxyIC"]

_logger = _cmlogger.getChild(__name__)


scaling_relation_mapping = dict(
    moster=Moster10,
    girelli=Girelli20,
    behroozi=Behroozi19,
    sahu=lambda x: 10 ** Sahu19(x),
)
bh_spin_models = dict(
    dry=zlochower_dry_spins, cold=zlochower_cold_spins, hot=zlochower_hot_spins
)

MSUN_TO_GADGET = 1e-10


def convert_msun_to_gadget(d):
    for k in ("mass", "particle_mass"):
        if k in d.keys():
            d[k] = d[k] * MSUN_TO_GADGET
    return d


class GalaxyIC:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.pars = read_parameters(self.parameter_file)
        self.name = self.pars["general"].pop("galaxy_name")
        self._center_CoM = self.pars["general"].pop("center_CoM")
        self.file_pars = self.pars["file_locations"]
        self.hdf5_file_name = os.path.join(
            self.file_pars["save_location"], f"{self.name}/{self.name}.hdf5"
        )
        try:
            self._rng = np.random.default_rng(self.pars["general"].pop("random_seed"))
        except KeyError:
            self._rng = np.random.default_rng()
        self.components = []
        self.anisotropy_radius = None
        self._stellar_mass = None
        self._dm_mass = None
        self._bh_mass = None
        self._calc_quants = {}
        self.marker_kwargs = {
            "ls": "",
            "marker": "o",
            "mec": "k",
            "mew": 0.5,
            "label": "IC",
        }

    def _set_up(self):
        # set up components
        # stars
        try:
            star_pars = self.pars["stars"]
            self._calc_quants["stars"] = {}
            self.anisotropy_radius = star_pars.pop("anisotropy_radius", None)
            profile_type = star_pars.pop("profile")
            self._stellar_mass = star_pars["mass"]
            star_pars = convert_msun_to_gadget(star_pars)
            _logger.debug(f"Stellar parameters are {star_pars}")
            if profile_type.lower() == "dehnen":
                c = mg.DehnenSphere(**star_pars, particle_type=mg.ParticleType.STARS)
            else:
                c = mg.GenericSphericalComponent(
                    **star_pars, particle_type=mg.ParticleType.STARS
                )
            self.components.append(c)
        except KeyError:
            _logger.warning("No stellar component")

        # DM halo
        try:
            dm_pars = self.pars["dm"]
            self._calc_quants["dm"] = {}
            profile_type = dm_pars.pop("profile")
            if "mass" not in dm_pars.keys():
                try:
                    dm_pars["mass"] = scaling_relation_mapping[
                        dm_pars.pop("mass_relation")
                    ](self._stellar_mass)
                except KeyError:
                    _logger.exception(
                        "key 'mass_relation' must be specified to determine DM mass from stellar mass scaling relation if key 'mass' is not provided.",
                        exc_info=True,
                    )
                    raise
            self._dm_mass = dm_pars["mass"]
            dm_pars = convert_msun_to_gadget(dm_pars)
            if profile_type.lower() == "nfw":
                dm_pars.setdefault("z", 0)
                dm_pars["M200"] = dm_pars.pop("mass")
                dm_pars.setdefault(
                    "concentration", Duffy08(dm_pars["M200"], dm_pars["z"])
                )
                _logger.debug(f"DM parameters are {dm_pars}")
                c = mg.NFWSphere(**dm_pars, particle_type=mg.ParticleType.DM_HALO)
            else:
                _logger.debug(f"DM parameters are {dm_pars}")
                c = mg.DehnenSphere(**dm_pars, particle_type=mg.ParticleType.DM_HALO)
            self.components.append(c)
        except KeyError:
            _logger.warning("No DM component")

        # SMBH
        try:
            bh_pars = self.pars["bh"]
            self._calc_quants["bh"] = {}
            if bh_pars["mass"] is None:
                bh_pars["mass"] = scaling_relation_mapping["sahu"](
                    np.log10(self._stellar_mass)
                )
            self._bh_mass = bh_pars["mass"]
            bh_pars = convert_msun_to_gadget(bh_pars)
            print(f"{self._bh_mass:.2e}")
            if not isinstance(bh_pars["spin"], list):
                bh_spin_pars = bh_spin_models[bh_pars["spin"]]
                _logger.info(f"Generating BH spins from {bh_pars["spin"]}")
                spin_mag = scipy.stats.beta.rvs(
                    *bh_spin_pars.values(), random_state=self._rng
                )
                t, p = uniform_sample_sphere(1, rng=self._rng)
                bh_pars["spin"] = (
                    spin_mag
                    * np.array(
                        [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
                    ).flatten()
                )
                bh_pars["chi"] = bh_pars.pop("spin")
                self._calc_quants["bh"]["chi"] = bh_pars["chi"]
                _logger.debug(f"BH parameters are {bh_pars}")
                c = mg.CentralPointMass(**bh_pars, particle_type=mg.ParticleType.BH)
                self.components.append(c)
        except KeyError:
            _logger.warning("No BH component")

    @classmethod
    def load_from_hdf5(cls, parameter_file):
        C = cls(parameter_file=parameter_file)
        return C

    def fig_loc(self, fname):
        return os.path.join(
            self.file_pars["save_location"],
            self.name,
            self.file_pars["figure_location"],
            f"{self.name}_{fname}",
        )

    def write_calculated_parameters(self):
        """
        Write calculated parameters to the parameter file
        """
        now = datetime.now()
        self._calc_quants["last_update"] = now.strftime(date_format)
        write_calculated_parameters(self._calc_quants, self.parameter_file)

    def generate_galaxy(self, allow_overwrite=False):
        """
        Generate the initial conditions.

        Parameters
        ----------
        allow_overwrite : bool, optional
            allow overwriting of the IC file, by default False
        """
        self._set_up()
        self.pars["general"]["rng"] = self._rng
        try:
            assert np.all([c.enclosed_mass(1e10) > 0 for c in self.components])
        except AssertionError:
            _logger.exception("Negative masses detected!", exc_info=True)
            raise
        if self.anisotropy_radius is None:
            gal = mg.ErgodicSphericalSystem(*self.components, **self.pars["general"])
        else:
            gal = mg.AnisotropicSphericalSystem(
                *self.components, ra=self.anisotropy_radius, **self.pars["general"]
            )

        # clean centre
        if self._bh_mass is not None:
            gal = mg.TransformedSystem(
                gal,
                mg.FilterParticlesBoundToCentralMass(
                    central_object_mass=self._bh_mass / MSUN_TO_GADGET,
                    minimum_semi_major_axis=self.pars["general"]["rmin"],
                ),
            )

        # ensure no particles dropped
        ensure_reasonable_particle_counts(gal, 1e3)

        # save galaxy
        try:
            assert not os.path.exists(self.hdf5_file_name) or allow_overwrite
            mg.write_hdf5_ic_file(self.hdf5_file_name, gal, center_CoM=self._center_CoM)
            self.write_calculated_parameters()
            _logger.info(f"IC file {self.hdf5_file_name} created")
        except AssertionError:
            _logger.exception(
                f"File {self.hdf5_file_name} already exists! Overwriting not allowed when 'allow_overwrite' is False.",
                exc_info=True,
            )
            raise

    def _load_ic_file(self):
        # read in the IC file as a snapshot
        snap = pygad.Snapshot(self.hdf5_file_name, physical=True)
        stellar_mass = np.sum(snap.stars["mass"])
        log_stellar_mass = np.log10(stellar_mass)
        log_dm_mass = np.log10(np.sum(snap.dm["mass"]))
        return snap, stellar_mass, log_stellar_mass, log_dm_mass

    def plot_mass_scaling_relations(self):
        """
        Plot the stellar mass distribution, and the scaling relations of BH mass -- bulge mass and bulge mass -- DM mass.
        """
        snap, stellar_mass, log_stellar_mass, log_dm_mass = self._load_ic_file()
        # read in literature data
        mass_data = LiteratureTables.load_sdss_mass_data()
        bh_data = LiteratureTables.load_sahu_2020_data()

        # set up figure
        fig, ax = plt.subplots(1, 3)
        make_wide_figure(fig)

        # plot bulge mass distribution
        for q, ls in zip((0.5, 0.16, 0.84), ("-", ":", ":")):
            ax[0].axhline(q, ls=ls, c="k")
        ecdf = EmpiricalCDF(mass_data.table["logMstar"])
        ecdf.plot(ax=ax[0], label="SDSS ETGs")
        ax[0].plot(log_stellar_mass, ecdf.cdf(log_stellar_mass), **self.marker_kwargs)
        ax[0].legend(loc="upper left")

        # plot bh - bulge relation
        bh_data.scatter(
            "logM*_sph",
            "logMbh",
            xerr="logM*_sph_ERR",
            yerr="logMbh_ERR",
            ax=ax[1],
            mask=bh_data.table.loc[:, "Cored"],
            use_label=False,
            scatter_kwargs={"label": "Cored"},
        )
        _, eb = bh_data.scatter(
            "logM*_sph",
            "logMbh",
            xerr="logM*_sph_ERR",
            yerr="logMbh_ERR",
            ax=ax[1],
            mask=~bh_data.table.loc[:, "Cored"],
            scatter_kwargs={"marker": "s", "label": r"Sersic"},
            use_label=False,
        )
        logmstar_seq = np.linspace(8, 12, 500)
        ax[1].plot(logmstar_seq, Sahu19(logmstar_seq), c="k", alpha=0.4)
        ax[1].plot(
            log_stellar_mass,
            np.log10(snap.bh["mass"]),
            **self.marker_kwargs,
            zorder=eb.lines[0].get_zorder() + 0.2,
        )
        ax[1].legend(loc="upper left")

        # plot bulge mass - DM halo mass relation
        for halo_func, label in zip(
            (Moster10, Girelli20, Behroozi19),
            ("Moster+10", "Girelli+20", "Behroozi+19"),
        ):
            halo_mass_seq, log_star_seq = halo_func(
                stellar_mass, [1e10, 1e15], plotting=True
            )
            halo_mass_seq = np.log10(halo_mass_seq)
            ax[2].plot(halo_mass_seq, log_star_seq - halo_mass_seq, label=label)
        ax[2].plot(log_dm_mass, log_stellar_mass - log_dm_mass, **self.marker_kwargs)
        ax[2].legend(loc="lower right")

        # set desired axis labels
        ax[0].set_xlabel(r"log(M$_\mathrm{bulge}$ / M$_\odot)$")
        ax[0].set_ylabel(r"ECDF")
        ax[0].set_title("SDSS Bulge Mass Distribution")
        ax[1].set_xlim(8.7, 12.5)
        ax[1].set_ylim(7.2, 11)
        ax[1].set_xlabel(r"log(M$_\mathrm{bulge}$/M$_\odot$)")
        ax[1].set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
        ax[1].set_title("Bulge - BH Mass")
        ax[2].set_xlabel(r"log(M$_\mathrm{halo}$/M$_\odot$)")
        ax[2].set_ylabel(r"log(M$_\mathrm{stellar}$/M$_\mathrm{halo}$)")
        ax[2].set_title(r"M$_\mathrm{stellar} - $M$_\mathrm{DM}$")

        savefig(self.fig_loc("masses.png"))

    def plot_kinematics(self, num_rots=3):
        """
        Plot kinematic properties of the ICs to check for consistency with
        observations.

        Parameters
        ----------
        num_rots : int, optional
            number of rotations performed for projected quantities, by default 3
        """
        self._calc_quants["kinematics"] = {}
        # load literature data
        bulgeBHData = LiteratureTables.load_sahu_2020_data()
        fDMData = LiteratureTables.load_jin_2020_data()
        BHsigmaData = LiteratureTables.load_vdBosch_2016_data()

        # load IC file as snapshot
        snap, stellar_mass, log_stellar_mass, log_dm_mass = self._load_ic_file()

        # set up figure
        fig, ax = plt.subplot_mosaic(
            """
        ACEG
        BDFH
        """
        )
        make_wide_figure(fig)

        # density profiles
        radial_bin_edges = dict(
            stars=np.geomspace(1e-2, 1e2, 51),
            dm=np.geomspace(10, self.pars["general"]["rmax"], 51),
        )
        for fam, axk in zip(("stars", "dm"), "AB"):
            if hasattr(snap, fam):
                subsnap = getattr(snap, fam)
                ax[axk].loglog(
                    get_histogram_bin_centres(radial_bin_edges[fam]),
                    pygad.analysis.profile_dens(
                        subsnap, "mass", r_edges=radial_bin_edges[fam]
                    ),
                )
            ax[axk].set_xlabel(r"$r\mathrm{kpc}$")
            ax[axk].set_ylabel(r"$\rho / (\mathrm{M}_\odot\,\mathrm{kpc}^{-3}$")
            ax[axk].set_title(f"{fam} density (3D)")

        # projected quantities
        eff_rad, vsig2_Re, *_ = projected_quantities(snap, obs=num_rots)
        eff_rad = np.nanmedian(list(eff_rad.values())[0])
        vsig2_Re = np.nanmedian(list(vsig2_Re.values())[0])
        self._calc_quants["kinematics"]["projected_half_mass_radius"] = {
            "unit": "kpc",
            "value": float(eff_rad),
        }
        # use an unbiased estimator of standard deviation
        LOS_sigma = np.sqrt(vsig2_Re)
        self._calc_quants["kinematics"]["LOS_velocity_dispersion"] = {
            "unit": "km/s",
            "value": float(LOS_sigma),
        }

        # plot of stellar mass against half mass radius
        ax["C"].set_xlabel(r"log(R$_\mathrm{e, sph}/$kpc)")
        ax["C"].set_ylabel(r"log(M$_\mathrm{*,sph}$ / M$_\odot$)")
        ax["C"].set_title(r"R$_\mathrm{e}$ - log(M$_*$) Relation", fontsize="small")
        _, eb = bulgeBHData.scatter(
            "log_Re_maj_kpc",
            "logM*_sph",
            yerr="logM*_sph_ERR",
            ax=ax["C"],
        )
        bulgeBHData.plot_lin_regress(
            "log_Re_maj_kpc", "logM*_sph", ax=ax["C"], plot_scatter=False
        )
        hmr = pygad.analysis.half_mass_radius(snap.stars)
        self._calc_quants["kinematics"]["half_mass_radius"] = hmr
        ax["C"].plot(
            np.log10(hmr),
            log_stellar_mass,
            zorder=eb.lines[0].get_zorder() + 0.2,
            **self.marker_kwargs,
        )
        ax["C"].legend()

        # inner dark matter
        ax["D"].set_ylim(0, 1)
        ax["D"].set_xlabel(r"log(M$_*$/M$_\odot$)")
        ax["D"].set_ylabel(r"f$_\mathrm{DM}(r<1\,$R$_\mathrm{e})$")
        ax["D"].set_title("Inner DM Fraction", fontsize="small")

        binned_fdm = scipy.stats.binned_statistic(
            fDMData.table.loc[:, "log(M*/Msun)"],
            values=fDMData.table.loc[:, "f_DM"],
            bins=5,
            statistic="median",
        )
        _, eb = fDMData.scatter("log(M*/Msun)", "f_DM", ax=ax["D"])
        fdm_radii = get_histogram_bin_centres(binned_fdm[1])
        ax["D"].plot(fdm_radii, binned_fdm[0], "-x", label="Median")
        ball_mask = pygad.BallMask(eff_rad)
        inner_dm_mass = np.sum(snap.dm[ball_mask]["mass"])
        idmf = inner_dm_mass / (inner_dm_mass + np.sum(snap.stars[ball_mask]["mass"]))
        self._calc_quants["kinematics"]["inner_DM_frac"] = idmf
        ax["D"].plot(
            log_stellar_mass,
            idmf,
            zorder=eb.lines[0].get_zorder() + 0.2,
            **self.marker_kwargs,
        )
        ax["D"].legend(loc="upper left")

        # virial info
        vr, vm = pygad.analysis.virial_info(snap, N_min=10)
        self._calc_quants["kinematics"]["virial_info"] = {
            "mass": {"unit": "Msol", "value": float(vm)},
            "radius": {"unit": "kpc", "value": float(vr)},
        }

        # histogram of LOS velocities
        ax["E"].ticklabel_format(
            axis="y", style="scientific", scilimits=(0, 0), useMathText=True
        )
        ax["E"].hist(snap.stars["vel"][:, 2].ravel(), 50, density=True)
        ax["E"].set_xlabel(r"V$_*$ [km/s]")
        ax["E"].set_ylabel("Density")
        ax["E"].set_title("Stellar Velocity", fontsize="small", loc="right")

        # MBH-sigma relation
        ax["F"].set_xlabel(r"log($\sigma_*$/ km/s)")
        ax["F"].set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
        ax["F"].set_title("BH Mass - Stellar Dispersion", fontsize="small")

        _, eb = BHsigmaData.scatter(
            "logsigma",
            "logBHMass",
            xerr="e_logsigma",
            yerr=["e_logBHMass", "E_logBHMass"],
            ax=ax["F"],
            # use_label=False
        )
        BHsigmaData.plot_lin_regress(
            "logsigma", "logBHMass", ax=ax["F"], plot_scatter=False
        )
        ax["F"].plot(
            np.log10(LOS_sigma),
            np.log10(snap.bh["mass"]),
            zorder=eb.lines[0].get_zorder() + 0.2,
            **self.marker_kwargs,
        )
        ax["F"].legend()

        # BH spin distribution
        if not isinstance(self.pars["bh"]["spin"], list):
            spin_mag = scipy.stats.beta(
                *bh_spin_models[self.pars["bh"]["spin"]].values()
            )
        else:
            spin_mag = scipy.stats.uniform(0, 1)

        spin_seq = np.linspace(0, 1, 1000)
        # TODO convert units correctly
        print(pygad.physics.G, pygad.physics.c)
        print(f"BH spins {snap.bh["Spins"]}")
        print(snap.bh["angmom"].units)
        bhspin_mag = np.linalg.norm(snap.bh["Spins"]) * (snap.bh["mass"])
        ax["G"].plot(spin_seq, spin_mag.pdf(spin_seq), **self.marker_kwargs)
        _logger.info(f"SMBH spin magnitude: {bhspin_mag}")
        ax["G"].scatter(bhspin_mag, spin_mag.pdf(bhspin_mag), zorder=10)
        ax["G"].set_title(r"BH $\chi$", fontsize="small")
        ax["G"].set_xlabel(r"$\chi$")
        ax["G"].set_ylabel("PDF")

        # save figure
        savefig(self.fig_loc("kinematics.png"))
        self.write_calculated_parameters()

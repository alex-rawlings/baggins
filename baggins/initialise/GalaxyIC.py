import os
import shutil
from copy import copy
from datetime import datetime
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import merger_ic_generator as mg
import pygad
from baggins.env_config import TMPDIRs, _cmlogger, date_format
from baggins.analysis import (
    projected_quantities,
    velocity_anisotropy,
    inner_DM_fraction,
)
from baggins.mathematics import (
    uniform_sample_sphere,
    EmpiricalCDF,
    get_histogram_bin_centres,
    quantiles_relative_to_median,
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
        os.makedirs(os.path.dirname(self.hdf5_file_name), exist_ok=True)
        os.makedirs(os.path.dirname(self.fig_loc("")), exist_ok=True)
        try:
            self._rng = np.random.default_rng(self.pars["general"].pop("random_seed"))
        except KeyError:
            _logger.warning("No random seed set")
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
        self.errorbar_kwargs_data = {
            "fmt": ".",
            "markersize": 2,
            "mew": 0,
            "zorder": 0.5,
        }

    def _set_up(self):
        # set up components
        # stars
        try:
            star_pars = copy(self.pars["stars"])
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
            dm_pars = copy(self.pars["dm"])
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
                    "concentration",
                    Duffy08(dm_pars["M200"] / MSUN_TO_GADGET, dm_pars["z"]),
                )
                _logger.debug(f"DM parameters are {dm_pars}")
                self._calc_quants["dm"]["concentration"] = dm_pars["concentration"]
                c = mg.NFWSphere(**dm_pars, particle_type=mg.ParticleType.DM_HALO)
            else:
                _logger.debug(f"DM parameters are {dm_pars}")
                c = mg.DehnenSphere(**dm_pars, particle_type=mg.ParticleType.DM_HALO)
            self.components.append(c)
        except KeyError:
            _logger.warning("No DM component")

        # SMBH
        try:
            bh_pars = copy(self.pars["bh"])
            self._calc_quants["bh"] = {}
            if bh_pars["mass"] is None:
                bh_pars["mass"] = scaling_relation_mapping["sahu"](
                    np.log10(self._stellar_mass)
                )
                self._calc_quants["bh"]["mass"] = bh_pars["mass"]
            self._bh_mass = bh_pars["mass"]
            bh_pars = convert_msun_to_gadget(bh_pars)
            if not isinstance(bh_pars["spin"], list):
                bh_spin_pars = bh_spin_models[bh_pars["spin"]]
                _logger.info(f"Generating BH spins from {bh_pars["spin"]} distribution")
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

    def generate_galaxy(self, *transforms, allow_overwrite=False, plot_df=False):
        """
        Generate the initial conditions.

        Parameters
        ----------
        transforms : , optional
            transformations to apply to the system from merger-ic-generator, e.g. Translations
        allow_overwrite : bool, optional
            allow overwriting of the IC file, by default False
        plot_df : bool, optional
            plot the distribution function, by default False
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
            transforms = transforms + (
                mg.FilterParticlesBoundToCentralMass(
                    central_object_mass=self._bh_mass / MSUN_TO_GADGET,
                    minimum_semi_major_axis=self.pars["general"]["rmin"],
                ),
            )
            """gal = mg.TransformedSystem(
                gal,
                mg.FilterParticlesBoundToCentralMass(
                    central_object_mass=self._bh_mass / MSUN_TO_GADGET,
                    minimum_semi_major_axis=self.pars["general"]["rmin"],
                ),
            )"""

        # apply transformations
        gal = mg.TransformedSystem(gal, *tuple(t for t in transforms if t is not None))

        # ensure no particles dropped
        ensure_reasonable_particle_counts(gal, 1e3)

        # save galaxy
        try:
            assert not os.path.exists(self.hdf5_file_name) or allow_overwrite
            mg.write_hdf5_ic_file(self.hdf5_file_name, gal, center_CoM=self._center_CoM)
            self.write_calculated_parameters()
            _logger.info(f"IC file {self.hdf5_file_name} created")
            # copy parameter file to simulation directory
            shutil.copyfile(
                self.parameter_file,
                os.path.join(
                    os.path.dirname(self.hdf5_file_name),
                    os.path.basename(self.parameter_file),
                ),
            )
        except AssertionError:
            _logger.exception(
                f"File {self.hdf5_file_name} already exists! Overwriting not allowed when 'allow_overwrite' is False.",
                exc_info=True,
            )
            raise
        if plot_df:
            mg.plot_distribution_function(gal, self.fig_loc("df.png"))

    def generate_galaxy_components_separately(self, allow_overwrite=False):
        """
        Generate the components of a galaxy independently from each other. Note physical consistency of the system is not guaranteed.

        Parameters
        ----------
        allow_overwrite : bool, optional
            allow overwriting of the IC file, by default False
        """
        self._set_up()
        _logger.warning(
            "Generating galaxy components indepdently, physical consistency of DF is not guaranteed."
        )
        TMPDIRs.make_new_dir()
        self.pars["general"]["rng"] = self._rng
        _temp_files = []
        # save each component as a separate file -> means the sampled DF for a
        # static component is not altered by changing the profile for another
        for i, c in enumerate(self.components):
            _temp_file_name = os.path.join(TMPDIRs.register[-1], f"comp{i}.hdf5")
            _temp_files.append(_temp_file_name)
            comp = mg.ErgodicSphericalSystem(c, **self.pars["general"])
            mg.write_hdf5_ic_file(_temp_file_name, comp, center_CoM=self._center_CoM)
        # now read in components and join
        comps = []
        for fname in _temp_files:
            comps.append(mg.SnapshotSystem(fname))
        gal = mg.JoinedSystem(*comps)

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
        if hasattr(snap, "stars"):
            stellar_mass = np.sum(snap.stars["mass"])
            log_stellar_mass = np.log10(stellar_mass)
            self._calc_quants["stars"] = {}
        else:
            stellar_mass = 0
            log_stellar_mass = np.nan
        if hasattr(snap, "dm"):
            log_dm_mass = np.log10(np.sum(snap.dm["mass"]))
            self._calc_quants["dm"] = {}
        else:
            log_dm_mass = np.nan
        return snap, stellar_mass, log_stellar_mass, log_dm_mass

    def particle_counts(self, snap=None):
        if snap is None:
            snap, *_ = self._load_ic_file()
        for fam in ("stars", "dm"):
            if fam in snap.families():
                self._calc_quants[fam]["total_mass"] = np.sum(
                    getattr(snap, fam)["mass"]
                )
                self._calc_quants[fam]["particle_count"] = float(
                    len(getattr(snap, fam))
                )
        self.write_calculated_parameters()

    def plot_mass_scaling_relations(self):
        """
        Plot the stellar mass distribution, and the scaling relations of BH mass -- bulge mass and bulge mass -- DM mass.
        """
        snap, stellar_mass, log_stellar_mass, log_dm_mass = self._load_ic_file()
        self.particle_counts(snap)
        # read in literature data
        mass_data = LiteratureTables.load_sdss_mass_data()
        bh_data = LiteratureTables.load_sahu_2020_data()

        # set up figure
        fig, ax = plt.subplots(1, 3)
        make_wide_figure(fig)
        fig.suptitle(self.name)

        # plot bulge mass distribution
        for q, ls in zip((0.5, 0.16, 0.84), ("-", ":", ":")):
            ax[0].axhline(q, ls=ls, c="k")
        ecdf = EmpiricalCDF(mass_data.table["log_Mstar"])
        ecdf.plot(ax=ax[0], label="SDSS ETGs")
        ax[0].plot(log_stellar_mass, ecdf.cdf(log_stellar_mass), **self.marker_kwargs)
        ax[0].legend(loc="upper left")

        # plot bh - bulge relation
        bh_data.scatter(
            "log_M*_sph",
            "log_Mbh",
            xerr="log_M*_sph_ERR",
            yerr="log_Mbh_ERR",
            ax=ax[1],
            mask=bh_data.table.loc[:, "Cored"],
            use_label=False,
            scatter_kwargs={"label": "Cored"},
        )
        _, eb = bh_data.scatter(
            "log_M*_sph",
            "log_Mbh",
            xerr="log_M*_sph_ERR",
            yerr="log_Mbh_ERR",
            ax=ax[1],
            mask=~bh_data.table.loc[:, "Cored"],
            scatter_kwargs={"marker": "s", "label": r"Sersic"},
            use_label=False,
        )
        log_Mstar_seq = np.linspace(8, 12, 500)
        ax[1].plot(log_Mstar_seq, Sahu19(log_Mstar_seq), c="k", alpha=0.4)
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
        ax[0].set_xlabel(r"$\mathrm{log}(M_\star / \mathrm{M}_\odot)$")
        ax[0].set_ylabel(r"ECDF")
        ax[0].set_title("Stellar mass CDF")
        ax[1].set_xlim(8.7, 12.5)
        ax[1].set_ylim(7.2, 11)
        ax[1].set_xlabel(r"$\mathrm{log}(M_\star / \mathrm{M}_\odot)$")
        ax[1].set_ylabel(r"$\mathrm{log}(M_\bullet / \mathrm{M}_\odot)$")
        ax[1].set_title("Stellar -BH mass relation")
        ax[2].set_xlabel(r"$\mathrm{log}(M_\mathrm{halo} / \mathrm{M}_\odot)$")
        ax[2].set_ylabel(r"$\mathrm{log}(M_\star / M_\mathrm{halo})$")
        ax[2].set_title("Halo - stellar mass relation")

        savefig(self.fig_loc("masses.png"))
        self.write_calculated_parameters()

    def plot_kinematics(self, num_rots=3, ax=None):
        """
        Plot kinematic properties of the ICs to check for consistency with
        observations.

        Parameters
        ----------
        num_rots : int, optional
            number of rotations performed for projected quantities, by default 3
        """
        self._calc_quants["kinematics"] = {}

        # load IC file as snapshot
        snap, stellar_mass, log_stellar_mass, log_dm_mass = self._load_ic_file()
        self.particle_counts(snap)

        # set up figure
        if ax is None:
            need_obs_comp = True
            fig, ax = plt.subplot_mosaic(
                """
            ACEG
            BDFH
            """
            )
            make_wide_figure(fig)
            fig.suptitle(self.name)
            # load literature data
            bulgeBHData = LiteratureTables.load_sahu_2020_data()
            fDMData = LiteratureTables.load_jin_2020_data()
            BHsigmaData = LiteratureTables.load_vdBosch_2016_data()
            bulgesigmaData = LiteratureTables.load_kauffman_2003_data()
            bulgesigmaData2 = LiteratureTables.load_veale_2018_data()
        else:
            need_obs_comp = False

        # density profiles
        radial_bin_edges = dict(
            stars=np.geomspace(1e-2, 1e2, 51),
            dm=np.geomspace(0.1, self.pars["general"]["rmax"], 51),
        )
        for fam in ("stars", "dm"):
            if fam in snap.families():
                _logger.debug(f"Density profile for {fam}")
                subsnap = getattr(snap, fam)
                ax["A"].loglog(
                    get_histogram_bin_centres(radial_bin_edges[fam]),
                    pygad.analysis.profile_dens(
                        subsnap, "mass", r_edges=radial_bin_edges[fam]
                    ),
                    label=fam,
                )
        ax["A"].set_xlabel(r"$r/\mathrm{kpc}$")
        ax["A"].set_ylabel(r"$\rho / (\mathrm{M}_\odot\,\mathrm{kpc}^{-3})$")
        ax["A"].set_title("3D density", fontsize="small")
        ax["A"].legend()

        # projected quantities
        eff_rad, vsig2_Re, _, surf_dens = projected_quantities(
            snap, obs=num_rots, r_edges=radial_bin_edges["stars"], rng=self._rng
        )
        eff_rads = list(eff_rad.values())[0]
        eff_rad = np.nanmedian(eff_rads)
        vsig2_Re = quantiles_relative_to_median(list(vsig2_Re.values())[0])
        self._calc_quants["kinematics"]["projected_half_mass_radius"] = {
            "unit": "kpc",
            "value": float(eff_rad),
        }
        # use an unbiased estimator of standard deviation
        LOS_sigma = [np.sqrt(x) for x in vsig2_Re]
        self._calc_quants["kinematics"]["LOS_velocity_dispersion"] = {
            "unit": "km/s",
            "value": float(LOS_sigma[0]),
        }

        # velocity dispersion against central stellar density
        if need_obs_comp:
            bulgesigmaData.plot_lin_regress(
                "stellardens",
                "sigma",
                fit_in_log=True,
                plot_scatter=True,
                scatter_kwargs=self.errorbar_kwargs_data,
                ax=ax["B"],
            )
        central_stellar_densities = np.full_like(eff_rads, np.nan)
        rcentres = get_histogram_bin_centres(radial_bin_edges["stars"])
        for i, (_re, _surf_rho) in enumerate(
            zip(eff_rads, list(surf_dens.values())[0])
        ):
            central_stellar_densities[i] = np.interp(_re, rcentres, _surf_rho)
        central_stellar_density = quantiles_relative_to_median(
            central_stellar_densities
        )
        ax["B"].errorbar(
            central_stellar_density[0],
            LOS_sigma[0],
            xerr=central_stellar_density[1],
            yerr=LOS_sigma[1],
            **self.marker_kwargs,
        )
        ax["B"].set_xscale("log")
        ax["B"].set_yscale("log")
        ax["B"].set_xlabel(r"$\Sigma_\star/(\mathrm{M}_\odot\,\mathrm{kpc}^{-2})$")
        ax["B"].set_ylabel(r"$\sigma_\star$/ km/s")
        ax["B"].set_title(
            "Dispersion - surface mass density relation", fontsize="small"
        )
        ax["B"].legend()

        # plot of stellar mass against half mass radius
        if need_obs_comp:
            _, eb = bulgeBHData.plot_lin_regress(
                "Re_maj_kpc", "M*_sph", ax=ax["C"], plot_scatter=True, fit_in_log=True
            )
        hmr = pygad.analysis.half_mass_radius(snap.stars)
        self._calc_quants["kinematics"]["half_mass_radius"] = hmr
        ax["C"].plot(
            hmr,
            stellar_mass,
            zorder=eb.lines[0].get_zorder() + 0.2,
            **self.marker_kwargs,
        )
        ax["C"].legend()
        ax["C"].set_xscale("log")
        ax["C"].set_yscale("log")
        ax["C"].set_xlabel(r"R$_\mathrm{e}/$kpc")
        ax["C"].set_ylabel(r"M$_\star$/M$_\odot$")
        ax["C"].set_title(r"Size - mass relation", fontsize="small")

        # inner dark matter
        ax["D"].set_ylim(0, 1)
        if need_obs_comp:
            binned_fdm = scipy.stats.binned_statistic(
                fDMData.table.loc[:, "log_M*"],
                values=fDMData.table.loc[:, "f_DM"],
                bins=5,
                statistic="median",
            )
            _, eb = fDMData.scatter("M*", "f_DM", ax=ax["D"])
            fdm_masses = 10 ** get_histogram_bin_centres(binned_fdm[1])
            ax["D"].plot(fdm_masses, binned_fdm[0], "-x", label="Median")
        idmf = inner_DM_fraction(snap, eff_rad)
        self._calc_quants["kinematics"]["inner_DM_frac"] = idmf
        ax["D"].plot(
            stellar_mass,
            idmf,
            zorder=eb.lines[0].get_zorder() + 0.2,
            **self.marker_kwargs,
        )
        ax["D"].legend(loc="upper left")
        ax["D"].set_xscale("log")
        ax["D"].set_xlabel(r"M$_\star$/M$_\odot$")
        ax["D"].set_ylabel(r"f$_\mathrm{DM}(r<1\,$R$_\mathrm{e})$")
        ax["D"].set_title("Inner DM fraction", fontsize="small")

        # virial info
        vr, vm = pygad.analysis.virial_info(snap, N_min=10)
        self._calc_quants["kinematics"]["virial_info"] = {
            "mass": {"unit": "Msol", "value": float(vm)},
            "radius": {"unit": "kpc", "value": float(vr)},
        }

        # velocity anisotropy
        ax["E"].axhline(0, c="k")
        for fam in ("stars", "dm"):
            if fam in snap.families():
                beta = velocity_anisotropy(getattr(snap, fam), radial_bin_edges[fam])[0]
                ax["E"].semilogx(
                    get_histogram_bin_centres(radial_bin_edges[fam]), beta, label=fam
                )
        ax["E"].legend()
        ax["E"].set_title("Velocity anisotropy", fontsize="small")
        ax["E"].set_ylim(max(ax["E"].get_ylim()[0], -2), 1)
        ax["E"].set_xlabel(r"$r/\mathrm{kpc}$")
        ax["E"].set_ylabel(r"$\beta(r)$")

        # MBH-sigma relation
        ax["F"].set_title("BH mass - stellar dispersion", fontsize="small")
        if need_obs_comp:
            _, eb = BHsigmaData.plot_lin_regress(
                "sigma", "BHMass", ax=ax["F"], plot_scatter=True, fit_in_log=True
            )
        if "bh" in snap.families():
            ax["F"].errorbar(
                LOS_sigma[0],
                snap.bh["mass"],
                xerr=LOS_sigma[1],
                zorder=eb.lines[0].get_zorder() + 0.2,
                **self.marker_kwargs,
            )
            ax["F"].legend()
            ax["F"].set_xscale("log")
            ax["F"].set_yscale("log")
            ax["F"].set_xlabel(r"$\sigma_\star$/ km/s")
            ax["F"].set_ylabel(r"M$_\bullet$/M$_\odot$")

            # BH spin distribution
            try:
                spin_model = self.pars["bh"]["spin"]
            except KeyError:
                spin_model = self.pars["bh"]["chi"]
            if isinstance(spin_model, str):
                # select spin model
                spin_mag = scipy.stats.beta(*bh_spin_models[spin_model].values())
            else:
                spin_mag = scipy.stats.uniform(0, 1)

            spin_seq = np.linspace(0, 1, 1000)
            bh_spin = (
                pygad.UnitArr(
                    snap.bh["Spins"].view(np.ndarray) * 1e10,
                    units=snap.bh["angmom"].units,
                )
                * pygad.physics.c
                / pygad.physics.G
                / (snap.bh["mass"] ** 2)
            )
            bh_spin.convert_to_base_units()
            bh_spin = np.linalg.norm(bh_spin)
            ax["G"].plot(spin_seq, spin_mag.pdf(spin_seq))
            _logger.info(f"SMBH spin magnitude: {bh_spin:.3f}")
            ax["G"].plot(bh_spin, spin_mag.pdf(bh_spin), **self.marker_kwargs)
            ax["G"].set_title("BH spin", fontsize="small")
            ax["G"].set_xlabel(r"$\chi$")
            ax["G"].set_ylabel("PDF")

        if need_obs_comp:
            _, eb = bulgesigmaData.plot_lin_regress(
                "mstar",
                "sigma",
                fit_in_log=True,
                plot_scatter=True,
                scatter_kwargs=self.errorbar_kwargs_data,
                ax=ax["H"],
            )
            bulgesigmaData2.scatter(
                "mstar",
                "sigavg",
                scatter_kwargs={"marker": "s", "markersize": 2, "mew": 0},
                ax=ax["H"],
            )
        ax["H"].errorbar(
            stellar_mass,
            LOS_sigma[0],
            yerr=LOS_sigma[1],
            zorder=eb.lines[0].get_zorder() + 0.2,
            **self.marker_kwargs,
        )
        ax["H"].set_xscale("log")
        ax["H"].set_yscale("log")
        ax["H"].legend()
        ax["H"].set_xlabel(r"M$_\star$/M$_\odot$")
        ax["H"].set_ylabel(r"$\sigma_\star$/ km/s")
        ax["H"].set_title("Stellar mass - dispersion relation", fontsize="small")

        # save figure
        savefig(self.fig_loc("kinematics.png"))
        self.write_calculated_parameters()

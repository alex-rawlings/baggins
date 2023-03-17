import os.path
from datetime import datetime
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import merger_ic_generator as mg
import pygad

from .galaxy_components import _GalaxyICBase, _StellarCore, _StellarCusp, _DMHaloDehnen, _DMHaloNFW, _SMBH
from ..analysis import projected_quantities
from ..env_config import _cmlogger, date_format
from ..literature import *
from ..mathematics import get_histogram_bin_centres
from ..plotting import mplColours, savefig
from ..utils import to_json, write_calculated_parameters

__all__ = ["GalaxyIC"]

_logger = _cmlogger.copy(__file__)

# some plotting parameters
markersz = 1.5
cols = mplColours()


class GalaxyIC(_GalaxyICBase):
    def __init__(self, parameter_file, stellar_mass=None):
        """
        Class that interfaces between parameter file for galaxy initial 
        conditions, the galaxy object generator, and pygad.

        Parameters
        ----------
        parameter_file : str, path-like
            path to parameter file describing the galaxy ICs
        stellar_mass : float, optional
            stellar mass in Msol, only used by the DM methods if a stellar 
            component is not specified in the parameter file, by default None

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        super().__init__(parameter_file=parameter_file)
        self._calc_quants = {"stars":{}, "dm":{}, "bh":{}}
        # add redshift to the parameter file
        self._calc_quants["redshift"] = self.redshift
        self.stars = None
        self.dm = None
        self.bh = None
        # set up stars
        star_pars = self.parameters["stars"]
        if star_pars["particle_mass"]["value"] is not None:
            if star_pars["use_cored"]:
                self.stars = _StellarCore(parameter_file=parameter_file)
                # save the kpc values
                self._calc_quants["input_effective_radius"] = {"value":self.stars.effective_radius, "unit":self.stars.stellar_distance_units}
                self._calc_quants["input_core_radius"] = {"value":self.stars.core_radius, "unit":self.stars.stellar_distance_units}
            else:
                self.stars = _StellarCusp(parameter_file=parameter_file)
        else:
            _logger.logger.warning(f"No stellar component generated!")
        # set up DM
        dm_pars = self.parameters["dm"]
        if dm_pars["particle_mass"]["value"] is not None:
            try:
                _star_mass = self.stars.total_mass
            except AttributeError:
                if stellar_mass is None:
                    msg = "If no stellar component supplied, a stellar mass must be given to initialise the DM halo"
                    _logger.logger.error(msg)
                    raise ValueError(msg)
                else:
                    _star_mass = stellar_mass
            if dm_pars["use_NFW"]:
                self.dm = _DMHaloNFW(stellar_mass=_star_mass,parameter_file=parameter_file)
            else:
                self.dm = _DMHaloDehnen(stellar_mass=_star_mass, parameter_file=parameter_file)
            self._calc_quants["dm"]["peak_mass"] = self.dm.peak_mass
        else:
            _logger.logger.warning(f"No DM component generated!")
        # set up SMBH
        bh_pars = self.parameters["bh"]
        if bh_pars["set_spin"] is not None:
            try:
                _star_mass = self.stars.total_mass
            except AttributeError:
                if stellar_mass is None:
                    msg = "If no stellar component supplied, a stellar mass must be given to initialise the SMBH mass"
                    _logger.logger.error(msg)
                    raise ValueError(msg)
                else:
                    _star_mass = stellar_mass
            self.bh = _SMBH(np.log10(_star_mass), parameter_file=parameter_file)
            # manually set BH mass if desired
            if bh_pars["mass"]["value"] is not None:
                _logger.logger.warning(f"Setting BH mass to user defined value!")
                self.bh.mass = bh_pars["mass"]["value"]
            self._calc_quants["bh"]["mass"] = self.bh.mass
            #save the new spin value
            self._calc_quants["bh"]["spin"] = self.bh.spin
        else:
            _logger.logger.warning(f"No SMBH component generated!")
        self.hdf5_file_name = os.path.join(self.save_location, f"{self.name}.hdf5")
    

    def convert_to_gadget_mass_units(self):
        """
        Convert all masses from default units of Msol to 1e10 Msol, the Gadget
        standard.
        """
        _logger.logger.warning("Converting mass units to Gadget default (1e10 Msol)")
        _logger.logger.debug("Masses of galaxy components")
        assert self.mass_units == "msol"
        try:
            self.stars.particle_mass /= 1e10
            _logger.logger.debug(f"Stellar particle mass: {self.stars.particle_mass}")
            self.stars.total_mass /= 1e10
            _logger.logger.debug(f"Total stellar mass: {self.stars.total_mass}")
        except AttributeError:
            pass
        try:
            self.dm.particle_mass /= 1e10
            _logger.logger.debug(f"DM particle mass: {self.dm.particle_mass}")
            self.dm.peak_mass /= 1e10
            _logger.logger.debug(f"DM peak mass: {self.dm.peak_mass}")
        except AttributeError:
            pass
        try:
            self.bh.mass /= 1e10
            _logger.logger.debug(f"SMBH particle mass: {self.bh.mass}")
        except AttributeError:
            pass
        self.mass_units = "gadget"
        _logger.logger.warning(f"Mass units are now in {self.mass_units} standard.")
    

    def write_calculated_parameters(self):
        """
        Write calculated parameters to the parameter file
        """
        now = datetime.now()
        self._calc_quants["last_update"] = now.strftime(date_format)
        write_calculated_parameters(self._calc_quants, self.parameter_file)
    
    
    def plot_mass_scaling_relations(self):
        """
        Plot the stellar mass distribution, and the scaling relations of BH mass -- bulge mass and bulge mass -- DM mass.
        """
        #read in literature data
        mass_data = LiteratureTables("sdss_mass")
        bh_data = LiteratureTables("sahu_2020")

        # set up figure
        fig, ax = plt.subplots(1, 3)
        ax[0].set_xlabel(r"log(M$_\mathrm{bulge}$ / M$_\odot)$")
        ax[0].set_ylabel(r"Density")
        ax[0].set_title("SDSS Bulge Mass Distribution")
        ax[1].set_xlim(8.7, 12.5)
        ax[1].set_ylim(7.2, 11)
        ax[1].set_xlabel(r"log(M$_\mathrm{bulge}$/M$_\odot$)")
        ax[1].set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
        ax[1].set_title("Bulge - BH Mass")
        ax[2].set_xlabel(r"log(M$_\mathrm{halo}$/M$_\odot$)")
        ax[2].set_ylabel(r"log(M$_\mathrm{stellar}$/M$_\mathrm{halo}$)")
        ax[2].set_title(r"M$_\mathrm{stellar} - $M$_\mathrm{DM}$")
        ax[2].text(13.5, -1.5, f"z: {self.redshift}")

        #plot bulge mass distribution
        mass_data.hist("logMstar", ax=ax[0], label="SDSS DR7")
        ax[0].axvline(x=self.stars.log_total_mass, color=cols[3], label="Simulation")
        mass_data.add_qauntile_to_plot(0.5, "logMstar", ax[0], lkwargs={"c":cols[2], "ls":"--"})
        for q in [0.16, 0.84]:
            mass_data.add_qauntile_to_plot(q, "logMstar", ax[0], {"c":cols[2], "ls":":"})
        ax[0].legend(loc="upper left")
        
        #plot bh - bulge relation
        bh_data.scatter("logM*_sph", "logMbh", xerr="logM*_sph_ERR", yerr="logMbh_ERR", ax=ax[1], mask=bh_data.table.loc[:,"Cored"], label="Cored")
        bh_data.scatter("logM*_sph", "logMbh", xerr="logM*_sph_ERR", yerr="logMbh_ERR", ax=ax[1], mask=~bh_data.table.loc[:,"Cored"], scatter_kwargs={"marker":"s"}, label=r"S$\acute\mathrm{e}$rsic")
        logmstar_seq = np.linspace(8, 12, 500)
        ax[0].plot(logmstar_seq, Sahu19(logmstar_seq), c="k", alpha=0.4)
        ax[0].scatter(self.stars.log_total_mass, self.bh.log_mass, zorder=10, ls="None", color=cols[3], label="Simulation")
        ax[1].legend(loc="upper left")

        #plot bulge mass - DM halo mass relation
        halo_mass_seq, moster_seq = Moster10(self.stars.total_mass, [1e10, 1e15], z=self.redshift, plotting=True)
        ax[2].plot(halo_mass_seq, moster_seq-halo_mass_seq, label="Moster+10", color=cols[0])
        halo_mass_seq, girelli_seq = Girelli20(self.stars.total_mass, [1e10, 1e15], z=self.redshift, plotting=True)
        ax[2].plot(halo_mass_seq, girelli_seq-halo_mass_seq, label="Girelli+20", color=cols[1])
        halo_mass_seq, behroozi_seq = Behroozi19(self.stars.total_mass, [1e10, 1e15], z=self.redshift, plotting=True)
        ax[2].plot(halo_mass_seq, behroozi_seq-halo_mass_seq, label="Behroozi+19", color=cols[2])
        ax[2].scatter(self.dm.log_peak_mass, self.stars.log_total_mass-self.dm.log_peak_mass, color=cols[3], zorder=10, label="Simulation")
        ax[2].legend(loc="lower right")

        savefig(os.path.join(self.figure_location, f"{self.name}_ic_masses.png"))
    

    def plot_dm_cut_function(self, cut_params):
        """
        Plot the cut function applied to the NFW profile.

        Parameters
        ----------
        cut_params : dict
            cut parameters to the cut function
        """
        x = np.linspace(0, 10, 1000)
        div99 = (cut_params["shift"] - np.log(1/(1-0.99) - 1)) / cut_params["slope"] #where the function reaches 0.99
        y = -1/(1+np.exp(-cut_params["slope"]* x + cut_params["shift"])) + 1
        fig, ax = plt.subplots(1,1)
        ax.plot(x, y)
        ax.scatter(div99, 0.99, c=cols[1], zorder=10, label=f"({div99:.2f}, 0.99)")
        ax.legend()
        ax.set_xlabel(r"r/R$_\mathrm{vir}$")
        ax.set_ylabel("Cut Function")
        savefig(os.path.join(self.figure_location, f"{self.name}_nfwcut.png"))


    def generate_galaxy(self, allow_overwrite=False):
        """
        Generate the initial conditions as a hdf5 file that can be used by 
        Gadget.

        Parameters
        ----------
        allow_overwrite : bool, optional
            allow exisiting models to be overwritten, by default False
        """
        try:
            if os.path.exists(self.hdf5_file_name):
                assert allow_overwrite
        except AssertionError:
            _logger.logger.exception(f"File {self.hdf5_file_name} already exists!", exc_info=True)
            raise

        self.convert_to_gadget_mass_units()
        dists = []
        if self.stars is not None:
            # set up stars
            if isinstance(self.stars, _StellarCore):
                # dictionary for density function
                df_kwargs = dict(
                                rhob = self.stars.core_density,
                                rb = self.stars.core_radius,
                                n = self.stars.sersic_index,
                                g = self.stars.core_slope,
                                b = self.stars.sersic_b_parameter,
                                a = self.stars.transition_index,
                                Re = self.stars.effective_radius
                            )
                rho = lambda r: 0.1 * self.stars.mass_light_ratio * Terzic05(r, **df_kwargs)
                star_distribution = mg.GenericSphericalComponent(density_function=rho, particle_mass=self.stars.particle_mass, particle_type=mg.ParticleType.STARS)
                self._calc_quants["stars"]["actual_total_mass"] = star_distribution.mass * 1e10
            else:
                star_distribution = mg.DehnenSphere(self.stars.total_mass, scale_radius=self.stars.scale_radius, gamma=self.stars.gamma, particle_mass=self.stars.particle_mass, particle_type=mg.ParticleType.STARS)
            dists.append(star_distribution)

        if self.dm is not None:
            # set up dm
            if isinstance(self.dm, _DMHaloNFW):
                if self.parameters["dm"]["NFW"]["cut_pars"] is not None:
                    cut_params = self.self.parameters["dm"]["NFW"]["cut_pars"]
                    _logger.logger.info("Using user-defined NFW cut parameters")
                else:
                    cut_params = dict(
                                        slope = 1.0,
                                        approx0 = 1e-5,
                                        max_scale_radius = 20.0
                    )
                    _logger.logger.warning("Using default NFW cut parameters")
                dm_distribution = mg.NFWSphere(Mvir=self.dm.peak_mass, particle_mass=self.dm.particle_mass, particle_type=mg.ParticleType.DM_HALO, z=self.redshift, use_cut=True, cut_params=cut_params)
                self._calc_quants["dm"]["actial_total_mass"] = dm_distribution.mass * 1e10
                self._calc_quants["dm"]["concentration"] = self.dm.concentration
                self.plot_dm_cut_function(cut_params=cut_params)
            else:
                dm_distribution = mg.DehnenSphere(mass=self.dm.peak_mass, scale_radius=self.dm.scale_radius, gamma=self.dm.gamma, particle_mass=self.dm.particle_mass, particle_type=mg.ParticleType.DM_HALO)
            dists.append(dm_distribution)

        if self.bh is not None:
            # set up bh
            bh_particle = mg.CentralPointMass(mass=self.bh.mass, softening=self.bh.softening, chi=self.bh.spin, particle_type=mg.ParticleType.BH)
            dists.append(bh_particle)
        
        # generate the galaxy
        generated_galaxy = mg.SphericalSystem(*dists, rmax=self.maximum_radius, anisotropy_radius=self.stars.anisotropy_radius, rng=self._rng)

        # clean centre
        if self.bh is not None:
            generated_galaxy = mg.TransformedSystem(generated_galaxy, mg.FilterParticlesBoundToCentralMass(central_object_mass=self.bh.mass, minimum_semi_major_axis=self.minimum_radius))
        
        # ensure no particles dropped
        for k in generated_galaxy.particle_counts.keys():
            particle_count = generated_galaxy.particle_counts.get(k)
            t = str(k).split(".")[1].lower().replace("_halo", "")
            self._calc_quants[t]["particle_count"] = float(particle_count)
            if k != mg.ParticleType.BH and particle_count < 1e3:
                _logger.logger.warning(f"{k} has: {particle_count} particles!")
        
        # save galaxy
        mg.write_hdf5_ic_file(self.hdf5_file_name, generated_galaxy)
        self.write_calculated_parameters()


    def plot_ic_kinematics(self, num_rots=3):
        """
        Plot kinematic properties of the ICs to check for consistency with 
        observations.

        Parameters
        ----------
        update_file : bool, optional
            Allow updates to parameters in parameter file, by default False
        num_rots : int, optional
            number of rotations performed for projected quantities, by default 3
        """
        self._calc_quants["kinematics"] = {}
        # load literature data
        bulgeBHData = LiteratureTables("sahu_2020")
        fDMData = LiteratureTables("jin_2020")
        BHsigmaData = LiteratureTables("vdBosch_2016")

        radial_bin_edges = dict(
            stars = np.logspace(-2, 2, 50),
            dm = np.geomspace(10, self.maximum_radius, 50),
            stars_dm = np.geomspace(1e-2, self.maximum_radius, 50)
        )

        radial_bin_centres = dict()
        for k,v in radial_bin_edges.items():
            radial_bin_centres[k] = get_histogram_bin_centres(v)

        # load IC file as snapshot
        ic = pygad.Snapshot(self.hdf5_file_name, physical=True)
        mass_centre = pygad.analysis.shrinking_sphere(ic.stars, pygad.analysis.center_of_mass(ic.stars), 25.0)
        total_stellar_mass = np.sum(ic.stars["mass"])
        total_dm_mass = np.sum(ic.dm["mass"])

        #determine radial surface density profiles
        radial_surf_dens = dict(
            stars = pygad.analysis.profile_dens(ic.stars, qty="mass", r_edges=radial_bin_edges["stars"], center=mass_centre),
            dm = pygad.analysis.profile_dens(ic.dm, qty="mass", r_edges=radial_bin_edges["dm"], center=mass_centre),
            stars_dm = pygad.analysis.profile_dens(ic, qty="mass", r_edges=radial_bin_edges["stars_dm"], center=mass_centre)
        )

        # projected quantities
        eff_rad, vsig2_Re, *_ = projected_quantities(ic, obs=num_rots)
        eff_rad = np.nanmedian(list(eff_rad.values())[0])
        vsig2_Re = np.nanmedian(list(vsig2_Re.values())[0])
        self._calc_quants["kinematics"]["projected_half_mass_radius"] = {"unit":"kpc", "value":float(eff_rad)}
        # use an unbiased estimator of standard deviation
        LOS_sigma = np.sqrt(vsig2_Re)
        self._calc_quants["kinematics"]["LOS_velocity_dispersion"] = {"unit":"km/s", "value":float(LOS_sigma)}

        # estimate number of particles in Ketju region
        max_softening = max([self.stars.softening, self.bh.softening])
        ketju_radius = 3 * max_softening
        print(f"Assumed Ketju radius: {ketju_radius:.2f} kpc")
        ketju_mask = pygad.BallMask(ketju_radius, center=mass_centre)
        number_ketju_particles = len(ic.stars[ketju_mask]) + 1 #smbh
        self._calc_quants["kinematics"]["ketju_particles"] = number_ketju_particles

        # generate figure layout
        fig, ax = plt.subplots(3,3)

        # plot of the radial density profiles
        ax[0,0].set_xscale("log")
        ax[0,0].set_yscale("log")
        ax[0,0].set_title("Stellar Density", fontsize="small")
        ax[0,0].set_xlabel("Distance [kpc]")
        ax[0,0].set_ylabel(r"Density [$M_\odot$/kpc$^3$]")
        ax[0,0].plot(radial_bin_centres["stars"], radial_surf_dens["stars"], color=cols[3], lw=5, alpha=0.6)
        if isinstance(self.stars, _StellarCusp):
            dehnen_params_fitted = fit_Dehnen_profile(radial_bin_centres["stars"], radial_surf_dens["stars"], total_stellar_mass, bounds=([1, 0.5], [20,2]))
            ax[0,0].plot(radial_bin_centres["stars"], Dehnen(radial_bin_centres["stars"], *dehnen_params_fitted, total_stellar_mass), color=cols[1], label=r"a:{:.1f}, $\gamma$:{:.1f}".format(*dehnen_params_fitted))
            ax[0,0].legend()

        ax[0,1].set_xscale("log")
        ax[0,1].set_yscale("log")
        ax[0,1].set_title("DM Density", fontsize="small")
        ax[0,1].set_xlabel("Distance [kpc]")
        ax[0,1].set_ylabel(r"Density [$M_\odot$/kpc$^3$]")
        ax[0,1].plot(radial_bin_centres["dm"], radial_surf_dens["dm"], color=cols[3], lw=5, alpha=0.6)
        if isinstance(self.dm, _DMHaloDehnen):
            dehnen_params_fitted = fit_Dehnen_profile(radial_bin_centres["dm"], radial_surf_dens["dm"], total_dm_mass)
            ax[0,1].plot(radial_bin_centres["dm"], Dehnen(radial_bin_centres["dm"], *dehnen_params_fitted, total_dm_mass), color=cols[1], label=r"a:{:.1f}, $\gamma$:{:.1f}".format(*dehnen_params_fitted))
            ax[0,1].legend()

        ax[0,2].set_xscale("log")
        ax[0,2].set_yscale("log")
        ax[0,2].set_title("Total Density", fontsize="small")
        ax[0,2].set_xlabel("Distance [kpc]")
        ax[0,2].set_ylabel(r"Density [$M_\odot$/kpc$^3$]")
        ax[0,2].plot(radial_bin_centres["stars_dm"], radial_surf_dens["stars_dm"], color=cols[3], lw=5, alpha=0.6)
        if isinstance(self.stars, _StellarCusp) and isinstance(self.dm, _DMHaloDehnen):
            dehnen_params_fitted = fit_Dehnen_profile(radial_bin_centres["stars_dm"], radial_surf_dens["stars_dm"], total_stellar_mass + total_dm_mass + ic.bh["mass"])
            ax[0,2].plot(radial_bin_centres["stars_dm"], Dehnen(radial_bin_centres["stars_dm"], *dehnen_params_fitted, total_stellar_mass+total_dm_mass+ic.bh["mass"]), color=cols[3], label=r"a:{:.1f}, $\gamma$:{:.1f}".format(*dehnen_params_fitted))
            ax[0,2].legend()
        ax[0,2].plot(radial_bin_centres["stars"], radial_surf_dens["stars"], color="k", lw=0.8, alpha=0.6, ls=":")
        ax[0,2].plot(radial_bin_centres["dm"], radial_surf_dens["dm"], color="k", lw=0.8, alpha=0.6, ls="--")

        # plot of stellar mass against half mass radius
        ax[1,0].set_xlabel(r"log(R$_\mathrm{e, sph}/$kpc)")
        ax[1,0].set_ylabel(r"log(M$_\mathrm{*,sph}$ / M$_\odot$)")
        ax[1,0].set_title(r"R$_\mathrm{e}$ - log(M$_*$) Relation", fontsize="small")
        if isinstance(self.stars, _StellarCusp):
            hmr = halfMassDehnen(self.stars.scale_radius, self.stars.gamma)[0]
            self._calc_quants["kinematics"]["half_mass_radius_analytic"] = {"unit":"kpc", "value":hmr}
            ax[1,0].scatter(np.log10(hmr), np.log10(np.unique(ic.stars["mass"]) * len(ic.stars["mass"])), color=cols[3], marker="x", s=60, zorder=10, label="Theory")
        logRe_vals = np.log10(bulgeBHData.table["Re_maj"].astype("float") * bulgeBHData.table["scale"].astype("float"))
        ax[1,0].errorbar(logRe_vals, bulgeBHData.table.loc[:, "logM*_sph"], yerr=bulgeBHData.table.loc[:, "logM*_sph_ERR"], marker=".", ls="None", elinewidth=0.5, capsize=0, color=cols[1], ms=markersz, zorder=1, label="Sahu+20")
        logRe_seq = np.linspace(np.min(logRe_vals)*0.99, 1.01*np.max(logRe_vals))
        hmrT = pygad.analysis.half_mass_radius(ic.stars, center=mass_centre)
        self._calc_quants["kinematics"]["half_mass_radius_true"] = {"unit":"kpc", "value":hmrT}
        ax[1,0].plot(logRe_seq, Sahu20(logRe_seq), c=cols[1])
        ax[1,0].scatter(np.log10(hmrT), np.log10(np.unique(ic.stars["mass"]) * len(ic.stars["mass"])), color=cols[3], zorder=10, label="Actual")
        ax[1,0].legend()

        # inner dark matter
        ax[1,1].set_xlim(9.8, 12.1)
        ax[1,1].set_ylim(0, 1)
        ax[1,1].set_xlabel(r"log(M$_*$/M$_\odot$)")
        ax[1,1].set_ylabel(r"f$_\mathrm{DM}(r<1\,$R$_\mathrm{e})$")
        ax[1,1].set_title("Inner DM Fraction", fontsize="small")

        binned_fdm = scipy.stats.binned_statistic(fDMData.table.loc[:, "log(M*/Msun)"], values=fDMData.table.loc[:,"f_DM"], bins=5, statistic="median")
        ax[1,1].scatter(fDMData.table.loc[:, "log(M*/Msun)"], fDMData.table.loc[:,"f_DM"], c=cols[1], alpha=0.6, s=3, label="Jin+20")
        fdm_radii = get_histogram_bin_centres(binned_fdm[1])
        ax[1,1].plot(fdm_radii, binned_fdm[0], "-x", c=cols[2], label="Median")
        ball_mask = pygad.BallMask(eff_rad, center=mass_centre)
        inner_dm_mass = np.sum(ic.dm[ball_mask]["mass"])
        idmf = inner_dm_mass / (inner_dm_mass + np.sum(ic.stars[ball_mask]["mass"]))
        self._calc_quants["kinematics"]["inner_DM_frac"] = idmf
        ax[1,1].scatter(self.stars.log_total_mass, idmf, c=cols[3], zorder=10)
        ax[1,1].legend(loc="upper left")

        # virial info
        vr, vm = pygad.analysis.virial_info(ic, center=mass_centre, N_min=10)
        self._calc_quants["kinematics"]["virial_info"] = {"mass":{"unit":"Msol", "value":float(vm)}, "radius":{"unit":"kpc", "value":float(vr)}}
        ax[1,2].set_xlabel("log(r/kpc)")
        ax[1,2].set_ylabel("Count")
        ax[1,2].set_title("Star Count", fontsize="small")
        ax[1,2].set_yscale("log")
        star_rad_dist = np.sort(np.log10(ic.stars["r"]))
        ax[1,2].hist(star_rad_dist, 100)
        ax[1,2].axvline(star_rad_dist[100], c=cols[1], label=r"$10^2$")
        ax[1,2].axvline(star_rad_dist[1000], c=cols[1], label=r"$10^3$")
        ax[1,2].legend()
        i100star = 10**star_rad_dist[100]
        i1000star = 10**star_rad_dist[1000]
        self._calc_quants["stars"]["radius_to"] = {"unit":"kpc", "inner_100":i100star, "inner_1000":i1000star}
        # add the virial radius to the density plots
        for axi in (ax[0,1], ax[0,2]):
            axi.axvline(vr, c=cols[1], zorder=0, lw=0.7, label=r"R$_\mathrm{vir}$")
            axi.axvline(5*vr, c=cols[2], zorder=0, lw=0.7, label=r"5R$_\mathrm{vir}$")
        ax[0,1].legend()

        # histogram of LOS velocities
        ax[2,0].ticklabel_format(axis="y", style="scientific", scilimits=(0,0), useMathText=True)
        ax[2,0].hist(ic.stars["vel"][:,2].ravel(), 50, density=True)
        ax[2,0].set_xlabel(r"V$_*$ [km/s]")
        ax[2,0].set_ylabel("Density")
        ax[2,0].set_title("Stellar Velocity", fontsize="small", loc="right")

        # MBH-sigma relation
        ax[2,1].set_xlabel(r"log($\sigma_*$/ km/s)")
        ax[2,1].set_ylabel(r"log(M$_\bullet$/M$_\odot$)")
        ax[2,1].set_title("BH Mass - Stellar Dispersion", fontsize="small")

        ax[2,1].scatter(np.log10(LOS_sigma), np.log10(ic.bh["mass"]), zorder=10, color=cols[3])
        ax[2,1].errorbar(BHsigmaData.table.loc[:,"logsigma"], BHsigmaData.table.loc[:,"logBHMass"], xerr=BHsigmaData.table.loc[:,"e_logsigma"], yerr=[BHsigmaData.table.loc[:,"e_logBHMass"], BHsigmaData.table.loc[:,"E_logBHMass"]], marker=".", ls="None", elinewidth=0.5, capsize=0, color=cols[1], ms=markersz, zorder=1, label="Bosch+16")
        ax[2,1].legend()

        # BH spin distribution
        has_spin_distribution = False
        if self.parameters["bh"]["spin_relation"] is not None:
            if self.bh.spin_relation == "zlochower_dry":
                bh_spin_params = zlochower_dry_spins
                has_spin_distribution = True
            elif self.bh.spin_relation == "zlochower_cold":
                bh_spin_params = zlochower_cold_spins
                has_spin_distribution = True
            elif self.bh.spin_relation == "zlochower_hot":
                bh_spin_params = zlochower_hot_spins
                has_spin_distribution = True
            else:
                _logger.logger.error("Invalid BH spin parameters provided, no distribution will be plotted.")
        else:
            _logger.logger.warning("No BH spin distribution provided.")
        
        spin_seq = np.linspace(0, 1, 1000)
        bhspin_mag = np.linalg.norm(self.bh.spin)
        if has_spin_distribution:
            bh_chi_dist = scipy.stats.beta(*bh_spin_params.values())
            ax[2,2].plot(spin_seq, bh_chi_dist.pdf(spin_seq), color=cols[0])
        else:
            bh_chi_dist = scipy.stats.uniform()
        _logger.logger.info(f"SMBH spin magnitude: {bhspin_mag:.3f}")
        ax[2,2].scatter(bhspin_mag, bh_chi_dist.pdf(bhspin_mag), color=cols[3], zorder=10)
        ax[2,2].set_title(r"BH $\chi$", fontsize="small")
        ax[2,2].set_xlabel(r"$\chi$")
        ax[2,2].set_ylabel("PDF")

        # save figure
        savefig(os.path.join(self.figure_location, f"{self.name}_kinematics_ic.png"))
        self.write_calculated_parameters()


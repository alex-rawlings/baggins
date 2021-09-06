import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import h5py
import scipy.stats
import pandas as pd

import cm_functions as cmf
import pygad


#get the command line arguments
parser = cmf.utils.argparse_for_initialise(description='Analyse the kinematics of the initial conditions.', update_help='allow updates of kinematic quantities')
parser.add_argument('--plotcontours', dest='plot_contours', help='plot contours of scale radius and gamma for Dehnen models to show degeneracy', action='store_true')
parser.add_argument('--numrotations', dest='num_rots', help='number of rotations to use for averaging projected quantities', default=10)
args = parser.parse_args()

print('\nRunning ic_kinematics.py\n')

pfv = cmf.utils.read_parameters(args.paramFile)

galaxy = cmf.initialise.galaxy_ic_base(pfv, stars=True, dm=True, bh=True)
galaxy.dm.total_mass = (args.verbose, None)
galaxy.bh.mass = (args.verbose, None)

#this is the "snapshot"
icfile = galaxy.general.save_location + '/' + galaxy.general.name +'.hdf5'

markersz = 1.5
linewd = 1
capsize=2
sim_col = 'tab:blue'
fit_col = ['tab:orange', 'tab:red', 'tab:green']
erralpha=0.9
legend_font_size = 'x-small'

#load literature data
bulgeBHData = pd.read_table(galaxy.general.lit_location + '/' + pfv.bulgeBHData, sep=',', header=0)
#restrict to only ETGs (exclude also S0)
bulgeBHData = bulgeBHData.loc[np.logical_or(bulgeBHData.loc[:,'Type']=='E', bulgeBHData.loc[:,'Type']=='ES'), :]
cmf.utils.create_error_col(bulgeBHData, 'logM*_sph')

fDMData = pd.read_fwf(galaxy.general.lit_location + '/' + pfv.fDMData, comment='#', names=['MaNGAID', 'log(M*/Msun)', 'Re(kpc)', 'f_DM', 'p_e', 'q_e', 'T_e', 'f_cold', 'f_warm', 'f_hot', 'f_CR', 'f_prolong', 'f_CRlong', 'f_box', 'f_SR'])

BHsigmaData = pd.read_table(galaxy.general.lit_location + '/' + pfv.BHsigmaData, sep=';', header=0, skiprows=[1])


radial_bin_edges = dict(
    stars = np.logspace(-2, 2, 50),
    dm = np.logspace(1, np.log10(galaxy.general.maximum_radius), 50),
    stars_dm = np.logspace(-2, np.log10(galaxy.general.maximum_radius), 50)
)

radial_bin_centres = dict()


#generate figure layout
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(7,7))

#load ic file as a pygad snapshot, convert to physical units
ic = pygad.Snapshot(icfile)
ic.to_physical_units()
mass_centre = pygad.analysis.shrinking_sphere(ic.stars, pygad.analysis.center_of_mass(ic.stars), 25.0)
total_stellar_mass = np.sum(ic.stars['mass'])
total_dm_mass = np.sum(ic.dm['mass'])

#estimate number of particles in Ketju region
max_softening = max([galaxy.stars.softening, galaxy.bh.softening])
ketju_radius = 3 * max_softening
print('Assumed Ketju radius: {} kpc'.format(ketju_radius))
number_ketju_particles = np.sum(ic.stars['r'] <= ketju_radius) + 1 #smbh
pfv.number_ketju_particles = number_ketju_particles

#determine radial surface density profiles
radial_surf_dens = dict(
    stars = pygad.analysis.profile_dens(ic.stars, qty='mass', r_edges=radial_bin_edges['stars']),
    dm = pygad.analysis.profile_dens(ic.dm, qty='mass', r_edges=radial_bin_edges['dm']),
    stars_dm = pygad.analysis.profile_dens(ic, qty='mass', r_edges=radial_bin_edges['stars_dm'])
)

#plot of the radial density profiles
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Stellar Density', fontsize='small')
ax1.set_xlabel('Distance [kpc]')
ax1.set_ylabel(r'Density [$M_\odot$/kpc$^3$]')
radial_bin_centres['stars'] = cmf.mathematics.get_histogram_bin_centres(radial_bin_edges['stars'])
ax1.plot(radial_bin_centres['stars'], radial_surf_dens['stars'], color=sim_col, lw=5*linewd, alpha=0.6)
if isinstance(galaxy.stars, cmf.initialise.stellar_cuspy_ic):
    dehnen_params_fitted = cmf.literature.fit_Dehnen_profile(radial_bin_centres['stars'], radial_surf_dens['stars'], total_stellar_mass, bounds=([1, 0.5], [20,2]))
    ax1.plot(radial_bin_centres['stars'], cmf.literature.Dehnen(radial_bin_centres['stars'], *dehnen_params_fitted, total_stellar_mass), color=fit_col[0], label=r'a:{:.1f}, $\gamma$:{:.1f}'.format(*dehnen_params_fitted))
    ax1.legend(fontsize=legend_font_size)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('DM Density', fontsize='small')
ax2.set_xlabel('Distance [kpc]')
ax2.set_ylabel(r'Density [$M_\odot$/kpc$^3$]')
radial_bin_centres['dm'] = cmf.mathematics.get_histogram_bin_centres(radial_bin_edges['dm'])
ax2.plot(radial_bin_centres['dm'], radial_surf_dens['dm'], color=sim_col, lw=5*linewd, alpha=0.6)
if isinstance(galaxy.dm, cmf.initialise.dm_halo_dehnen):
    dehnen_params_fitted = cmf.literature.fit_Dehnen_profile(radial_bin_centres['dm'], radial_surf_dens['dm'], total_dm_mass)
    ax2.plot(radial_bin_centres['dm'], cmf.literature.Dehnen(radial_bin_centres['dm'], *dehnen_params_fitted, total_dm_mass), color=fit_col[0], label=r'a:{:.1f}, $\gamma$:{:.1f}'.format(*dehnen_params_fitted))
    ax2.legend(fontsize=legend_font_size)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_title('Total Density', fontsize='small')
ax3.set_xlabel('Distance [kpc]')
ax3.set_ylabel(r'Density [$M_\odot$/kpc$^3$]')
radial_bin_centres['stars_dm'] = cmf.mathematics.get_histogram_bin_centres(radial_bin_edges['stars_dm'])
ax3.plot(radial_bin_centres['stars_dm'], radial_surf_dens['stars_dm'], color=sim_col, lw=5*linewd, alpha=0.6)
if isinstance(galaxy.stars, cmf.initialise.stellar_cuspy_ic) and isinstance(galaxy.dm, cmf.initialise.dm_halo_dehnen):
    dehnen_params_fitted = cmf.initialise.fit_Dehnen_profile(radial_bin_centres['stars_dm'], radial_surf_dens['stars_dm'], total_stellar_mass + total_dm_mass + ic.bh['mass'])
    ax3.plot(radial_bin_centres['stars_dm'], cmf.initialise.Dehnen(radial_bin_centres['stars_dm'], *dehnen_params_fitted, total_stellar_mass+total_dm_mass+ic.bh['mass']), color=fit_col[0], label=r'a:{:.1f}, $\gamma$:{:.1f}'.format(*dehnen_params_fitted))
    ax3.legend(fontsize=legend_font_size)
ax3.plot(radial_bin_edges['stars'][:-1], radial_surf_dens['stars'], color='k', lw=0.8, alpha=0.6, ls=':')
ax3.plot(radial_bin_edges['dm'][:-1], radial_surf_dens['dm'], color='k', lw=0.8, alpha=0.6, ls='--')


if args.plot_contours:
    figc, (axc1, axc2, axc3) = plt.subplots(1,3, figsize=(7,3))
    for ax in (axc1, axc2, axc3):
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'a [kpc]')
    axc1.set_title('Stars')
    axc2.set_title('DM')
    axc3.set_title('Both')
    cmf.plotting.plot_parameter_contours(axc1, cmf.literature.Dehnen, radial_bin_centres['stars'], radial_surf_dens['stars'], ([1, 10], [0.5, 2.99]), args=[total_stellar_mass])
    cmf.plotting.plot_parameter_contours(axc2, cmf.literature.Dehnen, radial_bin_centres['dm'], radial_surf_dens['dm'], ([50, 400], [0.5, 2.99]), args=[total_dm_mass])
    cmf.plotting.plot_parameter_contours(axc3, cmf.literature.Dehnen, radial_bin_centres['stars_dm'], radial_surf_dens['stars_dm'], ([50, 400], [0.5, 2.99]), args=[total_stellar_mass+total_dm_mass+ic.bh['mass'][0]])
    figc.tight_layout()
    figc.savefig(galaxy.general.figure_location + '/' + galaxy.general.name + '_dehnen_contours.png', dpi=300)
    plt.close()


#plot of stellar mass against half mass radius
ax4.set_xlabel(r'log(R$_\mathrm{e, sph}/$kpc)')
ax4.set_ylabel(r'log(M$_\mathrm{*,sph}$ / M$_\odot$)')
ax4.set_title(r'R$_\mathrm{e}$ - log(M$_*$) Relation', fontsize='small')
if isinstance(galaxy.stars, cmf.initialise.stellar_cuspy_ic):
    pfv.half_mass_radius, *_ = cmf.literature.halfMassDehnen(galaxy.stars.scale_radius, galaxy.stars.gamma)
    ax4.scatter(np.log10(pfv.half_mass_radius), np.log10(np.unique(ic.stars['mass']) * len(ic.stars['mass'])), color=sim_col, marker='x', s=60, zorder=10, label='Theory')
logRe_vals = np.log10(bulgeBHData['Re_maj'].astype('float') * bulgeBHData['scale'].astype('float'))
ax4.errorbar(logRe_vals, bulgeBHData.loc[:, 'logM*_sph'], yerr=bulgeBHData.loc[:, 'logM*_sph_ERR'], marker='.', ls='None', elinewidth=0.5, capsize=0, color=fit_col[0], ms=markersz, zorder=1, label='Sahu+20')
logRe_seq = np.linspace(np.min(logRe_vals)*0.99, 1.01*np.max(logRe_vals))
ax4.plot(logRe_seq, cmf.literature.Sahu20(logRe_seq), lw=linewd, c=fit_col[0])
ax4.scatter(np.log10(pygad.analysis.half_mass_radius(ic.stars, center=mass_centre)), np.log10(np.unique(ic.stars['mass']) * len(ic.stars['mass'])), color=sim_col, zorder=10, label='Actual')
ax4.legend(fontsize=legend_font_size)


#inner dark matter
binned_fdm = scipy.stats.binned_statistic(fDMData.loc[:, 'log(M*/Msun)'], values=fDMData.loc[:,'f_DM'], bins=5, statistic='median')
ax5.scatter(fDMData.loc[:, 'log(M*/Msun)'], fDMData.loc[:,'f_DM'], c=fit_col[0], alpha=0.6, s=3, label='Jin+20')
fdm_radii = cmf.mathematics.get_histogram_bin_centres(binned_fdm[1])
ax5.plot(fdm_radii, binned_fdm[0], '-x', c=fit_col[2], label='Median')
#find the average projected half mass radius, as a proxy to the
#(by defintion) projected effective radius
r_half_mass_proj = 0
for ind in range(3):
    r_half_mass_proj += pygad.analysis.half_mass_radius(ic.stars, proj=ind, center=mass_centre)
r_half_mass_proj /= 3

#determine the LOS quantities
rot_axis = np.ones(3)
#initialise the projected half mass radius and LOS velocity dispersions
r_half_mass_proj = 0
LOS_vel_variance = np.full(3*args.num_rots, np.nan)
for ind, angle in enumerate(np.linspace(0, np.pi/2, args.num_rots)):
    rotation = pygad.transformation.rot_from_axis_angle(rot_axis, angle)
    rotation.apply(ic)
    for viewaxis in range(3):
        print('Determining LOS quantities: {}.{}         '.format(ind, viewaxis), end='\r')
        proj_mask = [i for i in range(3) if i != viewaxis]
        this_half_mass_radius = pygad.analysis.half_mass_radius(ic.stars, center=mass_centre, proj=viewaxis)
        r_half_mass_proj += this_half_mass_radius
        projected_separation = pygad.utils.dist(ic.stars['pos'][:, proj_mask]-mass_centre[proj_mask])
        LOS_vel_variance[3*ind+viewaxis] = pygad.analysis.los_velocity_dispersion(ic.stars[projected_separation < this_half_mass_radius], proj=viewaxis)**2
r_half_mass_proj /= (args.num_rots*3)

pfv.projected_half_mass_radius = r_half_mass_proj
dm_mass_in_1_re = pygad.analysis.radially_binned(ic.dm, 'mass', r_edges=[0, r_half_mass_proj, galaxy.general.maximum_radius], center=mass_centre)[0]
all_mass_in_1_re = pygad.analysis.radially_binned(ic, 'mass', r_edges=[0, r_half_mass_proj, galaxy.general.maximum_radius], center=mass_centre)[0]
pfv.inner_DM_fraction = dm_mass_in_1_re/all_mass_in_1_re
ax5.scatter(galaxy.stars.log_total_mass, pfv.inner_DM_fraction, color=sim_col, zorder=10)
ax5.set_xlim(9.8, 12.1)
ax5.set_ylim(0, 1)
ax5.set_xlabel(r'log(M$_*$/M$_\odot$)')
ax5.set_ylabel(r'f$_\mathrm{DM}(r<1\,$R$_\mathrm{e})$')
ax5.set_title('Inner DM Fraction', fontsize='small')
ax5.legend(loc='upper left', fontsize=legend_font_size)


#virial info
pfv.virial_radius, pfv.virial_mass = pygad.analysis.virial_info(ic, center=mass_centre, N_min=10)
ax6.set_xlabel('log(r/kpc)')
ax6.set_ylabel('Count')
ax6.set_title('Star Count', fontsize='small')
ax6.set_yscale('log')
star_rad_dist = np.sort(np.log10(ic.stars['r']))
ax6.hist(star_rad_dist, 100)
ax6.axvline(star_rad_dist[100], c=fit_col[0], label=r'$10^2$')
ax6.axvline(star_rad_dist[1000], c=fit_col[1], label=r'$10^3$')
ax6.legend(fontsize=legend_font_size)
pfv.inner_100_star_radius = 10**star_rad_dist[100]
pfv.inner_1000_star_radius = 10**star_rad_dist[1000]
#add the virial radius to the density plots
for axi in (ax2, ax3):
    axi.axvline(pfv.virial_radius, c=fit_col[1], zorder=0, lw=0.7, label=r'R$_\mathrm{vir}$')
    axi.axvline(5*pfv.virial_radius, c=fit_col[2], zorder=0, lw=0.7, label=r'5R$_\mathrm{vir}$')
ax2.legend(fontsize=legend_font_size)


#histogram of LOS velocities
ax7.ticklabel_format(axis='y', style='scientific', scilimits=(0,0), useMathText=True)
ax7.hist(ic.stars['vel'][:,2].ravel(), 50, density=True)
ax7.set_xlabel(r'V$_*$ [km/s]')
ax7.set_ylabel('Density')
ax7.set_title('Stellar Velocity', fontsize='small', loc='right')


#MBH-sigma relation
#clean data
BHsigmaData.loc[BHsigmaData.loc[:,'e_logBHMass']==BHsigmaData.loc[:,'logBHMass'], 'e_logBHMass'] = np.nan
BHsigmaData.loc[BHsigmaData.loc[:,'logBHMass']<1, 'logBHMass'] = np.nan


pfv.LOS_vel_dispersion = np.sqrt(np.mean(LOS_vel_variance) * args.num_rots * 3 / (args.num_rots*3-1))
print('LOS velocity dispersion calculated                       ')
ax8.scatter(np.log10(pfv.LOS_vel_dispersion), np.log10(ic.bh['mass']), zorder=10, color=sim_col)
ax8.errorbar(BHsigmaData.loc[:,'logsigma'], BHsigmaData.loc[:,'logBHMass'], xerr=BHsigmaData.loc[:,'e_logsigma'], yerr=[BHsigmaData.loc[:,'e_logBHMass'], BHsigmaData.loc[:,'E_logBHMass']], marker='.', ls='None', elinewidth=0.5, capsize=0, color=fit_col[0], ms=markersz, zorder=1, label='Bosch+16')
ax8.legend(fontsize=legend_font_size)
ax8.set_xlabel(r'log($\sigma_*$/ km/s)')
ax8.set_ylabel(r'log(M$_\bullet$/M$_\odot$)')
ax8.set_title('BH Mass - Stellar Dispersion', fontsize='small')


if pfv.BH_spin_from == 'zlochower_dry':
    bh_chi_dist = scipy.stats.beta(
                            cmf.literature.zlochower_dry_spins['spin_mag_a'], cmf.literature.zlochower_dry_spins['spin_mag_b']
                            )
else:
    raise NotImplementedError('Dry Spins only implemented currently')
spin_seq = np.linspace(0, 1, 1000)
bhspin_mag = np.linalg.norm(galaxy.bh.spin)
if args.verbose:
    print('SMBH spin magnitude: {:.3}'.format(bhspin_mag))
ax9.plot(spin_seq, bh_chi_dist.pdf(spin_seq), color=fit_col[0])
ax9.scatter(bhspin_mag, bh_chi_dist.pdf(bhspin_mag), color=sim_col, zorder=10)
ax9.set_title(r'BH $\chi$', fontsize='small')
ax9.set_xlabel(r'$\chi$')
ax9.set_ylabel('PDF')

fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.08, hspace=0.5, wspace=0.5)
plt.savefig(galaxy.general.figure_location + '/' + galaxy.general.name + '_kinematics.png', dpi=300)
plt.close()

if args.verbose:
    print('Update <user input> variables in {} if necessary, and re-run ./galaxy_gen.py, and then this script.'.format(args.paramFile))
    print('If changing parameter values, either delete the outputted variables from galaxy_gen.py and this script, or ensure both scripts are run with the -u flag to ensure parameter values are updated.')
    print('When satisfied, evolve system in isolation with KETJU for a sufficiently long (>250 Myr) time period.')
    print('When system evolved, analyse stability with ./stability.py')

if args.parameter_update:
    cmf.utils.write_parameters(pfv, allow_updates=('number_ketju_particles', 'half_mass_radius', 'projected_half_mass_radius', 'inner_DM_fraction', 'inner_100_star_radius', 'inner_1000_star_radius', 'LOS_vel_dispersion'), verbose=args.verbose)
else:
    cmf.utils.write_parameters(pfv, verbose=args.verbose)

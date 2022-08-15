import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import warnings

import merger_ic_generator as mg
import cm_functions as cmf


#get the command line arguments
parser = cmf.utils.argparse_for_initialise(description='Generate the galaxy initial conditions.', update_help='allow updates of SMBH spin and other derived quantities')
parser.add_argument("-v", "--verbose", help="verbose printing", dest="verbose", action="store_true")
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.paramFile)
parameters_to_update=[]

print('\nRunning galaxy_gen.py\n')

galaxy = cmf.initialise.galaxy_ic_base(pfv, stars=True, dm=True, bh=True)

galaxy.dm.peak_mass = (args.verbose, None)
galaxy.bh.mass = (args.verbose, None)
#seed the random generator
np.random.seed(galaxy.general.seed)

if pfv.BH_spin_from == 'zlochower_dry':
    bh_spin_params = cmf.literature.zlochower_dry_spins
elif pfv.BH_spin_from == 'zlochower_hot':
    bh_spin_params = cmf.literature.zlochower_hot_spins
elif pfv.BH_spin_from == 'zlochower_cold':
    bh_spin_params = cmf.literature.zlochower_cold_spins
else:
    raise NotImplementedError('Spins from Zlochower+Lousto only implemented currently')


#set SMBH spin
if isinstance(pfv.BH_spin, str):
    #assign random spin
    spin_mag = scipy.stats.beta.rvs(bh_spin_params['spin_mag_a'], bh_spin_params['spin_mag_b'], size=1)
    spin_dir_theta = 2*np.pi*scipy.stats.uniform.rvs(size=1)
    spin_dir_phi = np.arccos(2*scipy.stats.uniform.rvs(size=1) - 1)
    galaxy.bh.spin = np.array([
        spin_mag * np.sin(spin_dir_phi) * np.cos(spin_dir_theta),
        spin_mag * np.sin(spin_dir_phi) * np.sin(spin_dir_theta),
        spin_mag * np.cos(spin_dir_phi)
    ]).flatten()
    parameters_to_update.append('BH_spin')
else:
    galaxy.bh.spin = pfv.BH_spin


#need to convert units to gadget units!!
galaxy.to_gadget_mass_units()
if args.verbose:
    galaxy.print_masses()

#create galaxy
if isinstance(galaxy.stars, cmf.initialise.stellar_cored_ic):
    #dictionary for density function
    df_kwargs = dict(
        rhob = galaxy.stars.core_density,
        rb=galaxy.stars.core_radius,
        n=galaxy.stars.sersic_index,
        g=galaxy.stars.core_slope,
        b=galaxy.stars.sersic_b_parameter,
        a=galaxy.stars.transition_index,
        Re = galaxy.stars.effective_radius
    )
    star_distribution = mg.GenericSphericalComponent(density_function=lambda r: galaxy.stars.mass_light_ratio*1e-1*cmf.literature.Terzic05(r, **df_kwargs), particle_mass=galaxy.stars.particle_mass, particle_type=mg.ParticleType.STARS)
    pfv.stellar_actual_total_mass = star_distribution.mass*1e10
    parameters_to_update.append('input_Re_in_kpc')
    parameters_to_update.append('input_Rb_in_kpc')
    parameters_to_update.append('stellar_actual_total_mass')
else:
    star_distribution = mg.DehnenSphere(mass=galaxy.stars.total_mass, scale_radius=galaxy.stars.scale_radius, gamma=galaxy.stars.gamma, particle_mass=galaxy.stars.particle_mass, particle_type=mg.ParticleType.STARS)

if isinstance(galaxy.dm, cmf.initialise.dm_halo_NFW):
    if hasattr(pfv, 'DM_cut'):
        cut_params = pfv.DM_cut
        if args.verbose:
            print('Using user-defined NFW cut parameters...')
    else:
        cut_params = dict(
            slope = 1,
            approx0 = 1e-5,
            max_scaled_radius = 20
        )
        if args.verbose:
            print('Using default NFW cut parameters...')
    dm_distribution = mg.NFWSphere(Mvir=galaxy.dm.peak_mass, particle_mass=galaxy.dm.particle_mass, particle_type=mg.ParticleType.DM_HALO, z=galaxy.general.redshift, use_cut=True, cut_params=cut_params)
    pfv.DM_actual_total_mass = dm_distribution.mass*1e10
    pfv.DM_concentration = galaxy.dm.concentration
    parameters_to_update.append('DM_actual_total_mass')
    parameters_to_update.append('DM_concentration')

    #plot the cut function
    x = np.linspace(0, 10, 1000)
    div99 = (cut_params['shift'] - np.log(1/(1-0.99) - 1)) / cut_params['slope'] #where the function reaches 0.99
    y = -1/(1+np.exp(-cut_params['slope']* x + cut_params['shift'])) + 1
    fig, ax = plt.subplots(1,1)
    ax.plot(x, y)
    ax.scatter(div99, 0.99, c='tab:orange', zorder=10, label='({:.2f}, 0.99)'.format(div99))
    ax.legend()
    ax.set_xlabel(r'r/R$_\mathrm{vir}$')
    ax.set_ylabel('Cut Function')
    fig.savefig(galaxy.general.figure_location + '/' + galaxy.general.name + '_nfwcut.png', dpi=300)
    plt.close()

else:
    dm_distribution = mg.DehnenSphere(mass=galaxy.dm.peak_mass, scale_radius=galaxy.dm.scale_radius, gamma=galaxy.dm.gamma, particle_mass=galaxy.dm.particle_mass, particle_type=mg.ParticleType.DM_HALO)

generated_galaxy = mg.SphericalSystem(star_distribution,
                       dm_distribution,
                       mg.CentralPointMass(mass=galaxy.bh.mass, particle_type=mg.ParticleType.BH, softening=galaxy.bh.softening, chi=galaxy.bh.spin),
                       rmax=galaxy.general.maximum_radius,
                       anisotropy_radius=galaxy.general.anisotropy_radius)

#clean centre
generated_galaxy = mg.TransformedSystem(generated_galaxy, mg.FilterParticlesBoundToCentralMass(galaxy.bh.mass, galaxy.general.minimum_radius))

#ensure no particles dropped
for k in generated_galaxy.particle_counts.keys():
    var_name = 'count_' + str(k).split('.')[1]
    particle_count = generated_galaxy.particle_counts.get(k)
    if k != mg.ParticleType.BH and particle_count < 1e3:
        warnings.warn('{} has: {} particles!'.format(str(k), particle_count))
    setattr(pfv, var_name, particle_count)
    parameters_to_update.append(var_name)

#save galaxy
mg.write_hdf5_ic_file((galaxy.general.save_location+'/'+galaxy.general.name+'.hdf5'), generated_galaxy)
if args.verbose:
    print('Now run ./ic_kinematics.py to analyse system kinematics.')

if args.parameter_update:
    cmf.utils.write_parameters(pfv, allow_updates=tuple(parameters_to_update), verbose=args.verbose)
else:
    cmf.utils.write_parameters(pfv, verbose=args.verbose)

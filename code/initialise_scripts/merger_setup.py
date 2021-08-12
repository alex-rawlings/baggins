import argparse
import numpy as np
import matplotlib.pyplot as plt

import pygad
import merger_ic_generator as mg
import cm_functions as cmf


#get the command line arguments
parser = cmf.utils.argparse_for_initialise(description='Generate the merger from two generated initial conditions.', update_help='allow updates of used virial_radius, r0, rperi, and e')
parser.add_argument('-p', '--plot', dest='plot', help='plot the merger setup', action='store_true')
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.paramFile)

print('\nRunning merger_gen.py\n')

galaxy1 = mg.SnapshotSystem(pfv.file1)
galaxy2 = mg.SnapshotSystem(pfv.file2)

if 'virial' in pfv.initialSeparation:
    pfv.virial_radius = -10
    for i, g in enumerate((galaxy1, galaxy2), start=1):
        gal = pygad.Snapshot(getattr(pfv, 'file{}'.format(i)))
        gal.to_physical_units()
        centre_guess = pygad.analysis.center_of_mass(gal.stars)
        #refine the center estimate with shrinking sphere
        center_ss = pygad.analysis.shrinking_sphere(gal.stars, centre_guess, 25.0)
        this_virial_radius, virial_mass = pygad.analysis.virial_info(gal, center=center_ss)
        if this_virial_radius > pfv.virial_radius:
            pfv.virial_radius = this_virial_radius.view(np.ndarray)
    r0_frac = float(pfv.initialSeparation[6:])
    pfv.r0 = r0_frac * pfv.virial_radius
else:
    raise NotImplementedError

if 'virial' in pfv.pericentreDistance:
    rperi_frac = float(pfv.pericentreDistance[6:])
    pfv.rperi = rperi_frac * pfv.virial_radius
else:
    raise NotImplementedError

#get the eccentricity of the approach
pfv.e = cmf.initialise.e_from_rperi(pfv.rperi/pfv.virial_radius)

merger = mg.Merger(galaxy1, galaxy2, pfv.r0, pfv.rperi, e=pfv.e)

save_file_as = '{}/{}-{}-{}-{}.hdf5'.format(pfv.saveLocation, pfv.galaxyName1, pfv.galaxyName2, r0_frac, rperi_frac)

mg.write_hdf5_ic_file(save_file_as, merger, save_plots=False)

if args.parameter_update:
    cmf.utils.write_parameters(pfv, allow_updates=('virial_radius', 'r0', 'rperi', 'e'), verbose=args.verbose)
else:
    cmf.utils.write_parameters(pfv, verbose=args.verbose)


if args.plot:
    #plot the setup
    if args.verbose:
        print('Plotting...')
    snap = pygad.Snapshot(save_file_as)
    snap.to_physical_units()
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6, 3.5))
    ax1.set_title('Stars')
    ax2.set_title('DM Halo')
    _,ax1,*_ = pygad.plotting.image(snap.stars, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1)
    _,ax2,*_ = pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2)
    fig.tight_layout()
    plt.savefig('{}.png'.format(save_file_as.split('.')[0]), dpi=300)

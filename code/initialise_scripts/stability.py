import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cm_functions as cmf
import pygad


#get the command line arguments
parser = cmf.utils.argparse_for_initialise(description='Analyse the stability of the initial conditions.')
parser.add_argument('-q', '--quantilecut', dest='quantile_cut', help='remove data that is above this \nradial quantile', default=1)
parser.add_argument('-p', '--percentages', dest='percentages_to_check', help='find the radius corresponding to these percentages of mass (can be specified multiple times)', action='append', default=None, type=float)
args = parser.parse_args()

print('\nRunning stability.py\n')

pfv = cmf.utils.read_parameters(args.paramFile)
#population percentages
if args.percentages_to_check is None:
    args.percentages_to_check = [5, 10, 30, 50, 80, 95]
linewd = 2
if args.verbose:
    print('Stability checked at radii enclosing mass percentages: {}'.format(args.percentages_to_check))

outpath = os.path.join(pfv.saveLocation, pfv.galaxyName)
datapath = os.path.join(outpath, pfv.dataLocation)
figPath = os.path.join(outpath, pfv.figureLocation)

datfiles = cmf.utils.get_snapshots_in_dir(datapath)

#create data frame
stability_data = pd.DataFrame(data={'Time': np.full_like(datfiles, np.nan)})
for ind, col in enumerate(args.percentages_to_check):
    stability_data.insert(1+ind*2, 'stars_'+str(col), np.full_like(datfiles, np.nan))
    stability_data.insert(2+ind*2, 'dm_'+str(col), np.full_like(datfiles, np.nan))

for ind, this_file in enumerate(datfiles):
    print('Reading: ' + this_file)
    gal = pygad.Snapshot(this_file)
    gal.to_physical_units()
    stability_data.loc[ind, 'Time'] = cmf.general.convert_gadget_time(gal) * 1e3
    particle_families = []
    if 'stars' in gal:
        particle_families.append('stars')
        mass_centre = pygad.analysis.shrinking_sphere(gal.stars, pygad.analysis.center_of_mass(gal.stars), 25.0)
    if 'dm' in gal:
        particle_families.append('dm')
        if 'stars' not in gal:
            mass_centre = pygad.analysis.shrinking_sphere(gal.dm, pygad.analysis.center_of_mass(gal.dm), 500)
    for p_ind, particle_type in enumerate(particle_families):
        print('  Searching: '+particle_type)
        subsnap = getattr(gal, particle_type)
        subsnap['pos'] -= mass_centre
        if args.quantile_cut < 1:
            #apply the optional quantile cut
            subsnap = subsnap[subsnap['r'] < pygad.UnitScalar(np.quantile(subsnap['r'], args.quantile_cut), units='kpc')]
        #find the radius corresponding to the percentage masses
        max_dists = np.percentile(subsnap['r'], args.percentages_to_check)
        for f_ind, percentage in enumerate(args.percentages_to_check):
            #remove the units from the pygad unit array
            #matplotlib seems to sometimes not like them
            stability_data.loc[ind, particle_type+'_'+str(percentage)] = float(np.max(subsnap[subsnap['r']<max_dists[f_ind]]['r']))

colours = cmf.plotting.mplColours()
point_style = {'stars': '*', 'dm':'s'}
fig, ax = plt.subplots(1,1, figsize=(7,3))
ax.set_yscale('log')
ax.set_xlabel('Time [Myr]')
ax.set_ylabel('Radius [kpc]')
#save some stats
with open((outpath+'/'+pfv.galaxyName+'_stability.txt'), 'w') as f:
    f.write('Percent_Radial_Difference = (R_f - R_i)/R_i*100\n')
    f.write('Particle_Type,Percentage,Percent_Radial_Difference\n')
    label_flag = 0
    for ind, col in enumerate(stability_data.columns[1:], start=1):
        particle, label_value = str(col).split('_')
        label_value += '%'
        label_flag = ind%len(particle_families)
        '''if not label_flag:
            label_flag = ind%len(particle_families)
        else:
            label_flag = ind%len(particle_families)'''
        if stability_data.iloc[0,ind] == 'nan':
            continue

        #particle = str(col).split('_')[0]
        colour_value = colours[int(np.floor((ind-1)/2))]
        #do separate plot and scatter so legend looks better
        ax.plot(stability_data.loc[:, 'Time'], stability_data.loc[:, col], c=colour_value, label=(label_value if not label_flag else ''))
        ax.scatter(stability_data.iloc[-1, 0], stability_data.iloc[-1, ind], c=colour_value, marker=point_style[particle], zorder=10)
        f.write('{:s},{:s},{:.3f}\n'.format(particle,label_value,((stability_data.iloc[-1,ind] - stability_data.iloc[0,ind])/stability_data.iloc[0,ind]*100)))
fig.legend(loc='right', ncol=1)
plt.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.15)
plt.savefig(figPath+'/'+pfv.galaxyName+'_stability.png')

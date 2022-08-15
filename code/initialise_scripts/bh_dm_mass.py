import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

import cm_functions as cmf


#get the command line arguments
parser = cmf.utils.argparse_for_initialise(description='Determine the SMBH and DM Halo masses as a function of the stellar mass.', update_help='allow updates of SMBH mass, DM mass, and redshift')
args = parser.parse_args()

pfv = cmf.utils.read_parameters(args.paramFile)

print('\nRunning bh_dm_mass.py\n')

galaxy = cmf.initialise.galaxy_ic_base(pfv, stars=True, dm=True, bh=True)

galaxy.dm.peak_mass = (1, None)
galaxy.bh.mass = (1, None)

markersz = 1.5
linewd = 1


#read in literature data
mass_data = pd.read_table(galaxy.general.lit_location + '/' + pfv.massData, sep=',')
bh_data = pd.read_table(galaxy.general.lit_location + '/' + pfv.bulgeBHData, sep=',', header=0)
#restrict to only ETGs (exclude also S0)
bh_data = bh_data.loc[np.logical_or(bh_data.loc[:,'Type']=='E', bh_data.loc[:,'Type']=='ES'), :]

cored_galaxies = np.zeros(bh_data.shape[0], dtype='bool')
for ind, gal in enumerate(bh_data.loc[:,'Galaxy']):
    if gal[-1] == 'a':
        cored_galaxies[ind] = 1
bh_data.insert(2, 'Cored', cored_galaxies)

cmf.utils.create_error_col(bh_data, 'logM*_sph')
cmf.utils.create_error_col(bh_data, 'logMbh')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4.5))

#plot bulge mass distribution
ax1.hist(mass_data.loc[:,'logMstar'], 20, density=False, weights=np.ones(mass_data.loc[:,'logMstar'].shape[0])/mass_data.loc[:,'logMstar'].shape[0], facecolor='tab:blue', alpha=0.5, linewidth=0.8, edgecolor='black', label='SDSS DR7')
ax1.axvline(x=galaxy.stars.log_total_mass, color='tab:red', label='Simulation')
ax1.axvline(x=np.mean(mass_data.loc[:,'logMstar']), color='tab:red', ls='--', label='Mean')
ax1.axvline(x=np.mean(mass_data.loc[:,'logMstar'])+np.std(mass_data.loc[:,'logMstar']), color='tab:red', ls=':', lw=linewd, label=r'1$\sigma$')
ax1.axvline(x=np.mean(mass_data.loc[:,'logMstar'])-np.std(mass_data.loc[:,'logMstar']), color='tab:red', ls=':', lw=linewd)
ax1.set_xlabel(r'log(M$_\mathrm{bulge}$ / M$_\odot)$')
ax1.set_ylabel(r'Proportion')
ax1.legend(loc='upper left', fontsize='x-small')
ax1.set_title('SDSS Bulge Mass Distribution')


#plot bh - bulge relation
cols = cmf.plotting.mplColours()
ax2.set_xlim(8.7, 12.5)
ax2.set_ylim(7.2, 11)
ax2.set_xlabel(r'log(M$_\mathrm{bulge}$/M$_\odot$)')
ax2.set_ylabel(r'log(M$_\bullet$/M$_\odot$)')
ax2.errorbar(bh_data.loc[bh_data.loc[:,'Cored'], 'logM*_sph'], bh_data.loc[bh_data.loc[:,'Cored'], 'logMbh'], xerr=bh_data.loc[bh_data.loc[:,'Cored'],'logM*_sph_ERR'], yerr=bh_data.loc[bh_data.loc[:,'Cored'],'logMbh_ERR'], ls='', marker='.', elinewidth=linewd/1.2, alpha=0.8,  label='Cored', zorder=3)
ax2.errorbar(bh_data.loc[~bh_data.loc[:,'Cored'], 'logM*_sph'], bh_data.loc[~bh_data.loc[:,'Cored'], 'logMbh'], xerr=bh_data.loc[~bh_data.loc[:,'Cored'],'logM*_sph_ERR'], yerr=bh_data.loc[~bh_data.loc[:,'Cored'],'logMbh_ERR'], ls='', marker='.', elinewidth=linewd/1.2, alpha=0.8,  label=r'S$\acute\mathrm{e}$rsic', zorder=3)
logmstar_seq = np.linspace(8, 12, 500)
ax2.plot(logmstar_seq, cmf.literature.Sahu19(logmstar_seq), c='k', alpha=0.4)
ax2.scatter(galaxy.stars.log_total_mass, galaxy.bh.log_mass, zorder=10, ls='None', color='tab:red', label='Simulation')
ax2.legend(fontsize='x-small', loc='upper left')
ax2.set_title('Bulge - BH Mass')


#plot bulge mass - DM halo mass relation
halo_mass_seq, moster_seq = cmf.literature.Moster10(galaxy.stars.total_mass, [1e10, 1e15], z=galaxy.general.redshift,  plotting=True)
ax3.plot(halo_mass_seq, moster_seq-halo_mass_seq, label='Moster+10', lw=linewd, color=cols[0])
halo_mass_seq, girelli_seq = cmf.literature.Girelli20(galaxy.stars.total_mass, [1e10, 1e15], z=galaxy.general.redshift,  plotting=True)
ax3.plot(halo_mass_seq, girelli_seq-halo_mass_seq, label='Girelli+20', lw=linewd, color=cols[1])
halo_mass_seq, behroozi_seq = cmf.literature.Behroozi19(galaxy.stars.total_mass, [1e10, 1e15], z=galaxy.general.redshift, plotting=True)
ax3.plot(halo_mass_seq, behroozi_seq-halo_mass_seq, label='Behroozi+19', lw=linewd, color=cols[2])
ax3.scatter(galaxy.dm.log_peak_mass, galaxy.stars.log_total_mass-galaxy.dm.log_peak_mass, color='tab:red', zorder=10, label='Simulation')
ax3.set_xlabel(r'log(M$_\mathrm{halo}$/M$_\odot$)')
ax3.set_ylabel(r'log(M$_\mathrm{stellar}$/M$_\mathrm{halo}$)')
ax3.legend(loc='lower right', fontsize='x-small')
ax3.set_title(r'M$_\mathrm{stellar} - $M$_\mathrm{DM}$')
ax3.text(13.5, -1.5, 'z: {:.2f}'.format(galaxy.general.redshift))

plt.subplots_adjust(wspace=0.4, left=0.1, right=0.98)
plt.savefig(galaxy.general.figure_location+'/'+galaxy.general.name+'_ic.png', dpi=300)

if True:
    print('Now run galaxy_gen.py to generate system.')
if args.parameter_update:
    cmf.utils.write_parameters(pfv, allow_updates=('BH_mass', 'DM_peak_mass', 'redshift'))
else:
    cmf.utils.write_parameters(pfv)

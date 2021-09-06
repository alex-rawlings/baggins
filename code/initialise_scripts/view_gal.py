import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf

parser = cmf.utils.argparse_for_initialise(description='View the initial conditions.')
args = parser.parse_args()

print('\nRunning view_gal.py\n')

#get the parameter file
pfv = cmf.utils.read_parameters(args.paramFile)
fig_loc = pfv.saveLocation + '/' + pfv.galaxyName + '/' + pfv.figureLocation
snap = pygad.Snapshot(pfv.saveLocation + '/' + pfv.galaxyName + '/' + pfv.galaxyName + '.hdf5')
snap.to_physical_units()

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6, 3.5))
ax1.set_title('Stars')
ax2.set_title('DM Halo')
_,ax1,*_ = pygad.plotting.image(snap.stars, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax1)
_,ax2,*_ = pygad.plotting.image(snap.dm, qty='mass', Npx=800, yaxis=2, fontsize=10, cbartitle='', scaleind='labels', ax=ax2)
fig.tight_layout()
plt.savefig('{}/{}_view.png'.format(fig_loc, pfv.galaxyName), dpi=300)
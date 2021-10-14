import matplotlib.pyplot as plt
import pygad
import cm_functions as cmf

parser = cmf.utils.argparse_for_initialise(description='View the initial conditions or snapshot.', allow_abbrev=False)
parser.add_argument("-s", "--snap", help="view an arbitrary snapshot",action="store_true", dest="snapview")
parser.add_argument("-V", "--View", help="view the snapshot", action="store_true", dest="view")
parser.add_argument("-o", "--orientate", help="orientate the galaxy", dest="orientate", choices=["ri", "L"], default=None)
args = parser.parse_args()

print('\nRunning view_gal.py\n')

if args.snapview:
    #load the snapshot
    if args.verbose:
        print("Reading from a user-defined snapshot...")
    snap = pygad.Snapshot(args.paramFile)
else:
    #get the parameter file
    print("Reading from a parameter file...")
    pfv = cmf.utils.read_parameters(args.paramFile)
    fig_loc = pfv.saveLocation + '/' + pfv.galaxyName + '/' + pfv.figureLocation
    snap = pygad.Snapshot(pfv.saveLocation + '/' + pfv.galaxyName + '/' + pfv.galaxyName + '.hdf5')
snap.to_physical_units()

if args.orientate is not None:
    if args.orientate == "ri":
        orientate_snap = "red I"
    else:
        orientate_snap = "L"
else:
    orientate_snap = None

cmf.plotting.plot_galaxies_with_pygad(snap, extent={"stars":400, "dm":5000}, orientate=orientate_snap)
if not args.snapview:
    plt.savefig('{}/{}_view.png'.format(fig_loc, pfv.galaxyName), dpi=300)
if args.view:
    plt.show()
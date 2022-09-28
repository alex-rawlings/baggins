import argparse
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Create ICs for Gadget which are somewhat inspired by observations", allow_abbrev=False)
parser.add_argument(type=str, help="parameter file", dest="pf")
parser.add_argument("-n", "--numberRots", type=int, help="number of rotations for projected quantities", dest="nrot", default=3)
args = parser.parse_args()


galaxy = cmf.initialise.GalaxyIC(parameter_file=args.pf)
galaxy.plot_mass_scaling_relations()
galaxy.generate_galaxy()
galaxy.plot_ic_kinematics(num_rots=args.nrot)

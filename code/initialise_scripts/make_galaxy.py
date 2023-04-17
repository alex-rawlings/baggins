import argparse
import re
import numpy as np
import cm_functions as cmf


parser = argparse.ArgumentParser(description="Create ICs for Gadget which are somewhat inspired by observations", allow_abbrev=False)
parser.add_argument(type=str, help="parameter file", dest="pf")
parser.add_argument("-n", "--numberRots", type=int, help="number of rotations for projected quantities", dest="nrot", default=3)
parser.add_argument("-b", "--batch", type=int, help="create a batch of galaxy ICs", dest="batch", default=1)
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args = parser.parse_args()


SL = cmf.ScriptLogger("script", console_level=args.verbose)
rng = np.random.default_rng()

if args.batch == 1:
    galaxy = cmf.initialise.GalaxyIC(parameter_file=args.pf)
    plot_flag = not any(getattr(galaxy, a) is None for a in ["stars", "dm", "bh"])
    if plot_flag:
        galaxy.plot_mass_scaling_relations()
    galaxy.generate_galaxy()
    if plot_flag:
        galaxy.plot_ic_kinematics(num_rots=args.nrot)
else:
    suffixes = "abcdefghijklmnopqrstuvwxyz"
    try:
        assert args.batch < len(suffixes)
    except AssertionError:
        SL.logger.exception(f"Can create a maximum of 26 galaxy ICs, {len(suffixes)} is too great!", exc_info=True)
        raise
    for s in suffixes[:args.batch]:
        # create a new parameter file for each realisation
        try:
            new_filename = cmf.utils.create_file_copy(args.pf, suffix=f"_{s}", exist_ok=False)
        except AssertionError:
            SL.logger.warning(f"Model realisation {s} already exists, skipping...")
            continue
        with open(new_filename, "r+") as f:
            contents = f.read()
            match = re.search("  galaxy_name:.*", contents, flags=re.MULTILINE)
            gal_name = match.group(0).replace("  galaxy_name: ", "").strip("'")
            gal_name = f"{gal_name}_{s}"
            for k, v in zip(("  galaxy_name:", "  random_seed:"), (gal_name, rng.integers(100000))):
                try:
                    assert isinstance(v, (str, int, np.int64, np.int32))
                except AssertionError:
                    SL.logger.exception(f"Only datatypes 'str' and 'int'-like are supported, not type {type(v)}", exc_info=True)
                    raise
                contents, sc = re.subn(f"{k}.*", f"{k} {str(v)}", contents, flags=re.MULTILINE)
                try:
                    assert sc == 1
                except AssertionError:
                    SL.logger.exception(f"Parameter '{k}' not updated properly! {sc} replacements were made!", exc_info=True)
                    raise
            # overwrite copied parameter file
            cmf.utils.overwrite_parameter_file(f, contents)
        SL.logger.warning(f"File {new_filename} created")
        # generate the galaxy
        galaxy = cmf.initialise.GalaxyIC(parameter_file=new_filename)
        galaxy.generate_galaxy()



import argparse
import os.path
import re
import numpy as np
import cm_functions as cmf


parser = argparse.ArgumentParser(
    description="Set up or perturb a merger system", allow_abbrev=False
)
parser.add_argument(type=str, dest="paramfile", help="path to parameter file")
parser.add_argument(
    type=str,
    dest="method",
    help="set up new, or perturb BH or field particle",
    choices=["new", "field", "bh"],
)
parser.add_argument(
    "-o",
    "--overwrite",
    dest="overwrite",
    help="allow overwriting of files?",
    action="store_true",
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    help="create a batch of merger ICs",
    dest="batch",
    default=1,
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=cmf.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()


SL = cmf.setup_logger("script", console_level=args.verbose)
rng = np.random.default_rng()


if args.method == "new":
    if args.batch == 1:
        # set up a single merger
        merger = cmf.initialise.MergerIC(args.paramfile, exist_ok=args.overwrite)
        merger.setup()
    else:
        # set up a number of mergers where the galaxy ICs have been created
        # using a batch method
        suffixes = "abcdefghijklmnopqrstuvwxyz"
        total_combins = int(len(suffixes) * (len(suffixes) + 1) / 2)
        try:
            assert args.batch < total_combins
        except AssertionError:
            SL.exception(
                f"Can create a maximum of 26 galaxy ICs, {len(total_combins)} is too great!",
                exc_info=True,
            )
            raise
        for i, s1 in enumerate(suffixes[: args.batch], start=1):
            for s2 in suffixes[i : args.batch]:
                # create a new parameter file for each realisation
                try:
                    new_filename = cmf.utils.create_file_copy(
                        args.paramfile, suffix=f"_{s1}{s2}", exist_ok=False
                    )
                except AssertionError:
                    SL.warning(
                        f"Merger realisation {s1}-{s2} already exists, skipping..."
                    )
                    continue
                with open(new_filename, "r+") as f:
                    contents = f.read()
                    # update the random seed
                    contents, sc = re.subn(
                        f"  random_seed:.*",
                        f"  random_seed: {str(rng.integers(100000))}",
                        contents,
                        flags=re.MULTILINE,
                    )
                    try:
                        assert sc == 1
                    except AssertionError:
                        SL.exception(
                            f"Parameter 'random_seed' not updated properly! {sc} replacements were made!",
                            exc_info=True,
                        )
                        raise
                    for k, v in zip(("  galaxy_name_1:", "  galaxy_name_2:"), (s1, s2)):
                        # update the galaxy names
                        try:
                            assert isinstance(v, str)
                        except AssertionError:
                            SL.exception(
                                f"Only datatypes 'str' is supported, not type {type(v)}",
                                exc_info=True,
                            )
                            raise
                        match = re.search(f"{k}.*", contents, flags=re.MULTILINE)
                        gal_name = match.group(0).replace(f"{k}", "").strip(" '")
                        gal_name = f"{gal_name}_{v}"
                        contents, sc = re.subn(
                            f"{k}.*", f"{k} {gal_name}", contents, flags=re.MULTILINE
                        )
                        try:
                            assert sc == 1
                        except AssertionError:
                            SL.exception(
                                f"Parameter '{k}' not updated properly! {sc} replacements were made!",
                                exc_info=True,
                            )
                            raise
                    for k, v in zip(("  galaxy_file_1:", "  galaxy_file_2:"), (s1, s2)):
                        # update the galaxy IC files to use
                        match = re.search(f"{k}.*", contents, flags=re.MULTILINE)
                        gal_file_full = match.group(0).replace(f"{k}", "").strip(" '")
                        gal_path, gal_file = os.path.split(gal_file_full)
                        gal_path = f"{gal_path}_{v}"
                        gal_file = gal_file.replace(".hdf5", f"_{v}.hdf5")
                        gal_file_full = os.path.join(gal_path, gal_file)
                        try:
                            assert os.path.exists(gal_file_full)
                        except AssertionError:
                            SL.exception(
                                f"File {gal_file_full} does not exist!", exc_info=True
                            )
                            raise
                        contents, sc = re.subn(
                            f"{k}.*",
                            f"{k} {gal_file_full}",
                            contents,
                            flags=re.MULTILINE,
                        )
                        try:
                            assert sc == 1
                        except AssertionError:
                            SL.exception(
                                f"Parameter '{k}' not updated properly! {sc} replacements were made!",
                                exc_info=True,
                            )
                            raise
                    # overwrite copied parameter file
                    cmf.utils.overwrite_parameter_file(f, contents)
                SL.warning(f"File {new_filename} created")
                # generate the merger
                merger = cmf.initialise.MergerIC(new_filename, exist_ok=args.overwrite)
                merger.setup()
elif args.method == "field":
    if args.batch > 1:
        SL.error(f"Perturbation methods can only be performed for non-batch mode.")
    merger = cmf.initialise.MergerIC(args.paramfile, exist_ok=args.overwrite)
    merger.perturb_field_particle()
else:
    if args.batch > 1:
        SL.error(f"Perturbation methods can only be performed for non-batch mode.")
    merger = cmf.initialise.MergerIC(args.paramfile, exist_ok=args.overwrite)
    merger.perturb_bhs()

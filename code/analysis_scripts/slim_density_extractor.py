import argparse
import os.path
import copy
import numpy as np
import pygad
import baggins as bgs


parser = argparse.ArgumentParser(
    description="Quick extraction method for density profiles, to be used in hierarchical and Bayesian modelling",
    allow_abbrev=False,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(type=str, help="analysis parameters", dest="params")
parser.add_argument(type=str, help="snapshot(s) to extract", dest="snap")
parser.add_argument(
    "-v",
    "--verbosity",
    type=str,
    choices=bgs.VERBOSITY,
    dest="verbose",
    default="INFO",
    help="verbosity level",
)
args = parser.parse_args()


SL = bgs.setup_logger("script", args.verbose)

params = bgs.utils.read_parameters(args.params)

if os.path.isdir(args.snap):
    # do all snapshots in this directory
    snapfiles = bgs.utils.get_snapshots_in_dir(args.snap)
elif os.path.splitext(args.snap)[1] == ".hdf5":
    snapfiles = [args.snap]
else:
    SL.error("Must pass a hdf5 file or path to snapshot files")
    raise RuntimeError

for i, snapfile in enumerate(snapfiles):
    SL.info(f"Doing {i}/{len(snapfiles)}: {snapfile}")
    data_to_save = dict(
        radial_edges=copy.copy(params["galaxy"]["radial_edges"]["value"]),
        projected_mass_density=None,
        eff_radius=None,
        eff_sigma=None,
        rinf=None,
        merger_id=os.path.basename(snapfile).replace(".hdf5", ""),
        particle_masses={"stars": None, "dm": None, "bh": []},
    )
    snap = pygad.Snapshot(snapfile, physical=True)
    centre = pygad.analysis.shrinking_sphere(
        snap.stars,
        pygad.analysis.center_of_mass(snap.stars),
        30,
    )
    SL.debug(f"Centre is {centre}")
    ball_mask = pygad.BallMask(5)
    vcom = pygad.analysis.mass_weighted_mean(snap.stars[ball_mask], "vel")
    pygad.Translation(-centre).apply(snap, total=True)
    pygad.Boost(-vcom).apply(snap, total=True)

    # save particle mass data
    data_to_save["particle_masses"]["stars"] = snap.stars["mass"][0]
    data_to_save["particle_masses"]["dm"] = snap.dm["mass"][0]
    data_to_save["particle_masses"]["bh"] = snap.bh["mass"]

    # get projected denisities
    eff_rad, vsig2_re, vsig2_r, proj_dens = bgs.analysis.projected_quantities(
        snap,
        obs=params["galaxy"]["num_projection_rotations"],
        r_edges=data_to_save["radial_edges"],
    )
    data_to_save["eff_radius"] = list(eff_rad.values())[0]
    data_to_save["eff_sigma"] = list(vsig2_re.values())[0]
    data_to_save["projected_mass_density"] = list(proj_dens.values())[0]

    # get influence radius
    try:
        data_to_save["rinf"] = list(
            bgs.analysis.influence_radius(snap, combined=True).values()
        )[0]
    except ValueError:
        SL.warning("No BH present in snapshot, influence radius will not be calculated!")

    # conserve memory
    snap.delete_blocks()
    del snap
    pygad.gc_full_collect()

    bgs.utils.save_data(
        data_to_save,
        os.path.join(
            params["file_locations"]["cube_dir"], f'{data_to_save["merger_id"]}.pickle'
        ),
    )

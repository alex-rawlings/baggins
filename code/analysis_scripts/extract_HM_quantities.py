import argparse
import multiprocessing as mp
import os
import datetime
import cm_functions as cmf


# set up command line arguments
parser = argparse.ArgumentParser(description="Extract key quantities from a simulation run for use in later Bayesian hierarchical modelling.", allow_abbrev=False)
parser.add_argument(type=str, help="path to merger parameter file", dest="mpf")
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument("-n", "--number", help="perturbation number", dest="pnum", action="append")
parser.add_argument("-o", "--overwrite", help="allow overwriting", dest="overwrite", action="store_true")
args=parser.parse_args()


# helper function to run with multiprocessing
def extractor_helper(child_num):
    data_path = os.path.join(merger_params.full_save_location, merger_params.perturbSubDir, f"{child_num}/output")
    file_save_name = os.path.join(file_save_dir, f"HMQ-cube-{merger_id}-{child_num}.hdf5")
    try:
        hmq = cmf.analysis.HMQuantities(args.apf, data_directory=data_path, merger_id=merger_id)
        hmq.make_hdf5(file_save_name, exist_ok=args.overwrite)
    except:
        print(f"\nWARNING: ERROR IN EXTRACTING FROM CHILD {child_num}")
        print(hmq.analysed_snapshots)
        raise



if __name__ == "__main__":
    # read the required parameter files
    merger_params = cmf.utils.read_parameters(args.mpf)
    analysis_params = cmf.utils.read_parameters(args.apf)

    # determine which perturbation directories to run
    if not args.pnum:
        perturb_dir = os.path.join(merger_params.full_save_location, merger_params.perturbSubDir)
        perturb_ids = next(os.walk(perturb_dir))[1]
    else:
        perturb_ids = args.pnum
    
    # save the cubes to arg.saveloc/Gal1-Gal2-R0-Rperi/
    merger_id = merger_params.full_save_location.rstrip("/").split("/")[-1]
    file_save_dir = os.path.join(analysis_params.cube_dir, merger_id)
    os.makedirs(file_save_dir, exist_ok=args.overwrite)

    # match the number of processes to the number of child runs to process
    num_processes = len(perturb_ids)
    start_time = datetime.datetime.now()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(extractor_helper, perturb_ids)
    print(f"\nElapsed Time: {datetime.datetime.now() - start_time}")
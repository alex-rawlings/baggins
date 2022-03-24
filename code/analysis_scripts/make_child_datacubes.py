import argparse
import multiprocessing as mp
import os
import time
import cm_functions as cmf

#set up command line arguments
parser = argparse.ArgumentParser(description="Create datacubes of simulation perturbed runs.", allow_abbrev=False)
parser.add_argument(type=str, help="path to parameter files", dest="pf")
parser.add_argument(type=str, help="perturbation number", dest="pnum")
parser.add_argument("-a", "--aparams", type=str, help="path to analysis parameter file", dest="apf", default=os.path.join(cmf.HOME, "projects/collisionless-merger-sample/parameters/parameters-analysis/datacubes.py"))
parser.add_argument("-v", "--verbose", help="verbose printing", dest="verbose", action="store_true")
parser.add_argument("-o", "--over", help="allow overwriting", dest="overwrite", action="store_true")
args = parser.parse_args()


#helper function to run
def run_maker(c_dir):
    #make the child sim cube for this subdir in the perturbation directory
    cdc = cmf.analysis.ChildSim(args.pf, c_dir, apfile=args.apf, verbose=args.verbose)
    file_save_name = os.path.join(file_save_dir, "cube-{}.hdf5".format(cdc.merger_name))
    cdc.make_hdf5(file_save_name)


if __name__ == "__main__":
    pfv = cmf.utils.read_parameters(args.pf, verbose=False)
    afv = cmf.utils.read_parameters(args.apf, verbose=False)
    #determine which perturbation directories to run
    if args.pnum == "all":
        perturb_dir = os.path.join(pfv.full_save_location, pfv.perturbSubDir)
        perturb_ids = next(os.walk(perturb_dir))[1]
    else:
        perturb_ids = [args.pnum]
    
    #save the cubes to arg.saveloc/Gal1-Gal2-R0-Rperi/
    file_save_dir = os.path.join(afv.cube_dir, pfv.full_save_location.rstrip("/").split("/")[-1])
    os.makedirs(file_save_dir, exist_ok=args.overwrite)

    #match the number of processes to the number of child runs to process
    num_processes = len(perturb_ids)
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(run_maker, perturb_ids)
    print("\n\nElapsed Time: {:.3f}".format(time.time() - start_time))

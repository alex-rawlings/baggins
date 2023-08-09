import argparse
import multiprocessing as mp
import os
import re
import datetime
from numpy import arange
import cm_functions as cmf


# set up command line arguments
parser = argparse.ArgumentParser(description="Extract key quantities from a simulation run for use in later Bayesian hierarchical modelling.", allow_abbrev=False)
parser.add_argument(type=str, help="path to analysis parameter file", dest="apf")
parser.add_argument(type=str, help="path to merger parameter file(s)", dest="mpf")
parser.add_argument("-m", "--method", help="method used to generate sample", type=str, dest="method", choices=["mc", "bump"], default="bump")
parser.add_argument("-n", "--number", help="perturbation number", dest="pnum", action="append")
parser.add_argument("-o", "--overwrite", help="allow overwriting", dest="overwrite", action="store_true")
parser.add_argument("-v", "--verbosity", type=str, choices=cmf.VERBOSITY, dest="verbose", default="INFO", help="verbosity level")
args=parser.parse_args()

SL = cmf.ScriptLogger("script", args.verbose)


class _Extractor:
    def __init__(self, apf, mpf, overwrite=False) -> None:
        """
        Data extraction base class, should not be directly called

        Parameters
        ----------
        apf : str, path-like
            path to analysis parameter file
        mpf : list
            list of merger parameter files
        overwrite : bool, optional
            allow output directory overwriting, by default False
        """
        self.analysis_params_file = apf
        self.merger_params_files = mpf
        self.overwrite = overwrite
        self.analysis_params = cmf.utils.read_parameters(apf)
        self.merger_params = []
        for m in self.merger_params_files:
            self.merger_params.append(
                cmf.utils.read_parameters(m)
            )
        self._make_family_name()
        self._merger_ids = []
        self._child_dirs = []
        self._cube_files = []
    
    @property
    def merger_family(self):
        return self._merger_family

    @property
    def merger_ids(self):
        return self._merger_ids

    @property
    def child_dirs(self):
        return self._child_dirs

    @property
    def cube_files(self):
        return self._cube_files


    def _make_family_name(self):
        """
        Create the family name for a set of simulations, will be the directory within which each of the simulation cubes will be stored. Assumes that given a `merger_parameter.yml` file, a unique family name can be constructed by taking the galaxy progenitor names (removing any within-set identifiers such as '_a', '_A' up to 26, i.e. '_z'), and then appending the r0 and rperi distance, as was done to make the simulation directory.
        """
        _helper = lambda x: re.sub(r"[-_][a-z]$", "", x, count=1, flags=re.IGNORECASE)
        gal1 = self.merger_params[0]["general"]["galaxy_name_1"]
        gal2 = self.merger_params[0]["general"]["galaxy_name_2"]
        gal1_basename = _helper(gal1)
        gal2_basename = _helper(gal2)
        sim_dir = self.merger_params[0]["calculated"]["full_save_location"].rstrip("/").split("/")[-1]
        orbit_par = re.sub(f"{gal1}-{gal2}-", "", sim_dir, count=1)
        self._merger_family = f"{gal1_basename}-{gal2_basename}-{orbit_par}"


    def _create_cube_names(self):
        """
        Create the cube names for each simulation in a set
        """
        d = os.path.join(self.analysis_params["file_locations"]["cube_dir"], f"{self.merger_family}")
        os.makedirs(d, exist_ok=self.overwrite)
        for m in self.merger_ids:
            self._cube_files.append(f"{d}/HMQ-cube-{m}.hdf5")


    def extract(self, i):
        try:
            hmq = cmf.analysis.HMQuantities(self.analysis_params_file, self.merger_params_files[i], self.child_dirs[i], self.merger_ids[i])
            hmq.make_hdf5(self.cube_files[i], self.overwrite)
        except:
            SL.logger.exception(f"UNABLE TO EXTRACT FROM CHILD {self.child_dirs[i]}", exc_info=True)



class ExtractorMCS(_Extractor):
    def __init__(self, apf, mpf, overwrite=False) -> None:
        """
        Data extraction class for sample run with independent MC sampling

        Parameters
        ----------
        apf : str, path-like
            path to analysis parameter file
        mpf : list
            list of merger parameter files
        overwrite : bool, optional
            allow output directory overwriting, by default False
        """
        super().__init__(apf, mpf, overwrite)
        run_dir = self.merger_params[0]["file_locations"]["save_location"]
        self._child_dirs = [os.path.join(run_dir, f"{p}/output") for p in next(os.walk(run_dir))[1]]
        self._child_dirs.sort()
        for m in self.merger_params:
            self._merger_ids.append(
                m["calculated"]["full_save_location"].rstrip("/").split("/")[-1]
                )
        self._create_cube_names()



class ExtractorPS(_Extractor):
    def __init__(self, apf, mpf, overwrite=False, subdir=None) -> None:
        """
        Data extraction class for sample run with perturbation method

        Parameters
        ----------
        apf : str, path-like
            path to analysis parameter file
        mpf : list
            list of merger parameter files
        overwrite : bool, optional
            allow output directory overwriting, by default False
        subdir : list, optional
            list of specific perturbation directories to run, by default None
        """
        super().__init__(apf, mpf, overwrite)
        run_dir = os.path.join(self.merger_params[0]["calculated"]["full_save_location"], self.merger_params[0]["file_locations"]["perturb_sub_dir"])
        if subdir is None:
            self._child_dirs = [os.path.join(run_dir, f"{p}/output") for p in next(os.walk(run_dir))[1]]
        else:
            self._child_dirs = [os.path.join(run_dir, f"{p}/output") for p in subdir]
        self._child_dirs.sort()
        for d in self.child_dirs:
            self._merger_ids.append(
                f"{self.merger_family}-{d.rstrip('/').split('/')[-2]}"
            )
        self._create_cube_names()



if __name__ == "__main__":
    if args.method == "mc":
        if args.pnum is not None:
            SL.logger.warning(f"Specific perturbation number analysis is not possible for method type 'mc'!")
        if os.path.isfile(args.mpf):
            extractor = ExtractorMCS(args.apf, [args.mpf], args.overwrite)
        elif os.path.isdir(args.mpf):
            mpfs = cmf.utils.get_files_in_dir(args.mpf, ext=".yml")
            extractor = ExtractorMCS(args.apf, mpfs, args.overwrite)
        else:
            try:
                assert 0
            except AssertionError:
                SL.logger.exception("A merger parameter file or a directory of merger parameter files must be specified", exc_info=True)
                raise
    else:
        extractor = ExtractorPS(args.apf, [args.mpf], args.overwrite, args.pnum)

    SL.logger.debug(f"Directories to analyse ({len(extractor.child_dirs)}):\n{extractor.child_dirs}")

    # match the number of processes to the number of child runs to process
    num_processes = len(extractor.child_dirs)
    start_time = datetime.datetime.now()
    with mp.Pool(processes=num_processes) as pool:
        pool.map(extractor.extract, arange(num_processes))
    print(f"\nElapsed Time: {datetime.datetime.now() - start_time}")


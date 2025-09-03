import pickle
import os
import inspect
import shutil
import h5py
from datetime import datetime, timezone
from multiprocessing import managers
from baggins.env_config import _cmlogger, git_hash, TMPDIRs


__all__ = [
    "save_data",
    "load_data",
    "get_files_in_dir",
    "get_snapshots_in_dir",
    "get_ketjubhs_in_dir",
    "create_file_copy",
    "get_mod_time",
]

_logger = _cmlogger.getChild(__name__)


def save_data(data, filename, protocol=pickle.HIGHEST_PROTOCOL, exist_ok=False):
    """
    Convenience function to save multiple objects to a pickle file, so that it
    may be read in again later.

    Parameters
    ----------
    data : dict
        variable names: value pairs whose data is to be saved
    filename : str
        filename to save to
    protocol : pickle.protocol, optional
        saving protocol, by default pickle.HIGHEST_PROTOCOL
    exist_ok : bool, optional
        allow files to be overwritten, by default False

    Raises
    ------
    AssertionError:
        data must be a dict
    AssertionError
        filename must have .pickle extension

    """
    if os.path.exists(filename) and not exist_ok:
        raise FileExistsError(filename)
    try:
        assert isinstance(data, (dict, managers.DictProxy))
    except AssertionError:
        _logger.exception("Input data must be a dict!", exc_info=True)
        raise
    try:
        assert filename.endswith(".pickle")
    except AssertionError:
        _logger.exception(
            f"Filename must be a .pickle file, not type {os.path.splitext(filename)[1]}",
            exc_info=True,
        )
        raise
    try:
        assert all([k not in data for k in ["__githash", "__script"]])
        data["__githash"] = git_hash
        data["__script"] = inspect.stack()[-1].filename
    except AssertionError:
        _logger.exception(
            "Reserved keyword has been used in input data!", exc_info=True
        )
        raise
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=protocol)
    _logger.info(f"File {filename} saved")


def load_data(filename, load_meta=False):
    """
    Convenience function to load pickle data

    Parameters
    ----------
    filename : str
         file to read in
    load_meta : bool, optional
        return meta data of pickle file, by default False

    Returns
    -------
    : dict
        variable names: value pairs
    """
    try:
        f_ext = os.path.splitext(filename)[1]
        assert f_ext == ".pickle"
    except AssertionError:
        _logger.exception(f"File must be a .pickle file, not {f_ext}", exc_info=True)
        raise
    with open(filename, "rb") as f:
        d = pickle.load(f)
    if load_meta:
        return d
    else:
        for k in ["__githash", "__script"]:
            d.pop(k)
        return d


def get_files_in_dir(path, ext=".hdf5", name_only=False, recursive=False):
    """
    Get a list of the full-path name of all files within a directory.

    Parameters
    ----------
    path : str
        host directory of files
    ext : str, optional
        file extension, by default ".hdf5"
    name_only : bool, optional
        return only the name of the file?, by default False
    recursive : bool, optional
        perform a recursive search? (Uses slower os.walk() function), by
        default False

    Returns
    -------
    file_list : list
        alphabetically-sorted list of files
    """
    returntype = "name" if name_only else "path"
    file_list = []
    if recursive:
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.split(".")[-1] == ext.lstrip("."):
                    file_list.append(os.path.join(root, f))
    else:
        with os.scandir(path) as s:
            for entry in s:
                if entry.name.endswith(ext):
                    file_list.append(getattr(entry, returntype))
    file_list.sort()
    return file_list


def get_snapshots_in_dir(path, ext=".hdf5", exclude=[]):
    """
    Get a list of the full-path name of all snapshots within a directory.

    Parameters
    ----------
    path : str
        host directory of snapshot files
    ext : str, optional
        file extension, by default '.hdf5'
    exclude : list, optional
        files with extension ext to exclude (ketju_bhs* always excluded), by
        default []

    Returns
    -------
    all_files : list
        alphabetically-sorted list of snapshot files
    """
    all_files = get_files_in_dir(path, ext=ext)
    exclude.append("ketju_bhs")
    for e in exclude:
        all_files = [f for f in all_files if e not in f]
    # re-sort, to protect agaist cases where there are more than 1000 snaps
    all_files.sort(
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    )
    # filter out corrupt snaps
    bad_snaps = []
    has_bak_snaps = False
    for s in all_files:
        try:
            with h5py.File(s, "r") as f:
                # this throws an error if corrupt
                f["Header"].attrs
        except KeyError:
            _logger.warning(
                f"Snapshot {s} potentially corrupt. Removing it from the list of snapshots!"
            )
            bad_snaps.append(s)
        if not has_bak_snaps and "bak-" in s:
            _logger.warning(
                "A 'bak-' file has been detected! The alphabetical order of the snapshot list cannot be guaranteed!"
            )
            has_bak_snaps = True
    return [a for a in all_files if a not in bad_snaps]


def get_ketjubhs_in_dir(path, file_name="ketju_bhs.hdf5"):
    """
    Get a list of the full-path name of all ketju BH data files within a
    directory. This is a recursive method.

    Parameters
    ----------
    path : str
        host directory of ketju bh files
    file_name : str, optional
        name of ketju file, by default "ketju_bhs.hdf5"

    Returns
    -------
    bh_files : list
        alphabetically-sorted list of ketju bh files
    """
    bh_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == file_name:
                new_f = create_file_copy(os.path.join(root, f))
                bh_files.append(new_f)
    bh_files.sort()
    return bh_files


def create_file_copy(f, suffix="_cp", exist_ok=True):
    """
    Create a copy of a file by appending <suffix> to the file name.

    Parameters
    ----------
    f : str
        file to copy
    suffix : str, optional
        add a suffix to the copy, by default "_cp"
    exist_ok : bool, optional
        allow overwriting of existing files, by default True

    Returns
    -------
    new_f : str
        copied file name
    """
    fname, fext = os.path.splitext(f)
    new_f = f"{fname}{suffix}{fext}"
    if not exist_ok:
        try:
            assert not os.path.exists(new_f)
        except AssertionError:
            _logger.exception(
                f"File '{new_f}' already exists: a new copy cannot be made!",
                exc_info=True,
            )
            raise
    # only copy file if the modification timestamp is more recent than an
    # already existing copy
    if not os.path.exists(new_f) or (os.path.getmtime(new_f) < os.path.getmtime(f)):
        try:
            shutil.copyfile(f, new_f)
        except PermissionError as e:
            # copy to a temporary directory
            if not TMPDIRs.register:
                TMPDIRs.make_new_dir()
            _logger.debug(
                f"{e}\n > Value error trying to copy {f}: copying to temporary directory {TMPDIRs.register[-1]}"
            )
            new_f = os.path.join(TMPDIRs.register[-1], new_f.replace("/", "_"))
            shutil.copyfile(f, new_f)
    return new_f


def get_mod_time(f):
    """
    Get the modification time of a file in UTC.

    Parameters
    ----------
    f : str, path-like
        file to check

    Returns
    -------
    : float
        timestamp of last modification
    """
    return datetime.fromtimestamp(os.path.getmtime(f), tz=timezone.utc).timestamp()

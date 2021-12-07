import pickle
import os


__all__ = ['save_data', 'load_data', "get_files_in_dir", 'get_snapshots_in_dir', "get_ketjubhs_in_dir"]


# TODO: not memory efficient to be using dicts for data
#would be better to save each object as a separate object
#and then use a generator to read in each object sequentially
# QUESTION: could we read in specific separate objects using a variable name?

def save_data(data, filename, protocol=pickle.HIGHEST_PROTOCOL):
    """
    Convenience function to save multiple objects to a pickle file, so that it
    may be read in again later.

    Parameters
    ----------
    data: dict of variable names whose data is to be saved
    filename: the filename to save to
    protocol: pickling protocol (int, 0-5)
    """
    # TODO: enforce a .pickle extension?
    assert(isinstance(data, dict))
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def load_data(filename):
    """
    Convenience function to load pickle data

    Parameters
    ----------
    filename: the file to read in
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_files_in_dir(path, ext=".hdf5", name_only=False):
    """
    Get a list of the full-path name of all files within a directory.

    Parameters
    ----------
    path: host directory of files
    ext: file extension

    Returns
    -------
    alphabetically-sorted list of files
    """
    returntype = "name" if name_only else "path"
    file_list = []
    with os.scandir(path) as s:
        for entry in s:
            if entry.name.endswith(ext):
                file_list.append(getattr(entry, returntype))
    file_list.sort()
    return file_list


def get_snapshots_in_dir(path, ext='.hdf5'):
    """
    Get a list of the full-path name of all snapshots within a directory.

    Parameters
    ----------
    path: host directory of snapshot files
    ext: file extension

    Returns
    -------
    alphabetically-sorted list of snapshot files
    """
    all_files = get_files_in_dir(path, ext=ext)
    return [f for f in all_files if "ketju_bhs" not in f]


def get_ketjubhs_in_dir(path, file_name="ketju_bhs.hdf5"):
    """
    Get a list of the full-path name of all ketju BH data files within a 
    directory.

    Parameters
    ----------
    path: host directory of ketju bh files

    Returns
    -------
    bh_files: alphabetically-sorted list of ketju bh files
    """
    bh_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == file_name:
                bh_files.append(os.path.join(root, f))
    bh_files.sort()
    return bh_files
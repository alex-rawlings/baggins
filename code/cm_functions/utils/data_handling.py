import pickle
import os
import shutil
import h5py


__all__ = ['save_data', 'load_data', "get_files_in_dir", 'get_snapshots_in_dir', "get_ketjubhs_in_dir", "create_file_copy"]


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
    name_only: return only the name of the file, not its full path

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


def get_snapshots_in_dir(path, ext='.hdf5', exclude=[]):
    """
    Get a list of the full-path name of all snapshots within a directory.

    Parameters
    ----------
    path: host directory of snapshot files
    ext: file extension
    exclude: with extension ext to exclude (ketju_bhs* always excluded)

    Returns
    -------
    alphabetically-sorted list of snapshot files
    """
    all_files = get_files_in_dir(path, ext=ext)
    exclude.append("ketju_bhs")
    for e in exclude:
        all_files = [f for f in all_files if e not in f]
    return all_files


def get_ketjubhs_in_dir(path, file_name="ketju_bhs.hdf5", copy=True):
    """
    Get a list of the full-path name of all ketju BH data files within a 
    directory.

    Parameters
    ----------
    path: host directory of ketju bh files
    file_name: name of ketju file
    copy (bool): should a copy be made (needed for ongoing runs)

    Returns
    -------
    bh_files: alphabetically-sorted list of ketju bh files
    """
    bh_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f == file_name:
                new_f = create_file_copy(os.path.join(root, f))
                bh_files.append(new_f)
    bh_files.sort()
    return bh_files


def create_file_copy(f, suffix="_cp"):
    """
    Create a copy of a file by appending <suffix> to the file name.
    """
    fname, fext = os.path.splitext(f)
    new_f = "{}{}{}".format(fname, suffix, fext)
    shutil.copyfile(f, new_f)
    return new_f

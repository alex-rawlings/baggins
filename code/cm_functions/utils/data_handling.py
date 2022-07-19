import pickle
import os
import shutil


__all__ = ["save_data", "load_data", "get_files_in_dir", "get_snapshots_in_dir", "get_ketjubhs_in_dir", "create_file_copy"]


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
    data : dict
        variable names: value pairs whose data is to be saved
    filename : str
        filename to save to
    protocol : pickle.protocol, optional
        saving protocol, by default pickle.HIGHEST_PROTOCOL
    
    Raises
    ------
    AssertionError:
        data must be a dict
    AssertionError
        filename must have .pickle extension

    """
    assert(isinstance(data, dict))
    assert(filename.endswith(".pickle"))
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def load_data(filename):
    """
    Convenience function to load pickle data

    Parameters
    ----------
    filename : str
         file to read in

    Returns
    -------
    : dict
        variable names: value pairs
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


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


def get_snapshots_in_dir(path, ext='.hdf5', exclude=[]):
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
    return all_files


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


def create_file_copy(f, suffix="_cp"):
    """
    Create a copy of a file by appending <suffix> to the file name.

    Parameters
    ----------
    f : str
        file to copy
    suffix : str, optional
        add a suffix to the copy, by default "_cp"

    Returns
    -------
    new_f : str
        copied file name
    """
    fname, fext = os.path.splitext(f)
    new_f = f"{fname}{suffix}{fext}"
    # only copy file if the modification timestamp is more recent than an 
    # already existing copy
    if not os.path.exists(new_f) or (os.path.getmtime(new_f) < os.path.getmtime(f)):
        shutil.copyfile(f, new_f)
    return new_f

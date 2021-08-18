import pickle


__all__ = ['save_data', 'load_data']


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

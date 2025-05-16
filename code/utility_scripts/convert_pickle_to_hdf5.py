import argparse
import os.path
import h5py
from baggins.analysis import HDF5Base
from baggins import utils


class DictRep(HDF5Base):
    def __init__(self, d=None):
        """
        Simple class that wraps a dict into the HDF5Base interface.

        Parameters
        ----------
        d : dict, optional
            data to save, by default None
        """
        super().__init__()
        if d is not None:
            self._new_attrs = []
            for k, v in d.items():
                setattr(self, k, v)
                self._new_attrs.append(k)

    def save(self, fname):
        """
        Save the data.

        Parameters
        ----------
        fname : str
            file name
        """
        with h5py.File(fname, "w") as f:
            self._saver(f.get("/"), self._new_attrs)

    @classmethod
    def load_from_file(cls, fname, decode="utf-8"):
        return super().load_from_file(fname, decode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert a .pickle file to a .hdf5 file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(help="file or path to files", dest="files", type=str)
    args = parser.parse_args()

    if os.path.isfile(args.files):
        all_files = [args.files]
    else:
        all_files = utils.get_files_in_dir(args.files, ".pickle")
    for _file in all_files:
        try:
            data = utils.load_data(_file)
        except AssertionError:
            print(f"Cannot open file {_file}. Skipping.")
            continue
        new_file_name = f"{os.path.splitext(_file)[0]}.hdf5"
        assert not os.path.exists(new_file_name)
        obj = DictRep(data)
        obj.save(new_file_name)
        print(f"Saved {new_file_name}")

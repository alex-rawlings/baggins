import numpy as np
import h5py
import datetime
import pygad
from baggins.general import unit_as_str
from baggins.env_config import _cmlogger, date_format

__all__ = ["HDF5Base"]

_logger = _cmlogger.getChild(__name__)


class HDF5Base:
    def __init__(self) -> None:
        """
        Base class that allows for restoring of HDF5 files to a class. Should
        not be instantiated directly.
        """
        self._log = ""
        self.allowed_types = (
            int,
            float,
            str,
            bytes,
            np.int64,
            np.float32,
            np.float64,
            np.ndarray,
            pygad.UnitArr,
            np.bool8,
            list,
            tuple,
        )
        self.hdf5_file_name = None

    def add_to_log(self, msg):
        """
        Add a message to the log

        Parameters
        ----------
        msg : str
            message to add to the log
        """
        now = datetime.datetime.now()
        now = now.strftime(date_format)
        self._log += f"{now}: {msg}\n"

    @classmethod
    def load_from_file(cls, fname, decode="utf-8"):
        """
        Unpack a HDF5 object into a class where hdf5 fields are saved as class  attributes. Note this class should not be called directly, but
        rather other classes where data is saved as a HDF5 object should inherit
        this class for easy unpacking.

        Parameters
        ----------
        fname : str, path-like
            hdf5 file
        decode : str, optional
            string decoding, by default "utf-8"

        Returns
        -------
        HDF5Base
            an instance of this class
        """
        # first create a new class instance. At this stage, no properties are
        # set
        C = cls()
        C.hdf5_file_name = fname

        # define some helpers

        def _recursive_dict_load(g):
            """
            Recursively load a dict object. This segment is inspired from 3ML
            https://threeml.readthedocs.io/en/stable/

            Parameters
            ----------
            g : h5py.Group
                HDF5 group to load into a dict

            Returns
            -------
            dict
                group is a dict object
            """
            d = {}
            for key, val in g.items():
                if isinstance(val, h5py.Dataset):
                    tmp = val[()]
                    try:
                        d[key] = tmp.decode(decode)
                    except AttributeError:
                        d[key] = tmp
                    # reload units if the data is a pygad.UnitArr
                    for a in val.attrs.values():
                        if np.array_equal(a, "pygad_UnitArr"):
                            d[key] = pygad.UnitArr(d[key], units=val.attrs["units"])
                            break
                    # unpack None value, courtesy Elisa
                    if np.array_equal(d[key], "NONE_TYPE"):
                        d[key] = None
                elif isinstance(val, h5py.Group):
                    d[key] = _recursive_dict_load(val)
            return d

        def _main_setter(k, v):
            """
            Set those class attributes which are datasets.

            Parameters
            ----------
            k : str
                dataset name
            v : h5py.Dataset
                dataset from HDF5 file to set as class attribute
            """
            tmp = v[()]
            try:
                std_val = tmp.decode(decode)
            except AttributeError:
                std_val = tmp
            # reload units if the data is a pygad.UnitArr
            for a in v.attrs.values():
                if np.array_equal(a, "pygad_UnitArr"):
                    std_val = pygad.UnitArr(std_val, units=v.attrs["units"])
                    break
            if np.array_equal(std_val, "NONE_TYPE"):
                std_val = None
            if k == "logs":
                k = "_log"
            setattr(C, k, std_val)

        # now we need to recursively unpack the given hdf5 file
        with h5py.File(fname, mode="r") as f:
            for key, val in f.items():
                if isinstance(val, h5py.Dataset):
                    # these are top level datasets
                    _main_setter(key, val)
                elif isinstance(val, h5py.Group):
                    for kk, vv in val.items():
                        try:
                            assert isinstance(vv, (h5py.Dataset, h5py.Group))
                            if isinstance(vv, h5py.Dataset):
                                _main_setter(kk, vv)
                            else:
                                dict_val = _recursive_dict_load(vv)
                                setattr(C, kk, dict_val)
                        except AssertionError:
                            _logger.exception(
                                f"{kk}: Unkown type for unpacking!", exc_info=True
                            )
                            raise
                        msg = f"Successfully loaded dataset {kk}"
                        _logger.debug(msg)
                        C.add_to_log(msg)
            _logger.debug(f"File {fname} loaded")
        return C

    def _saver(self, g, L):
        """
        Save specified elements to a given HDF5 group.

        Parameters
        ----------
        g : h5py.Group
            group to add elements to
        l : list
            attribute names to add to group
        """
        # attributes defined with the @property method are not in __dict__,
        # but their _members are. Append an underscore to all things in l
        L = ["_" + x for x in L]
        saved_list = []
        for _attr in self.__dict__:
            if _attr not in L:
                continue
            # now we strip the leading underscore if this should be saved
            attr = _attr.lstrip("_")
            attr_val = getattr(self, attr)
            try:
                assert (
                    isinstance(attr_val, self.allowed_types)
                    or isinstance(attr_val, dict)
                    or attr_val is None
                )
                if isinstance(attr_val, self.allowed_types):
                    dset = g.create_dataset(attr, data=attr_val)
                    if isinstance(attr_val, pygad.UnitArr):
                        dset.attrs["special_type"] = "pygad_UnitArr"
                        dset.attrs["units"] = unit_as_str(attr_val.units)
                elif attr is None:
                    g.create_dataset(attr, "NONE_TYPE")
                else:
                    self._recursive_dict_save(g, attr_val, attr)
            except AssertionError:
                _logger.exception(
                    f"Error saving {attr}: cannot save {type(attr_val)} type!"
                )
                raise
            except:  # noqa
                _logger.exception(
                    f"Unable to save <{attr}> (type {type(attr_val)} with values {attr_val})",
                    exc_info=True,
                )
                raise
            saved_list.append(_attr)
        # check that everything was saved
        not_saved = list(set(L) - set(saved_list))
        if not not_saved:
            for i in not_saved:
                msg = f"Property {i.lstrip('_')} was not saved!"
                _logger.warning(msg)
                self.add_to_log(msg)

    def _add_attr(self, dg, aname, aval):
        """
        Add an attribute to a HDF5 group or dataset. This is essentially a
        wrapper that handles None types.

        Parameters
        ----------
        dg : h5py.Dataset, h5py.Group
            dataset or group to add attribute to
        aname : str
            attribute name
        aval : any
            attribute value
        """
        if aval is None:
            aval = "NONE_TYPE"
        dg.attrs[aname] = aval

    def _recursive_dict_save(self, g, d, n):
        """
        Recursively save a dictionary. Inspired from 3ML
        https://threeml.readthedocs.io/en/stable/

        Parameters
        ----------
        g : h5py.Group
            group to save the dictionary to
        d : dict
            dictionary to save
        n : str
            name of the new group to save the dict object under
        """
        gnew = g.create_group(n)
        for key, val in d.items():
            try:
                assert (
                    isinstance(val, self.allowed_types)
                    or isinstance(val, dict)
                    or val is None
                )
                if isinstance(val, self.allowed_types):
                    dset = gnew.create_dataset(key, data=val)
                    if isinstance(val, pygad.UnitArr):
                        dset.attrs["special_type"] = "pygad_UnitArr"
                        dset.attrs["units"] = unit_as_str(val.units)
                elif val is None:
                    gnew.create_dataset(key, data="NONE_TYPE")
                else:
                    self._recursive_dict_save(gnew, val, key)
            except AssertionError:
                _logger.exception(f"Error saving {key}: cannot save {type(val)} type!")
                raise
            except:  # noqa
                _logger.exception(
                    f"Unable to save <{key}> (type {type(val)} with values {val})",
                    exc_info=True,
                )
                raise

    # public functions
    def add_hdf5_field(self, n, val, field, fname=None):
        """
        Add a new field to an existing HDF5 structure.

        Parameters
        ----------
        n : str
            attribute name
        val : any
            attribute value
        field : str
            field where to save the attribute to
        fname : str, path-like, optional
            hdf5 file name, by default None (uses self.hdf5_file_name)
        """
        #
        if fname is None:
            fname = self.hdf5_file_name
        field = field.rstrip("/")
        with h5py.File(fname, mode="a") as f:
            try:
                assert (
                    isinstance(val, self.allowed_types)
                    or isinstance(val, dict)
                    or val is None
                )
                if isinstance(val, self.allowed_types):
                    f.create_dataset(field + "/" + n, data=val)
                elif val is None:
                    f.create_dataset(field + "/" + n, data="NONE_TYPE")
                else:
                    # TODO this may not work...
                    self._recursive_dict_save(f[field], val, n)
            except AssertionError:
                _logger.exception(f"Error saving {n}: cannot save {type(val)} type!")
                raise
            msg = f"Attribute {n} has been added"
            _logger.info(msg)
            self.add_to_log(msg)
            if "/meta" not in f:
                meta = f.create_group("/meta")
            else:
                meta = f["/meta"]
            if "/meta/logs" not in f:
                meta.create_dataset("logs", data=self._log)
            else:
                f["/meta/logs"][...] = self._log

    def update_hdf5_field(self, d, val):
        """
        Update an already-existing HDF5 field.

        Parameters
        ----------
        d : str
            dataset name within existing HDF5 file
        val : any
            value that d is to be updated to
        """
        with h5py.File(self.hdf5_file_name, mode="r+") as f:
            try:
                assert isinstance(val, self.allowed_types) or val is None
                if isinstance(val, self.allowed_types):
                    f[d] = val
                else:
                    f[d] = "NONE_TYPE"
            except AssertionError:
                _logger.exception(
                    f"Error saving {d}: cannot update {type(val)} type!", exc_info=True
                )
            msg = f"Data {d} has been updated"
            _logger.info(msg)
            self.add_to_log(msg)
            if "/meta" not in f:
                meta = f.create_group("/meta")
            else:
                meta = f["/meta"]
            if "/meta/logs" not in f:
                meta.create_dataset("logs", data=self._log)
            else:
                f["/meta/logs"][...] = self._log

    def print_logs(self):
        """
        Print the logs associated with this HDF5 file.
        """
        print(self._log)

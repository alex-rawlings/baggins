import numpy as np
import h5py
import pygad

from . import BHBinaryData
from ...general import unit_as_str

__all__ = ["ChildSimData"]

class ChildSimData(BHBinaryData):
    def __init__(self) -> None:
        """
        A class that defines the fields which constitute the variables of 
        interest for the merger remannt. These properties are accessible to all 
        child classes, and also correspond to the fields which are loadable 
        from a hdf5 file. Those fields which are part of the inherited class
        are also part of this hdf5 file.
        """
        super().__init__()
        self.allowed_types = (int, float, str, bytes, np.int64, np.float64, np.ndarray, pygad.UnitArr, np.bool8, list, tuple)
        self.hdf5_file_name = None
    
    @property
    def parent_quantities(self):
        return self._parent_quantities
    
    @parent_quantities.setter
    def parent_quantities(self, v):
        assert isinstance(v, dict)
        self._parent_quantities = v
    
    @property
    def relaxed_remnant_flag(self):
        return self._relaxed_remnant_flag
    
    @relaxed_remnant_flag.setter
    def relaxed_remnant_flag(self, v):
        #assert isinstance(v, bool)
        self._relaxed_remnant_flag = v

    @property
    def relaxed_stellar_velocity_dispersion(self):
        return self._relaxed_stellar_velocity_dispersion
    
    @relaxed_stellar_velocity_dispersion.setter
    def relaxed_stellar_velocity_dispersion(self, v):
        self._relaxed_stellar_velocity_dispersion = v
    
    @property
    def relaxed_stellar_velocity_dispersion_projected(self):
        return self._relaxed_stellar_velocity_dispersion_projected
    
    @relaxed_stellar_velocity_dispersion_projected.setter
    def relaxed_stellar_velocity_dispersion_projected(self, v):
        assert isinstance(v, dict)
        self._relaxed_stellar_velocity_dispersion_projected = v
    
    @property
    def relaxed_inner_DM_fraction(self):
        return self._relaxed_inner_DM_fraction
    
    @relaxed_inner_DM_fraction.setter
    def relaxed_inner_DM_fraction(self, v):
        self._relaxed_inner_DM_fraction = v
    
    @property
    def virial_info(self):
        return self._virial_info
    
    @virial_info.setter
    def virial_info(self, v):
        self._virial_info = v
    
    @property
    def relaxed_effective_radius(self):
        return self._relaxed_effective_radius
    
    @relaxed_effective_radius.setter
    def relaxed_effective_radius(self, v):
        assert isinstance(v, dict)
        self._relaxed_effective_radius = v
    
    @property
    def relaxed_half_mass_radius(self):
        return self._relaxed_half_mass_radius
    
    @relaxed_half_mass_radius.setter
    def relaxed_half_mass_radius(self, v):
        self._relaxed_half_mass_radius = v
    
    @property
    def relaxed_core_parameters(self):
        return self._relaxed_core_parameters
    
    @relaxed_core_parameters.setter
    def relaxed_core_parameters(self, v):
        self._relaxed_core_parameters = v

    @property
    def relaxed_density_profile(self):
        return self._relaxed_density_profile
    
    @relaxed_density_profile.setter
    def relaxed_density_profile(self, v):
        self._relaxed_density_profile = v

    @property
    def relaxed_density_profile_projected(self):
        return self._relaxed_density_profile_projected
    
    @relaxed_density_profile_projected.setter
    def relaxed_density_profile_projected(self, v):
        self._relaxed_density_profile_projected = v
    
    @property
    def relaxed_triaxiality_parameters(self):
        return self._relaxed_triaxiality_parameters
    
    @relaxed_triaxiality_parameters.setter
    def relaxed_triaxiality_parameters(self, v):
        self._relaxed_triaxiality_parameters = v
    
    @property
    def total_stellar_mass(self):
        return self._total_stellar_mass
    
    @total_stellar_mass.setter
    def total_stellar_mass(self, v):
        self._total_stellar_mass = v
    
    @property
    def ifu_map_ah(self):
        return self._ifu_map_ah
    
    @ifu_map_ah.setter
    def ifu_map_ah(self, v):
        assert isinstance(v, dict)
        self._ifu_map_ah = v
    
    @property
    def ifu_map_merger(self):
        return self._ifu_map_merger
    
    @ifu_map_merger.setter
    def ifu_map_merger(self, v):
        assert isinstance(v, dict)
        self._ifu_map_merger = v
    
    @property
    def snapshot_times(self):
        return self._snapshot_times
    
    @snapshot_times.setter
    def snapshot_times(self, v):
        self._snapshot_times = v
    
    @property
    def stellar_shell_inflow_velocity(self):
        return self._stellar_shell_inflow_velocity
    
    @stellar_shell_inflow_velocity.setter
    def stellar_shell_inflow_velocity(self, v):
        self._stellar_shell_inflow_velocity = v
    
    @property
    def bh_binary_watershed_velocity(self):
        return self._bh_binary_watershed_velocity
    
    @bh_binary_watershed_velocity.setter
    def bh_binary_watershed_velocity(self, v):
        self._bh_binary_watershed_velocity = v
    
    @property
    def beta_r(self):
        return self._beta_r
    
    @beta_r.setter
    def beta_r(self, v):
        assert isinstance(v, dict)
        self._beta_r = v

    @property
    def ang_mom_diff_angle(self):
        return self._ang_mom_diff_angle
    
    @ang_mom_diff_angle.setter
    def ang_mom_diff_angle(self, v):
        self._ang_mom_diff_angle = v
    
    @property
    def loss_cone(self):
        return self._loss_cone
    
    @loss_cone.setter
    def loss_cone(self, v):
        self._loss_cone = v

    @property
    def stars_in_loss_cone(self):
        return self._stars_in_loss_cone
    
    @stars_in_loss_cone.setter
    def stars_in_loss_cone(self, v):
        self._stars_in_loss_cone = v

    @property
    def particle_count(self):
        return self._particle_count
    
    @particle_count.setter
    def particle_count(self, v):
        assert isinstance(v, dict)
        self._particle_count = v
    
    @classmethod
    def load_from_file(cls, fname, decode="utf-8", verbose=False):
        #first create a new class instance. At this stage, no properties are set
        C = cls()
        C.hdf5_file_name = fname

        #define some helpers
        def _recursive_dict_load(g):
            #recursively load a dictionary. Inspired from 3ML
            #g is a group object
            d = {}
            for key, val in g.items():
                if isinstance(val, h5py.Dataset):
                    tmp = val[()]
                    try:
                        d[key] = tmp.decode(decode)
                    except:
                        d[key] = tmp
                    #reload units if the data is a pygad.UnitArr
                    for a in val.attrs.values():
                        if np.array_equal(a, "pygad_UnitArr"):
                            d[key] = pygad.UnitArr(d[key], units=val.attrs["units"])
                            break
                    #unpack None value, courtesy Elisa
                    if np.array_equal(d[key], "NONE_TYPE"):
                        d[key] = None
                elif isinstance(val, h5py.Group):
                    d[key] = _recursive_dict_load(val)
            return d
        
        def _main_setter(k, v):
            #set those class attributes which are datasets
            tmp = v[()]
            try:
                std_val = tmp.decode(decode)
            except:
                std_val = tmp
            #reload units if the data is a pygad.UnitArr
            for a in v.attrs.values():
                if np.array_equal(a, "pygad_UnitArr"):
                    std_val = pygad.UnitArr(std_val, units=v.attrs["units"])
                    break
            if np.array_equal(std_val, "NONE_TYPE"):
                std_val = None
            if k == "logs":
                k = "_log"
            setattr(C, k, std_val)

        #now we need to recursively unpack the given hdf5 file
        with h5py.File(fname, mode="r") as f:
            for key, val in f.items():
                if isinstance(val, h5py.Dataset):
                    #these are top level datasets, and we don't expect there
                    #to be any
                    _main_setter(key, val)
                elif isinstance(val, h5py.Group):
                    #designed that datasets are grouped into two top-level 
                    #groups, so these need care unpacking
                    for kk, vv in val.items():
                        if isinstance(vv, h5py.Dataset):
                            _main_setter(kk, vv)
                        elif isinstance(vv, h5py.Group):
                            dict_val = _recursive_dict_load(vv)
                            setattr(C, kk, dict_val)
                        else:
                            ValueError("{}: Unkown type for unpacking!".format(kk))
                        if verbose:
                            print(" > Successfully loaded group {}".format(kk))
        return C
    
    def _saver(self, g, l):
        #given a HDF5 group g, save all elements in list l
        #attributes defined with the @property method are not in __dict__, 
        #but their _members are. Append an underscore to all things in l
        l = ["_" + x for x in l]
        for attr in self.__dict__:
            if attr not in l:
                continue
            #now we strip the leading underscore if this should be saved
            attr = attr.lstrip("_")
            attr_val = getattr(self, attr)
            if isinstance(attr_val, self.allowed_types):
                dset = g.create_dataset(attr, data=attr_val)
                if isinstance(attr_val, pygad.UnitArr):
                    dset.attrs["special_type"] = "pygad_UnitArr"
                    dset.attrs["units"] = unit_as_str(attr_val.units)
            elif attr is None:
                g.create_dataset(attr, "NONE_TYPE")
            elif isinstance(attr_val, dict):
                self._recursive_dict_save(g, attr_val, attr)
            else:
                raise ValueError("Error saving {}: cannot save {} type!".format(attr, type(attr_val)))

    def _recursive_dict_save(self, g, d, n):
        #recursively save a dictionary. Inspired from 3ML
        #g is group object, d is the dict, n is the new group name
        gnew = g.create_group(n)
        for key, val in d.items():
            if isinstance(val, self.allowed_types):
                dset = gnew.create_dataset(key, data=val)
                if isinstance(val, pygad.UnitArr):
                    dset.attrs["special_type"] = "pygad_UnitArr"
                    dset.attrs["units"] = unit_as_str(val.units)
            elif val is None:
                gnew.create_dataset(key, data="NONE_TYPE")
            elif isinstance(val, dict):
                self._recursive_dict_save(gnew, val, key)
            else:
                raise ValueError("Error saving {}: cannot save {} type!".format(key, type(val)))
    
    #public functions
    def add_hdf5_field(self, n, val, field, fname=None):
        #add a new field to an existing HDF5 structure
        #n is attribute name, val is its value, field is where to save to, 
        #fname is the file name
        if fname is None:
            fname = self.hdf5_file_name
        field = field.rstrip("/")
        with h5py.File(fname, mode="a") as f:
            if isinstance(val, self.allowed_types):
                f.create_dataset(field+"/"+n, data=val)
            elif val is None:
                f.create_dataset(field+"/"+n, data="NONE_TYPE")
            elif isinstance(val, dict):
                # TODO this may not work...
                self._recursive_dict_save(f[field], val, n)
            else:
                raise ValueError("Error saving {}: cannot save {} type!".format(n, type(val)))
            self.add_to_log("Attribute {} has been added".format(n))
            f["/meta/logs"][...] = self._log
    
    def print_logs(self):
        print(self._log)


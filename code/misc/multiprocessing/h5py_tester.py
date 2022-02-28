import h5py
import numpy as np
import datetime
import sys
import os

username = "me"
date_str = "%Y-%m-%d %H:%M:%S"

class h5Test:
    def __init__(self) -> None:
        self.allowed_types = (int, float, str, bytes, np.int64, np.float64, np.ndarray, list)
        self.a = 3
        self.a2 = [1,4,6,8,0]
        self.b = {"{}".format(3.4):np.arange(3), "{}".format(6.4):np.arange(4)}
        self.c = {"x":np.arange(6), "y":np.arange(10,15)}

    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, v):
        self._a = v
    
    @property
    def a2(self):
        return self._a2
    
    @a2.setter
    def a2(self, v):
        self._a2 = v

    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, v):
        self._b = v
    
    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, v):
        self._c = v
        
    def make_hdf5(self, fname):
        #some helper functions
        def _saver(g, l):
            l = ["_"+x for x in l]
            #given a HDF5 group g, save all elements in list l
            for attr in self.__dict__:
                if attr not in l:
                    print("skipping {}".format(attr))
                    continue
                attr = attr.lstrip("_")
                print("stuff for {}".format(attr))
                attr_val = getattr(self, attr)
                if isinstance(attr_val, self.allowed_types):
                    print(attr_val)
                    g.create_dataset(attr, data=attr_val)
                elif attr is None:
                    g.create_dataset(attr, "NONE_TYPE")
                elif isinstance(attr_val, dict):
                    _recursive_dict_save(g, attr_val, attr)
                else:
                    raise ValueError("Error saving {}: cannot save {} type!".format(attr, type(attr_val)))

        def _recursive_dict_save(g, d, n):
            #recursively save a dictionary. Inspired from 3ML
            #g is group object, d is the dict, n is the new group name
            gnew = g.create_group(n)
            for key, val in d.items():
                if isinstance(val, self.allowed_types):
                    gnew.create_dataset(key, data=val)
                elif val is None:
                    gnew.create_dataset(key, data="NONE_TYPE")
                elif isinstance(val, dict):
                    _recursive_dict_save(gnew, val, key)
                else:
                    raise ValueError("Cannot save {} type!".format(type(val)))

        with h5py.File(fname, mode="w") as f:
            #set up some meta data
            meta = f.create_group("meta")
            now = datetime.datetime.now()
            meta.attrs["created"] = now.strftime(date_str)
            meta.attrs["created_by"] = username
            meta.attrs["last_accessed"] = now.strftime(date_str)
            meta.attrs["last_user"] = username
        
            #save the binary info
            bhb = f.create_group("bh_binary")
            data_list = ["a", "a2", "b", "c"]
            _saver(bhb, data_list)
            f["/bh_binary/a"].attrs["quantiles"] = "HAHAH"
        

if __name__=="__main__":
    H = h5Test()
    print(H.__dir__)
    H.make_hdf5("testcube.hdf5")
import h5py
import numpy as np
import datetime
import sys
import os

dt_str = "%Y-%m-%d %H:%M:%S"

with h5py.File("myfile.hdf5", "a") as f:
    if "/meta" not in f:
        meta = f.create_group("meta")
        now = datetime.datetime.now()
        meta.attrs["created"] = now.strftime(dt_str)
        usr = os.path.expanduser("~")
        meta.attrs["created by"] = usr.rstrip("/").split("/")[-1]
    else:
        meta = f["/meta"]
    now = datetime.datetime.now()
    meta.attrs["last modifed"] = now.strftime(dt_str)
    usr = os.path.expanduser("~")
    meta.attrs["last user"] = usr.rstrip("/").split("/")[-1]

with h5py.File("myfile.hdf5", "r") as f:
    for k, v in f["/meta"].attrs.items():
        print("{}: {}".format(k,v))
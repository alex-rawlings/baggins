import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os
import pygad
import ketjugw
import cm_functions as cmf


#define a class to hold extracted information
class member_properties:
    def __init__(self):
        self.galaxy_A = self._properties()
        self.galaxy_B = self._properties()
    
    @property
    def id_order(self):
        return self.__id_order
    
    @id_order.setter
    def id_order(self, arr):
        self.__id_order = arr

    def update_property(self, variable, value):
        for i, gal in enumerate(('galaxy_A', 'galaxy_B')):
            current_value = getattr(getattr(self, gal), variable)
            if isinstance(value, dict):
                idx = self.id_order[i]
            else:
                idx = i
            if variable in ('bh_mass', 'time'):
                updated_value = np.hstack([current_value, value[idx]])
            else:
                updated_value = np.vstack([current_value, value[idx]])
            setattr(getattr(self, gal), variable, updated_value)
    
    def sort_by_time(self):
        for gal in ('galaxy_A', 'galaxy_B'):
            sorted_idx = np.argsort(getattr(getattr(self, gal), 'time'))
            for p in dir(getattr(self, gal)):
                if p == 'bh_mass' or p[:1]=='_':
                    continue
                current_arr = getattr(getattr(self, gal), p)
                if current_arr.ndim == 1:
                    sorted_arr = current_arr[sorted_idx]
                else:
                    sorted_arr = current_arr[sorted_idx, :]
                setattr(getattr(self, gal), p, sorted_arr)
    
    class _properties:
        def __init__(self):
            self.__bh_mass = np.empty(0, float)
            self.time = np.empty(0, float)
            self.xcom = np.empty((0,3), float)
            self.vcom = np.empty((0,3), float)
            self.bhx = np.empty((0,3), float)
            self.bhv = np.empty((0,3), float)
            self.abc = np.empty((0,3), float)
        
        @property
        def bh_mass(self):
            return self.__bh_mass
        
        @bh_mass.setter
        def bh_mass(self, value):
            self.__bh_mass = value


parser = argparse.ArgumentParser(description='Extract data.', allow_abbrev=False)
parser.add_argument(type=str, help='Path to simulation data', dest='path', default=None)
parser.add_argument('-o', '--orbit', type=str, help='Orbital approach type', dest='orbit', default=None)
parser.add_argument('-m' '--max', type=int, help='Maximum snapshot number to process', dest='max', default=None)
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose printing', dest='verbose')
args = parser.parse_args()



#we want to read in a new dataset
for ind, (root, directories, files) in enumerate(os.walk(args.path)):
    if ind == 0:
        #we are in the top directory
        res_dict = dict()
        for d in directories:
            res_dict[d] = member_properties()
    else:
        if args.orbit is not None and args.orbit not in root:
            #this isn't the orbit type we want
            continue
        #we need to know wich resolution we are dealing with
        resolution = [r for r in list(res_dict.keys()) if r in root][0]
        for ind2, file_ in enumerate(files):
            if file_.endswith('.hdf5') and 'ketju_bhs' not in file_:
                full_path_to_file = os.path.join(root, file_)
                #skip over files if desired
                if args.max is not None:
                    if args.orbit is not None and root.split('/')[-1] != args.orbit:
                        file_num = file_.split('_')[-1]
                        file_num = int(file_num.split('.')[0])
                        if file_num > args.max:
                            print('Skipping: {}'.format(full_path_to_file))
                            continue
                print('Reading: {}'.format(full_path_to_file))
                snap = pygad.Snapshot(full_path_to_file)
                snap.to_physical_units()
                #convention! com1 is the smaller ID number
                #an com2 is the larger ID number
                if len(res_dict[resolution].galaxy_A.bh_mass) == 0:
                    if args.verbose:
                        print('Setting BH ID order and masses...')
                    #get the BH masses
                    bh_id_order = np.sort(snap.bh['ID'])
                    res_dict[resolution].id_order = bh_id_order
                    res_dict[resolution].update_property('bh_mass', 
                                                        snap.bh['mass'])
                if args.orbit is not None and root.split('/')[-1] == args.orbit:
                    #this is the ic file
                    id_masks_for = resolution
                    star_id_masks = cmf.analysis.get_all_id_masks(snap)
                    res_dict[resolution].update_property('time', [0, 0])
                else:
                    #this is a regular file
                    snap_time = cmf.general.convert_gadget_time(snap)
                    res_dict[resolution].update_property('time', [snap_time, snap_time])
                #sanity check
                if id_masks_for != resolution:
                    raise RuntimeError('Need the correct ID mask!')
                #get the data
                xcoms = cmf.analysis.get_coms_of_each_galaxy(snap, masks=star_id_masks, verbose=args.verbose)
                vcoms = cmf.analysis.get_com_velocity_of_each_galaxy(snap, xcoms, masks=star_id_masks, verbose=args.verbose)
                
                #and assign to the class
                res_dict[resolution].update_property('xcom', xcoms)
                res_dict[resolution].update_property('vcom', vcoms)
                bhposdict = dict()
                bhveldict = dict()
                for bhid in snap.bh['ID']:
                    bhposdict[bhid] = snap.bh['pos'][snap.bh['ID']==bhid, :]
                    bhveldict[bhid] = snap.bh['vel'][snap.bh['ID']==bhid, :]
                res_dict[resolution].update_property('bhx', bhposdict)
                res_dict[resolution].update_property('bhv', bhveldict)
                snap.delete_blocks()
#sort the data based on time
if args.verbose:
    print('Sorting...')
for r in res_dict:
    print(res_dict[r].galaxy_A.time)
    print(res_dict[r].galaxy_A.xcom)
    res_dict[r].sort_by_time()
    print(res_dict[r].galaxy_A.time)
    print(res_dict[r].galaxy_A.xcom)

#save data
if args.orbit is not None:
    cmf.utils.save_data(res_dict, '{}.pickle'.format(args.orbit))
else:
    cmf.utils.save_data(res_dict, 'general-res-test.pickle')
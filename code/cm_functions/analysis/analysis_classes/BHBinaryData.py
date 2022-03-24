import datetime
import warnings
import pygad


__all__ = ["date_str", "BHBinaryData"]

date_str = "%Y-%m-%d %H:%M:%S"

class BHBinaryData:
    def __init__(self) -> None:
        """
        A class that defines the fields which constitute the variables of 
        interest for the BH Binary. These properties are accessible to all 
        child classes, and also correspond to the fields which are loadable 
        from a hdf5 file. 
        """
        self._log = ""
    
    @property
    def snaplist(self):
        return self._snaplist
    
    @snaplist.setter
    def snaplist(self, v):
        bad_snaps = []
        for i, vi in enumerate(v):
            try:
                s = pygad.Snapshot(vi, physical=True)
                s.delete_blocks()
            except KeyError:
                msg = "Snapshot {} potentially corrupt. Removing it from the list of snapshots for further analysis!".format(vi)
                warnings.warn(msg)
                self.add_to_log(msg)
                bad_snaps.append(vi)
        self._snaplist = [x for x in v if x not in bad_snaps]

    @property
    def bh_masses(self):
        return self._bh_masses
    
    @bh_masses.setter
    def bh_masses(self, v):
        self._bh_masses = v
    
    @property
    def binary_formation_time(self):
        return self._binary_formation_time
    
    @binary_formation_time.setter
    def binary_formation_time(self, v):
        self._binary_formation_time = v
    
    @property
    def binary_lifetime_timescale(self):
        return self._binary_lifetime_timescale
    
    @binary_lifetime_timescale.setter
    def binary_lifetime_timescale(self, v):
        self._binary_lifetime_timescale = v
    
    @property
    def r_infl(self):
        return self._r_infl
    
    @r_infl.setter
    def r_infl(self, v):
        self._r_infl = v
    
    @property
    def r_infl_time(self):
        return self._r_infl_time
    
    @r_infl_time.setter
    def r_infl_time(self, v):
        self._r_infl_time = v
    
    @property
    def r_infl_time_idx(self):
        return self._r_infl_time_idx
    
    @r_infl_time_idx.setter
    def r_infl_time_idx(self, v):
        self._r_infl_time_idx = v
    
    @property
    def r_infl_ecc(self):
        return self._r_infl_ecc
    
    @r_infl_ecc.setter
    def r_infl_ecc(self, v):
        self._r_infl_ecc = v
    
    @property
    def r_bound(self):
        return self._r_bound
    
    @r_bound.setter
    def r_bound(self, v):
        self._r_bound = v
    
    @property
    def r_bound_time(self):
        return self._r_bound_time
    
    @r_bound_time.setter
    def r_bound_time(self, v):
        self._r_bound_time = v
    
    @property
    def r_bound_time_idx(self):
        return self._r_bound_time_idx
    
    @r_bound_time_idx.setter
    def r_bound_time_idx(self, v):
        self._r_bound_time_idx = v
    
    @property
    def r_bound_ecc(self):
        return self._r_bound_ecc
    
    @r_bound_ecc.setter
    def r_bound_ecc(self, v):
        self._r_bound_ecc = v

    @property
    def r_hard(self):
        return self._r_hard
    
    @r_hard.setter
    def r_hard(self, v):
        self._r_hard = v
    
    @property
    def r_hard_time(self):
        return self._r_hard_time
    
    @r_hard_time.setter
    def r_hard_time(self, v):
        self._r_hard_time = v
    
    @property
    def r_hard_time_idx(self):
        return self._r_hard_time_idx
    
    @r_hard_time_idx.setter
    def r_hard_time_idx(self, v):
        self._r_hard_time_idx = v
    
    @property
    def r_hard_ecc(self):
        return self._r_hard_ecc
    
    @r_hard_ecc.setter
    def r_hard_ecc(self, v):
        self._r_hard_ecc = v
    
    @property
    def a_more_Xpc_idx(self):
        return self._a_more_Xpc_idx
    
    @a_more_Xpc_idx.setter
    def a_more_Xpc_idx(self, v):
        self._a_more_Xpc_idx = v
    
    @property
    def analytical_tspan(self):
        return self._analytical_tspan
    
    @analytical_tspan.setter
    def analytical_tspan(self, v):
        self._analytical_tspan = v
    
    @property
    def G_rho_per_sigma(self):
        return self._G_rho_per_sigma
    
    @G_rho_per_sigma.setter
    def G_rho_per_sigma(self, v):
        self._G_rho_per_sigma = v

    @property
    def H(self):
        return self._H
    
    @H.setter
    def H(self, v):
        self._H = v
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, v):
        self._K = v
    
    @property
    def gw_dominant_semimajoraxis(self):
        return self._gw_dominant_semimajoraxis
    
    @gw_dominant_semimajoraxis.setter
    def gw_dominant_semimajoraxis(self, v):
        self._gw_dominant_semimajoraxis = v
    
    @property
    def param_estimate_e_quantiles(self):
        return self._param_estimate_e_quantiles
    
    @param_estimate_e_quantiles.setter
    def param_estimate_e_quantiles(self, v):
        assert len(v) == 3, "An upper, middle, and lower quantile must be specified"
        v.sort()
        self._param_estimate_e_quantiles = v
    
    @property
    def predicted_orbital_params(self):
        return self._predicted_orbital_params
    
    @predicted_orbital_params.setter
    def predicted_orbital_params(self, v):
        self._predicted_orbital_params = v
    
    @property
    def formation_ecc_spread(self):
        return self._formation_ecc_spread
    
    @formation_ecc_spread.setter
    def formation_ecc_spread(self, v):
        self._formation_ecc_spread = v
    
    @property
    def binary_merger_remnant(self):
        return self._binary_merger_remnant
    
    @binary_merger_remnant.setter
    def binary_merger_remnant(self, v):
        self._binary_merger_remnant = v
    
    @property
    def binary_spin_flip(self):
        return self._binary_spin_flip
    
    @binary_spin_flip.setter
    def binary_spin_flip(self, v):
        self._binary_spin_flip = v

    def add_to_log(self, msg):
        #add a message to the log
        now = datetime.datetime.now()
        now = now.strftime(date_str)
        self._log += (now+": "+msg+"\n")


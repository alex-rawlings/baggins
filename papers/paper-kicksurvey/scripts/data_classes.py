import numpy as np

__all__ = ["RecoilCluster", "RecoilClusterSeries"]


class RecoilCluster:
    def __init__(self):
        """
        Class to hold information about the stars in a recoil cluster.
        """
        self.time = None
        self.bh_rad = None
        self.LOS_properties = dict(vel_disp=None, rhalf=None)
        self.intrinsic_properties = dict(vel_disp=None, rhalf=None, bound_mass=None)
        self.ids = None
        self.ambient_vel_disp = None
        self.kick_vel = None
        self.particle_masses = dict(bh=None, stars=None)
        self.snap_num = None


class RecoilClusterSeries:
    def __init__(self, *clusters):
        """
        Class to determine properties of related clusters (i.e. a time-series).

        Parameters
        ----------
        clusters : RecoilCluster
            clusters to add to the series
        """
        self.clusters = clusters
        kick_vels = [c.kick_vel for c in self.clusters]
        assert np.all(np.abs(np.diff(kick_vels)) < 1e-10)
        self.kick_vel = float(kick_vels[0])

    @property
    def bh_radii(self):
        """
        Displacement of BHs for each cluster

        Returns
        -------
        : list
            BH displacements
        """
        return [c.bh_rad for c in self.clusters]

    @property
    def max_rad(self):
        """
        Determine the maximum BH displacement

        Returns
        -------
        : float
            maximum radial displacement
        """
        return max(self.bh_radii)

    @property
    def apo(self):
        """
        Determine the cluster corresponding to apocentre.

        Returns
        -------
        : RecoilCluster
            apocentre cluster
        """
        return self.clusters[np.argmax(self.bh_radii)]

    @property
    def peri(self):
        """
        Determine the cluster corresponding to pericentre.

        Returns
        -------
        : RecoilCluster
            pericentre cluster
        """
        return self.clusters[np.argmin(self.bh_radii)]

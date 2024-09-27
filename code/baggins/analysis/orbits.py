import numpy as np
import gadgetorbits as go
from ..env_config import _cmlogger
from ..mathematics import get_histogram_bin_centres

__all__ = ["MergeMask", "OrbitClassifier"]

_logger = _cmlogger.getChild(__name__)

"""
The class IDs are a bit opaque, but can be found in orbit_classifier.hpp
By default:

ClassID | Family
------------------
        0 | pi-box
        1 | boxlet
        2 | x-tube
        3 | z-tube
        4 | rosette
        5 | irreg.
        6 | unclass.
"""


class MergeMask:
    def __init__(self) -> None:
        """
        Mask to group orbital families together into broader classifications.
        """
        self.families = []
        self.labels = []
        self._mask = np.full(27, -99, dtype=int)
        self._num_fams = -1

    @property
    def mask(self):
        try:
            assert np.all(self._mask) >= 0
        except AssertionError:
            _logger.warning("Not all members of the merge mask have been assigned a family!")
        return self._mask

    @property
    def num_fams(self):
        return self._num_fams


    def add_family(self, fam, idxs, label):
        """
        Add a family to the merge mask.

        Parameters
        ----------
        fam : str
            identifier for the family
        idxs : int, list-like
            which indices belong to this family
        label : str
            label for latex rendering for this family
        """
        self._num_fams = self._num_fams + 1
        self.families.append(fam)
        self.labels.append(label)
        self._mask[idxs] = self._num_fams


    def get_family(self, fam):
        """
        Determine the numeric identifier for a family.

        Parameters
        ----------
        fam : str
            family identifer

        Returns
        -------
        int
            numeric identifier for the family
        """
        try:
            return self.families.index(fam)
        except ValueError:
            _logger.exception(f"Family {fam} is not present for this mask! Available familes are {self.families}", exc_info=True)
            raise


    @classmethod
    def make_default_mask(cls):
        """
        Create the default merge mask which has:
        - pi-box
        - boxlet
        - x-tube
        - z-tube
        - rosette
        - irregular
        - unclassified

        Returns
        -------
        C : MergeMask
            class instance
        """
        C = cls()
        C.add_family("pi-box", [21, 24, 25], r"$\pi\mathrm{-box}$")
        C.add_family("boxlet", [1, 5, 9, 13, 17], r"$\mathrm{boxlet}$")
        C.add_family("x-tube", [3, 4, 7, 8, 11, 12, 15, 16, 19, 20], r"$x\mathrm{-tube}$")
        C.add_family("z-tube", [2, 6, 10, 14, 18], r"$z\mathrm{-tube}$")
        C.add_family("rosette", [26], r"$\mathrm{rosette}$")
        C.add_family("irreg", [22], r"$\mathrm{irregular}$")
        C.add_family("unclass", [0, 23], r"$\mathrm{unclassified}$")
        return C


    @classmethod
    def make_box_tube_mask(cls):
        """
        Create the merge mask which divides orbits into either boxes or tubes.

        Returns
        -------
        C : MergeMask
            class instance
        """
        C = cls()
        C.add_family("box", [1,5,9,13,17,21,24,25], r"$\mathrm{box}$")
        C.add_family("tube", [4,8,12,16,20,3,7,11,15,19,2,6,10,14,18,26], r"$\mathrm{tube}$")
        return C


class OrbitClassifier:
    def __init__(self, orbitcl, mergemask=None) -> None:
        """
        Class for orbit classification routines.

        Parameters
        ----------
        orbitcl : str, path-like
            orbit classification file
        mergemask : MergeMask, optional
            how sub-groups of orbital families should be merged, by default None
        """
        self.orbitcl = orbitcl
        if mergemask is None:
            _logger.info("Using the default merge mask for orbit families")
            self.mergemask = MergeMask.make_default_mask()
        else:
            self.mergemask = mergemask
        (
            self.particleids,
            self.classids,
            self.rad,
            self.rot_dir,
            self.energy,
            self.denergy,
            self.inttime,
            self.b92class,
            self.pericenter,
            self.apocenter,
            self.meanposrad,
            self.minangmom,
        ) = go.loadorbits(orbitcl, mergemask=self.mergemask.mask, addextrainfo=True)
        self.particleids = np.array(self.particleids, dtype=np.uint32)
        self.meanrads = None
        self.classfrequency = None
        self.radbincount = None


    def radial_frequency(self, radbins=None, supress_warnings=False):
        """
        Determine frequency of orbital classes in radial bins. Sets the 
        attributes 'meanrads', 'classfrequency', 'radbincount'.

        Parameters
        ----------
        radbins : array-like, optional
            radial bin edges, by default None
        """
        if radbins is None:
            radbins = np.geomspace(0.2, 30, 11)
        radbins = np.atleast_1d(radbins)
        nbin = len(radbins) - 1
        radbins = np.asarray(radbins)
        meanrads = get_histogram_bin_centres(radbins)
        possibleclasses = np.arange(np.max(self.classids) + 1).astype(int)
        pc_bins = np.arange(np.max(possibleclasses) + 1.5) - 0.5
        classfrequency = np.zeros((nbin, len(possibleclasses)))

        radbincount, _ = np.histogram(self.rad, bins=radbins)
        bin_ids = np.digitize(self.rad, bins=radbins) - 1
        bin_id_list = np.arange(np.max(bin_ids) + 1)
        if np.max(self.rad) > np.max(radbins):
            # ensure we are only consider particles within the maximum radius
            bin_id_list = bin_id_list[:-1]

        for bi in bin_id_list:
            try:
                # can't use bincount as we have some negative IDs
                classfrequency[bi, :] = np.histogram(self.classids[bin_ids==bi], bins=pc_bins)[0] / radbincount[bi]
            except IndexError:
                if not supress_warnings:
                    _logger.warning(f"Particle found beyond maximal radial edge of {np.nanmax(radbins):.1e}: skipping")
                continue
        self.meanrads = meanrads
        self.classfrequency = classfrequency
        self.radbincount = radbincount


    def box_tube_ratio(self, radbins=None, box_names=["pi-box", "boxlet"], tube_names=["x-tube", "z-tube", "rosette"]):
        """
        Determine the ratio of box orbits to tube orbits in radial bins.

        Parameters
        ----------
        radbins : array-like, optional
            radial bin edges, by default None
        box_names : list, optional
            MergeMask families to be classed as boxes, by default ["pi-box", 
            "boxlet"]
        tube_names : list, optional
            MergeMask families to be classed as tubes, by default ["x-tube", 
            "z-tube", "rosette"]

        Returns
        -------
        ratio : array-like
            ratio for each radial bin
        """
        self.radial_frequency(radbins=radbins)
        # get the numeric indentifiers for the classes
        box_class_ids = [self.mergemask.get_family(n) for n in box_names]
        tube_class_ids = [self.mergemask.get_family(n) for n in tube_names]
        ratio = np.nansum(np.einsum("ij,i->ij", self.classfrequency[:, box_class_ids], self.radbincount), axis=-1) / np.nansum(np.einsum("ij,i->ij", self.classfrequency[:, tube_class_ids], self.radbincount), axis=-1)
        return ratio


    def make_class_mask(self, fam):
        """
        Return a mask that can be used on properties to get the subset for a 
        family.

        Parameters
        ----------
        fam : str
            family name

        Returns
        -------
        : np.array
            boolean mask corresponding to family
        """
        return self.classids == self.mergemask.get_family(fam)


    def get_particle_ids_for_family(self, fam):
        """
        Return the particle IDs corresponding to an orbital class.

        Parameters
        ----------
        fam : str
            family name

        Returns
        -------
        : np.array
            particle IDs
        """
        return self.particleids[self.make_class_mask(fam)]


    def family_size_in_radius(self, fam, r):
        self.radial_frequency(radbins=np.array([0, r]))
        class_count = np.einsum("ij,i->ij", self.classfrequency, self.radbincount)
        return class_count[0, self.mergemask.get_family(fam)]


    def compare_class_change(self, other, fam, other_is_earlier=True):
        """
        Compare the change of particles of one class to what they were at a different classification time.
        The notion of whether or not the particles "were" a different type or 
        "will become" a different type is left to the discretion of the user, 
        depending on if the 'other' parameter is a past or future time, 
        respectively.

        Parameters
        ----------
        other : OrbitClassifier
            other classifier object to compare to
        fam : str
            the family in 'this' classifier to determine the change for
        other_is_earlier : bool, optional
            if 'other' corresponds to an earlier snapshot than this instance of 
            OrbitClassifier (for printing messages), by default True

        Returns
        -------
        other_classids : np.array
            class IDs of particles in the 'other' classifier
        """
        # check that we have a consistent merge mask
        try:
            assert self.mergemask.num_fams == other.mergemask.num_fams
            assert len(set(self.mergemask.families).difference(set(other.mergemask.families))) == 0
        except AssertionError:
            _logger.exception("Merge masks are not consistent for orbit comparison!", exc_info=True)
            raise
        mask = np.isin(other.particleids, self.get_particle_ids_for_family(fam))
        other_classids = other.classids[mask]
        N = np.sum(mask)
        if other_is_earlier:
            print(f"Families that became {fam}:")
        else:
            print(f"{fam} will become:")
        count_check = 0
        for i, c in enumerate(self.mergemask.families):
            n = np.sum(other_classids==i)
            count_check += n
            N2 = np.sum(other.classids==i)
            print(f"  {c}: {n} ({n/N*100:.2e}% of 'this' {fam}), ({n/N2*100:.2e}% of 'other' {c})")
        try:
            assert N == count_check
        except AssertionError:
            _logger.exception(f"Not all particles accounted for! Total should be {N}, but is {count_check}", exc_info=True)
            raise
        return other_classids

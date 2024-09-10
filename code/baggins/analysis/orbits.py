import numpy as np
import gadgetorbits as go
from ..env_config import _cmlogger
from ..mathematics import get_histogram_bin_centres

__all__ = ["MergeMask", "OrbitClassifier", "orbits_radial_frequency", "determine_box_tube_ratio"]

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
            self.orbitids,
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
        self.meanrads = None
        self.classfrequency = None
        self.radbincount = None


    def radial_frequency(self, radbins=None):
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

        for bi in np.arange(np.max(bin_ids) + 1):
            classfrequency[bi, :] = np.histogram(self.classids[bin_ids==bi], bins=pc_bins)[0] / radbincount[bi]
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



def orbits_radial_frequency(
    orbitcl, radbins=np.geomspace(0.2, 30, 11), returnextra=False, mergemask=None
):
    """
    Determine radial frequency of orbit families. The total fraction of orbits
    per radial bin sums to 1.

    Parameters
    ----------
    orbitcl : str, path-like
        orbit classification file
    radbins : array-like, optional
        radial bin edges, by default np.geomspace(0.2, 30, 11)
    returnextra : bool, optional
        return extra information, by default False
    mergemask : MergeMask, optional
        how sub-groups of orbital families should be merged, by default None

    Returns
    -------
    res : dict
        - centres of radial bins
        - frequency of each orbital class per bin
        - number of particles per bin
        - other properties if returnextra is True
    """
    _logger.info(f"Reading: {orbitcl}")
    if mergemask is None:
        _logger.info("Using the default merge mask for orbit families")
        mergemask = MergeMask.make_default_mask()
    (
        orbitids,
        classids,
        rad,
        rot_dir,
        energy,
        denergy,
        inttime,
        b92class,
        pericenter,
        apocenter,
        meanposrad,
        minangmom,
    ) = go.loadorbits(orbitcl, mergemask=mergemask.mask, addextrainfo=True)
    nbin = len(radbins) - 1
    radbins = np.asarray(radbins)
    meanrads = get_histogram_bin_centres(radbins)
    possibleclasses = np.arange(np.max(classids) + 1).astype(int)
    classfrequency = np.zeros((nbin, len(possibleclasses)))
    rad_len = []
    for i in np.arange(nbin) + 1:
        radcond = np.logical_and(radbins[i - 1] < rad, rad < radbins[i])
        radclassids = classids[radcond]
        rad_len.append(float(len(radclassids)))

        if rad_len[-1] > 0:
            for cl in possibleclasses:
                classfrequency[i - 1, cl] = (
                    float(len(radclassids[radclassids == cl])) / rad_len[-1]
                )
        else:
            _logger.debug("Warning: no particles in current radial bin")
            for cl in possibleclasses:
                classfrequency[i - 1, cl] = np.nan
    rad_len = np.array(rad_len)

    res = dict(meanrads=meanrads, classfrequency=classfrequency, rad_len=rad_len)
    if returnextra:
        res.update(
            dict(
                pid=orbitids,
                classids=classids,
                peri=pericenter,
                apo=apocenter,
                minangmom=minangmom,
                meanposrad=meanposrad,
                rad=rad,
            )
        )
    return res


def determine_box_tube_ratio(
    orbit_res,
    mergemask,
    within=None,
    box_class_names = ["pi-box", "boxlet"],
    tube_class_names = ["x-tube", "z-tube", "rosette"],
):
    """
    Determine the ratio of box orbits to tube orbits

    Parameters
    ----------
    orbit_res : dict
        output from orbits_radial_frequency()
    mergemask : MergeMask
        how orbit families are grouped
    within : float, optional
        determine ratio within some radius, by default None
    box_class_names : list, optional
        orbits to be classed as 'boxes', by default ["pi-box", "boxlet"]
    tube_class_names : list, optional
        orbits to be classed as 'tubes', by default ["x-tube", "z-tube", 
        "rosette"]

    Returns
    -------
    ratio : np.array
        box / tube ratio
    """
    # get the numeric indentifiers for the classes
    box_class_ids = [mergemask.get_family(n) for n in box_class_names]
    tube_class_ids = [mergemask.get_family(n) for n in tube_class_names]

    if within is not None:
        # allow determining ratio within a given radius
        try:
            mask = np.logical_and(orbit_res["meanrads"] > within[0], 
                                  orbit_res["meanrads"] <= within[1])
        except IndexError as e:
            _logger.exception(f"'within' must contain a lower and upper bound: {e}", exc_info=True)
            raise
        ratio = np.nansum(
            orbit_res["classfrequency"][mask][box_class_ids]
            ) / np.nansum(
                orbit_res["classfrequency"][mask][tube_class_ids]
            )
    else:
        # determine ratio per radial bin
        ratio = np.nansum(
            orbit_res["classfrequency"][:, box_class_ids] * orbit_res["rad_len"][box_class_ids], axis=1
            ) / np.nansum(
                orbit_res["classfrequency"][:, tube_class_ids] * orbit_res["rad_len"][tube_class_ids], axis=1
            )
    return np.atleast_1d(ratio)


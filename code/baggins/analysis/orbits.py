import numpy as np
import gadgetorbits as go
from ..env_config import _cmlogger
from ..mathematics import get_histogram_bin_centres

__all__ = ["MergeMask", "orbits_radial_frequency", "determine_box_tube_ratio"]

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


def orbits_radial_frequency(
    orbitcl, minrad=0.2, maxrad=30.0, nbin=10, returnextra=False, mergemask=None
):
    """
    Determine radial frequency of orbit families. The total fraction of orbits
    per radial bin sums to 1.

    Parameters
    ----------
    orbitcl : str, path-like
        orbit classification file
    minrad : float, optional
        minimum radial bin edge, by default 0.2
    maxrad : float, optional
        maximum radial bin edge, by default 30.0
    nbin : int, optional
        number of radial bins, by default 10
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
        - other properties of returnextra is True
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
    radbins = np.geomspace(minrad, maxrad, nbin + 1)
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
    meanrads,
    classfrequency,
    rad_len,
    within,
    box_class_ids=[0, 1],
    tube_class_ids=[2, 3, 4],
):
    """
    Calculate the ratio of box to tube orbits within some radii.

    Parameters
    ----------
    meanrads : array-like
        centres of radial bins
    classfrequency : array-like
        frequency of each orbital class for each radial bin
    rad_len : array-like
        number of particles per radial bin
    within : float
        radius to determine fraction within
    box_class_ids : list, optional
        orbital class IDs to be listed as boxes, by default [0, 1]
    tube_class_ids : list, optional
        orbital class IDs to be listed as tubes, by default [2,3,4]

    Returns
    -------
    : float
        ratio
    """
    mask = meanrads <= within
    boxes = 0
    tubes = 0
    for cid in box_class_ids:
        boxes += np.nansum(classfrequency[mask, cid] * rad_len[mask])
    for cid in tube_class_ids:
        tubes += np.nansum(classfrequency[mask, cid] * rad_len[mask])
    return boxes / tubes

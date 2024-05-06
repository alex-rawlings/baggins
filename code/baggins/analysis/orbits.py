import numpy as np
import gadgetorbits as go
from ..env_config import _cmlogger
from ..mathematics import get_histogram_bin_centres

__all__ = ["radial_frequency"]

_logger = _cmlogger.getChild(__name__)

# merge orbit classes into more general groups
# this is given to gadgetorbits.simplifyorbits
mergemask = [
    6,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    1,
    3,
    2,
    2,
    0,
    5,
    6,
    0,
    0,
    4,
]


def radial_frequency(
    orbitcl, minrad=0.2, maxrad=30.0, nbin=10, returnextra=False, mergemask=mergemask
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
    mergemask : array-like, optional
        how sub-groups of orbital families should be merged, by default
        mergemask (local variable defined above)

    Returns
    -------
    meanrads : array-like
        radial bin centres
    classfrequency : array-like
        frequency of each orbital class per radial bin
    rad_len : array-like
        number of stellar particles per radial bin
    """
    _logger.info(f"Reading: {orbitcl}")
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
    ) = go.loadorbits(orbitcl, mergemask=mergemask, addextrainfo=True)
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
    rad_len = np.array(rad_len)

    if returnextra:
        return (
            meanrads,
            classfrequency,
            rad_len,
            classids,
            pericenter,
            apocenter,
            minangmom,
        )
    else:
        return meanrads, classfrequency, rad_len

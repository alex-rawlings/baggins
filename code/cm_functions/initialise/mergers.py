__all__ = ['e_from_rperi']


def e_from_rperi(x, a=0.320, b=1.629, c=0.176):
    """
    Determine eccentricity from r/Rvir using fit to Khochfar & Burkett 2006
    Fig. 6

    Parameters
    ----------
    x: normalised rperi values (normalised to the virial radius of the larger
        progenitor)
    a, b, c = shape parameters

    Returns
    -------
    eccentricity of approach
    """
    return (1 + (x/a)**b)**(-c)

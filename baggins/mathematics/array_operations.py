import numpy as np

__all__ = ["get_histogram_bin_centres", "assert_all_unique", "get_pixel_value_in_image"]


def get_histogram_bin_centres(bins):
    """
    Convenience function to get the centres of some histogram bins.

    Parameters
    ----------
    bins : np.ndarray
        bin edges

    Returns
    -------
    : np.ndarray
        bin centres, has len = len(bins)-1
    """
    return (bins[:-1] + bins[1:]) / 2.0


def assert_all_unique(a, axis=None):
    """
    Assert all elements in an array-like list are unique.

    Parameters
    ----------
    a : array-like
        array to determine uniqueness of
    axis : int
        array axis to determine uniqueness over, by default None

    Returns
    -------
    bool
        True if all elements in a are unique
    """
    seen = set()
    if axis is None:
        return not any(i in seen or seen.add(i) for i in a)
    else:
        if axis == -1:
            axis = len(a.shape) - 1
        if axis == 0:
            for j in range(a.shape[axis]):
                res = not any(i in seen or seen.add(i) for i in a[j, :])
                if not res:
                    break
        else:
            for j in range(a.shape[axis]):
                res = not any(i in seen or seen.add(i) for i in a[:, j])
                if not res:
                    break
        return res


def get_pixel_value_in_image(x, y, im):
    """
    Determine the pixel value for a given (x,y) coordinate in the array returned from pyplot's imshow()

    Parameters
    ----------
    x : float
        x coordinate
    y : float
        y coordinate
    im : pyplot.AxesImage
        returned object from pyplot.imshow() call

    Returns
    -------
    : float
        pixel value for the desired coordinates
    row : int
        row index
    col : int
        column index
    """
    xmin, xmax, ymin, ymax = im.get_extent()
    nr, nc = im.get_array().shape
    col = np.clip(((x - xmin) / (xmax - xmin) * nc).astype(int), 0, nc - 1)
    row = np.clip(((y - ymin) / (ymax - ymin) * nr).astype(int), 0, nr - 1)
    return im.get_array()[row, col], row, col

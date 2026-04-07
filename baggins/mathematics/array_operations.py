import numpy as np

__all__ = [
    "get_histogram_bin_centres",
    "equal_count_bins",
    "dual_region_equal_bins",
    "assert_all_unique",
    "get_pixel_value_in_image",
]


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


def equal_count_bins(x, N_per_bin):
    """
    Create histogram bins with equal counts in each bin

    Parameters
    ----------
    x : array-like
        values to bin
    N_per_bin : int
        number of counts per bin

    Returns
    -------
    : array-like
        bin edges
    """
    return np.quantile(x, np.linspace(0, 1, int(len(x) / N_per_bin) + 1))


def dual_region_equal_bins(x, pivot, N_per_bin_inner, N_per_bin_outer):
    """
    Partition the array into two regions which have their own unique equal-particle-count binning strategy. The separation between the regions (pivot) specifies a value in the data space.

    Parameters
    ----------
    x : array-like
         values to bin
    pivot : float
        boundary between the two regions
    N_per_bin_inner : int
        number of counts per bin in the inner region
    N_per_bin_outer : int
        number of counts per bin in the outer region

    Returns
    -------
    edges : np.ndarray
        combined bin edges for the inner and outer regions, with a shared boundary at the pivot
    """
    x = np.asarray(x)

    # ---- split ----
    inner_mask = x <= pivot
    inner_x = x[inner_mask]
    outer_x = x[~inner_mask]

    # ---------- INNER REGION ----------
    n_inner = len(inner_x)

    if n_inner == 0:
        inner_edges = np.array([])
        boundary = None
    else:
        remainder = n_inner % N_per_bin_inner

        if remainder != 0 and len(outer_x) > 0:
            needed = N_per_bin_inner - remainder
            take = min(needed, len(outer_x))

            idx = np.argpartition(outer_x, take - 1)[:take]

            inner_x = np.concatenate([inner_x, outer_x[idx]])

            mask = np.ones(len(outer_x), dtype=bool)
            mask[idx] = False
            outer_x = outer_x[mask]

            n_inner = len(inner_x)

        n_bins_inner = n_inner // N_per_bin_inner

        k_inner = np.arange(0, n_bins_inner + 1) * N_per_bin_inner
        part_inner = np.partition(inner_x, k_inner[:-1])

        inner_edges = np.concatenate(
            [[part_inner[0]], part_inner[k_inner[1:-1]], [part_inner.max()]]
        )

        boundary = inner_edges[-1]

    # ---------- OUTER REGION ----------
    if boundary is None:
        # fallback: no inner region
        return np.array([])

    # only keep values >= boundary
    outer_x = outer_x[outer_x >= boundary]
    n_outer = len(outer_x)

    if n_outer == 0:
        return inner_edges

    # enforce full bins
    remainder = n_outer % N_per_bin_outer
    if remainder != 0:
        keep = n_outer - remainder
        if keep > 0:
            idx = np.argpartition(outer_x, keep - 1)[:keep]
            outer_x = outer_x[idx]
        else:
            outer_x = np.array([])

    n_outer = len(outer_x)

    if n_outer == 0:
        return inner_edges

    n_bins_outer = n_outer // N_per_bin_outer

    k_outer = np.arange(0, n_bins_outer + 1) * N_per_bin_outer
    part_outer = np.partition(outer_x, k_outer[:-1])

    outer_edges = np.concatenate(
        [
            [boundary],  # <-- enforce shared boundary
            part_outer[k_outer[1:-1]],
            [part_outer.max()],
        ]
    )

    # ---------- MERGE ----------
    edges = np.concatenate([inner_edges[:-1], outer_edges])

    return edges


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

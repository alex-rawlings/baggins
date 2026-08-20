import numpy as np

__all__ = [
    "get_histogram_bin_centres",
    "equal_count_bins",
    "dual_region_equal_bins",
    "radial_bins_by_count",
    "assert_all_unique",
    "get_pixel_value_in_image",
    "equal_tail_indices",
]


def get_histogram_bin_centres(bins, x=None):
    """
    Convenience function to get the centres of some histogram bins.

    Parameters
    ----------
    bins : np.ndarray
        bin edges
    x : np.ndarray, optional
        original data that was binned - if provided, the centre of a bin is the median value of the subset of elements in x which lie within the bin, by default None

    Returns
    -------
    : np.ndarray
        bin centres, has len = len(bins)-1
    """
    if x is None:
        return (bins[:-1] + bins[1:]) / 2.0
    else:
        centres = np.full(len(bins) - 1, np.nan)
        for i, (bi, bo) in enumerate(zip(bins[:-1], bins[1:])):
            mask = np.logical_and(x > bi, x <= bo)
            centres[i] = np.nanmedian(x[mask])
        return centres


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


def radial_bins_by_count(r, n_start=100, n_end=10000, n_bins=20, r_min=None):
    """
    Compute radial bin edges such that the number of particles per bin
    grows geometrically from n_start to n_end over n_bins bins, and then
    continues with constant-size bins of n_end particles for any
    remaining particles beyond that.

    Parameters
    ----------
    r : array_like
        Radii of all particles (any order).
    n_start : int
        Target number of particles in the innermost bin.
    n_end : int
        Target number of particles in the outermost geometric bin,
        and the fixed bin size used for all subsequent bins.
    n_bins : int
        Number of geometrically-spaced bins before switching to
        constant-size bins.
    r_min : float, optional
        Inner edge of the first bin. Defaults to the smallest radius
        in the data (use 0.0 if you want bins starting at the origin).

    Returns
    -------
    edges : ndarray, shape (n_bins_actual + 1,)
        The bin edges.
    counts : ndarray, shape (n_bins_actual,)
        The actual number of particles that fall in each bin.

    Notes
    -----
    Whichever bin ends up last (geometric or constant-size), if it is
    under-filled relative to its target count, it is merged into the
    previous bin instead of being kept as a short bin -- so the final
    edge always sits exactly at the outermost particle.
    """
    r = np.sort(np.asarray(r))
    N = r.size

    if r_min is None:
        r_min = r[0]

    # --- Geometric part -----------------------------------------------
    target_per_bin = np.geomspace(n_start, n_end, n_bins)
    cumul_counts = np.round(np.cumsum(target_per_bin))

    # --- Constant-size part (only if particles remain) -----------------
    if cumul_counts[-1] < N:
        n_remaining = N - cumul_counts[-1]
        n_extra_bins = int(np.ceil(n_remaining / n_end))
        extra_cumul = cumul_counts[-1] + n_end * np.arange(1, n_extra_bins + 1)
        cumul_counts = np.concatenate([cumul_counts, extra_cumul])

    # Clip to available particles and remove duplicate indices
    cumul_counts = np.clip(cumul_counts, 1, N)
    cumul_counts = np.unique(cumul_counts)

    # If the last bin is short of its target count (n_end), merge it
    # into the previous bin by dropping that intermediate edge.
    if len(cumul_counts) >= 2:
        last_bin_count = cumul_counts[-1] - cumul_counts[-2]
        if last_bin_count < n_end:
            cumul_counts = np.delete(cumul_counts, -2)

    # Make sure the final edge reaches all the way to the last particle
    if cumul_counts[-1] != N:
        cumul_counts[-1] = N

    # Bin edges: r_min, then the radius at each cumulative count
    cumul_counts = cumul_counts.astype(int)
    edge_radii = r[cumul_counts - 1]
    edges = np.concatenate(([r_min], edge_radii))

    counts = np.diff(cumul_counts, prepend=0)

    return edges, counts


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


def equal_tail_indices(a, b, rtol=1e-9, atol=1e-12):
    """
    For two arrays, find where (from the end) the arrays are equal. The arrays may be of differeing lengths. The return is the index in each array from which point onwards the arrays remain equal.

    Parameters
    ----------
    a : np.array
        array 1
    b : np.array
        array 2
    rtol : float, optional
        relative tolerance, by default 1e-9
    atol : float, optional
        absolute tolerance, by default 1e-12

    Returns
    -------
    start_a : int
        starting index for equal parts
    b_tail : int
        starting index for equal parts
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = min(len(a), len(b))
    if n == 0:
        return len(a), len(b)

    # Compare the last n elements of each, aligned from the end
    a_tail = a[-n:]
    b_tail = b[-n:]

    mismatches = np.flatnonzero(~np.isclose(a_tail, b_tail, rtol=rtol, atol=atol))
    offset_from_end = (mismatches[-1] + 1) if mismatches.size else 0

    # Convert "offset from the end of the tail" into an absolute index
    # in each original array
    start_a = len(a) - n + offset_from_end
    start_b = len(b) - n + offset_from_end

    return start_a, start_b

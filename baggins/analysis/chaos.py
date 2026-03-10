from tqdm import tqdm
import numpy as np
import pygad
from baggins.env_config import _cmlogger

__all__ = ["lyapunov_estimator"]

_logger = _cmlogger.getChild(__name__)


def lyapunov_estimator(snaplist1, snaplist2):
    """
    Estimate the Lyapunov timescale for chaotic divergence between two series of snapshots. All particles in the snapshots are used, provided a particle with a given ID exists in both series.

    Parameters
    ----------
    snaplist1 : list
        series of snapshots
    snaplist2 : list
        series of snapshots

    Returns
    -------
    : float
        Lyapunov exponent
    t : list
        times of snapshots
    distances : list
        distance metric for each time
    """
    Nsnap = min(len(snaplist1), len(snaplist2))

    # --- pass 1: estimate global variances ---
    sum_r2 = sum_v2 = 0.0
    p_count = 0

    def _move_to_centre(s):
        x = pygad.analysis.center_of_mass(s)
        v = pygad.analysis.mass_weighted_mean(s, "vel")
        pygad.Translation(-x).apply(s)
        pygad.Boost(-v).apply(s)

    for i, (f1, f2) in tqdm(
        enumerate(zip(snaplist1, snaplist2)), total=Nsnap, desc="Estimating variance"
    ):
        snap1 = pygad.Snapshot(f1, physical=True)
        snap2 = pygad.Snapshot(f2, physical=True)
        _move_to_centre(snap1)
        _move_to_centre(snap2)
        try:
            assert np.abs(snap1.time - snap2.time) < 1e-6
        except AssertionError:
            _logger.exception(
                f"Snapshots {i} are at inconsistent times!", exc_info=True
            )
            raise
        mask = pygad.IDMask(snap1["ID"]) & pygad.IDMask(snap2["ID"])
        r1, v1 = snap1[mask]["pos"], snap1[mask]["vel"]
        r2, v2 = snap2[mask]["pos"], snap2[mask]["vel"]
        sum_r2 += np.sum(r1 * r1) + np.sum(r2 * r2)
        sum_v2 += np.sum(v1 * v1) + np.sum(v2 * v2)
        p_count += r1.size + r2.size
        snap1.delete_blocks()
        snap2.delete_blocks()
        pygad.gc_full_collect()

    var_r = sum_r2 / p_count
    var_v = sum_v2 / p_count

    # --- pass 2: compute separations ---
    log_sum = 0.0
    t = np.full(Nsnap, np.nan)
    distances = np.full_like(t, np.nan)

    for i, (f1, f2) in tqdm(
        enumerate(zip(snaplist1, snaplist2)), total=Nsnap, desc="Calculating divergence"
    ):
        snap1 = pygad.Snapshot(f1, physical=True)
        snap2 = pygad.Snapshot(f2, physical=True)
        t[i] = snap1.time
        mask = pygad.IDMask(snap1["ID"]) & pygad.IDMask(snap2["ID"])

        dr = snap1[mask]["pos"] - snap2[mask]["pos"]
        dv = snap1[mask]["vel"] - snap2[mask]["vel"]
        distances[i] = np.sqrt(np.sum(dr * dr) / var_r + np.sum(dv * dv) / var_v)
        if distances[i] > 0:
            log_sum += np.log(distances[i])

        snap1.delete_blocks()
        snap2.delete_blocks()
        pygad.gc_full_collect()

    if i == 0:
        return np.nan, t, distances
    return log_sum / (t[-1] - t[0]), t, distances

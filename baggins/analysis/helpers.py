from tqdm import tqdm
from operator import itemgetter
from copy import copy
import pygad
from baggins.analysis.analyse_snap import basic_snapshot_centring
from baggins.general.pygad_helper import convert_gadget_time
from baggins.utils import get_snapshots_in_dir
from baggins.env_config import _cmlogger

__all__ = ["SnapshotIterator"]

_logger = _cmlogger.getChild(__name__)


class SnapshotIterator:
    def __init__(self, snapdir, centre="basic", stride=None, **kwargs):
        """
        Class to iterate over a series of snapshots in a directory. Snapshots may be centred, and cleaned up after control returns to the generator.

        Parameters
        ----------
        snapdir : str
            snapshot directory
        centre : str or None, optional
            centring of snapshot, by default "basic" (can be "bh_com" or None)
        stride : int, optional
            take every ith snapshot, by default None
        kwargs :
            other arguments for get_snapshots_in_dir()
        """
        self.snapdir = snapdir
        self.snapfiles = get_snapshots_in_dir(path=self.snapdir, **kwargs)
        if stride is not None:
            self.snapfiles = self.snapfiles[::stride]
        try:
            assert centre in ("basic", "bh_com") or centre is None
            self.centre = centre
        except AssertionError:
            _logger.exception(f"Invalid centering method {centre}", exc_info=True)

    @property
    def len(self):
        return len(self.snapfiles)

    def limit_to_snaps(self, *args):
        """
        Limit the snapshot generator to the specified snapshots in the directory.
        """
        self.snapfiles = itemgetter(*args)(self.snapfiles)
        if len(args) == 1:
            self.snapfiles = [self.snapfiles]
        else:
            self.snapfiles = list(self.snapfiles)

    def make_generator(self, hide_prog=False):
        """
        Make the snapshot generator.

        Parameters
        ----------
        hide_prog : bool, optional
            hide tqdm progress bar, by default False

        Yields
        ------
        i : int
            iterator number
        t : float
            snapshot time
        snap : pyad.Snapshot
            (centred) snapshot
        """
        if self.centre == "basic":

            def centring(s):
                basic_snapshot_centring(s)
        elif self.centre == "bh_com":

            def centring(s):
                xcom = pygad.analysis.center_of_mass(s.bh)
                vcom = pygad.analysis.mass_weighted_mean(s.bh, "vel")
                pygad.Translation(-xcom).apply(s, total=True)
                pygad.Boost(-vcom).apply(s, total=True)
        elif self.centre is None:

            def centring(s):
                pass

        for i, s in tqdm(
            enumerate(self.snapfiles),
            total=self.len,
            desc="Iterating snapshots",
            disable=hide_prog,
        ):
            # TODO: protect against instance of final snapshot being delta_t >> average
            snap = pygad.Snapshot(s, physical=True)
            centring(snap)
            t = convert_gadget_time(snap)
            yield i, t, snap

            # conserve memory
            snap.delete_blocks()
            del snap
            pygad.gc_full_collect()

    def get_min_max_times(self):
        """
        Get the minimum and maximum times of a series of snapshots.

        Returns
        -------
        ts : list
            minimum and maximum snapshot times, in Gyr
        """
        ts = [None, None]
        # ensure snapshots are in order, as this might be changed if limit_to_snaps() has been called with non-consecutive inputs
        sfiles = copy(self.snapfiles)
        sfiles.sort()
        for i, s in enumerate(itemgetter(0, -1)(sfiles)):
            snap = pygad.Snapshot(s, physical=True)
            ts[i] = convert_gadget_time(snap)
        return ts

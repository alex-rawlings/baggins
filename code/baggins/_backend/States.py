import os
from datetime import datetime

__all__ = ["PublishingState", "TmpDirRegister"]


class _State:
    """
    Simple object to store the state of a setting
    """

    def __init__(self) -> None:
        self._setting_set = False

    def turn_on(self):
        self._setting_set = True

    def turn_off(self):
        self._setting_set = False

    def is_set(self):
        return self._setting_set


class PublishingState(_State):
    def __init__(self) -> None:
        """
        Simple object to store the state of a plotting-style setting
        """
        super().__init__()


class TmpDirRegister:
    def __init__(self, basename) -> None:
        """
        Class to hold a register of temporary directories. This means that
        temporary directories are particular to the given invocation of
        cm_functions, and can be cleaned up at exit without affecting the
        temporary directories of other invocations running concurrently.

        Parameters
        ----------
        basename : str, path-like
            base temporary directory path to which a timestamp will be appended
        """
        self._basename = basename
        self._register = []

    @property
    def register(self):
        return self._register

    def make_new_dir(self):
        """
        Make a new temporary directory, and add it to the register
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self._basename}_{now}"
        i = 0
        N = 10
        while os.path.exists(dir_name) and i < N:
            dir_name = f"{dir_name}_{i}"
            i += 1
        assert i < N
        os.makedirs(dir_name)
        self._register.append(dir_name)

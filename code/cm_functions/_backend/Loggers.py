import logging
from datetime import datetime


__all__ = ["InternalLogger", "CustomLogger"]


class InternalLogger:
    def __init__(self, name, console_level="WARNING") -> None:
        """
        Class for logging information internal to the package. Two handlers are 
        supported: a console handler, and a file handler.

        Parameters
        ----------
        name : str
            name of logger
        console_level : str, optional
            logging level for console printing, by default "WARNING"
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel("DEBUG")
        self._ch_format = "%(name)s: %(levelname)s: %(message)s"
        self._fh_format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s"
        self._c_handler = logging.StreamHandler()
        self._set_handler(self._c_handler, console_level, fmt=self._ch_format)
        self._f_handler = None
    
    def _set_handler(self, handler, level, fmt):
        """
        Helper function to set a handler

        Parameters
        ----------
        handler : logging.Handler
            handler to add to logger
        level : str
            logging level of handler
        fmt : str
            format of logging message
        """
        handler.setLevel(getattr(logging, level))
        handler.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(handler)
    
    def _ensure_valid_level(self, v):
        """
        Helper function to ensure a valid logging level is parsed

        Parameters
        ----------
        v : str
            logging level

        Returns
        -------
        str
            logging level

        Raises
        ------
        ValueError
            if an invalid logging level is given
        """
        v = v.upper()
        if v not in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Logging level <{v}> invalid!")
        return v
    
    def add_file_handler(self, logfile, file_level="ERROR"):
        """
        Add a file handler to the logger

        Parameters
        ----------
        logfile : str, path-like
            file to print to
        file_level : str, optional
            logging level, by default "ERROR"
        """
        with open(logfile, "a") as f:
            f.write(f"\nLogger created at {datetime.now()}")
        self._f_handler = logging.FileHandler(filename=logfile)
        self._set_handler(self._f_handler, file_level, fmt=self._fh_format)


class CustomLogger(InternalLogger):
    def __init__(self, name, console_level="WARNING") -> None:
        """
        General purpose logger to be used in external scripts. The error levels
        can be updated, unlike the InternalLogger.

        Parameters
        ----------
        name : str
            name of logger
        console_level : str, optional
            logging level for console printing, by default "WARNING"
        """
        super().__init__(name, console_level)
        self.console_level = console_level
    
    def add_file_handler(self, logfile, file_level="ERROR"):
        self.file_level = file_level
        super().add_file_handler(logfile, self.file_level)
    
    @property
    def console_level(self):
        return self._console_level
    
    @console_level.setter
    def console_level(self, v):
        self._console_level = self._ensure_valid_level(v)
        if hasattr(self, "_c_handler"):
            self._c_handler.setLevel(self._console_level)
    
    @property
    def file_level(self):
        return self._file_level
    
    @file_level.setter
    def file_level(self, v):
        self._file_level = self._ensure_valid_level(v)
        if self._f_handler is not None:
            self._f_handler.setLevel(self._file_level)


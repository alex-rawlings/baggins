import logging

__all__ = ["setup_logger"]


class CustomLogger(logging.Logger):
    """
    Custom logger class to override some methods
    """
    def __init__(self, name: str, level = 0) -> None:
        super().__init__(name, level)


    def getChild(self, suffix: str) -> None:
        """
        Remove the leading cm_functions. part from a child name

        Parameters
        ----------
        suffix : str
            suffix to append to logger name

        Returns
        -------
        CustomLogger
            child logger
        """
        return super().getChild(suffix.replace("cm_functions.", ""))


    def setLevel(self, level) -> None:
        """
        Ensures a valid level is provided

        Parameters
        ----------
        level : str
            logging level

        Raises
        ------
        ValueError
            if invalid level given
        """
        level = level.upper()
        if level not in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Logging level <{level}> invalid!")
        super().setLevel(level)



def setup_logger(name, console_level="WARNING", logfile=None, file_level="WARNING", capture_warnings=True) -> logging.Logger:
    """
    Set up a logger instance

    Parameters
    ----------
    name : str
        name of logger
    console_level : str, optional
        logging level for console printing, by default "WARNING"
    logfile : path-like, optional
        file to print logs to, by default None
    file_level : str, optional
        logging level for file printing, by default "WARNING"
    capture_warnings : bool, optional
        redirect all warnings to the logger, by default True

    Returns
    -------
    logging.Logger
        logger
    """
    logging.setLoggerClass(CustomLogger)
    log = logging.getLogger(name)
    cfmt = logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s", "%H:%M:%S")
    ffmt = logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    # allow all messages to be captured, and then printed only if desired
    log.setLevel("DEBUG")
    # add the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(cfmt)
    log.addHandler(console_handler)

    # add a file handler if desired
    if logfile is not None:
        file_handler = logging.handlers.TimedRotatingFileHandler(logfile, when="D", interval=1, backupCount=10)
        file_handler.doRollover()
        file_handler.setLevel(file_level)
        file_handler.setFormatter(ffmt)
        log.addHandler(file_handler)
    logging.captureWarnings(capture_warnings)
    # prevent duplicate messages
    log.propagate = False
    return log

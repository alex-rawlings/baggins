import logging

# sometimes handlers submodule not loaded
from logging import handlers

__all__ = ["setup_logger"]


class CustomLogger(logging.Logger):
    """
    Custom logger class to override some methods
    """

    def __init__(self, name: str, level=0) -> None:
        super().__init__(name, level)

    def getChild(self, suffix: str) -> None:
        """
        Remove the leading baggins. part from a child name

        Parameters
        ----------
        suffix : str
            suffix to append to logger name

        Returns
        -------
        CustomLogger
            child logger
        """
        suffix = suffix.replace("baggins.", "")
        suffix = suffix.replace("analysis_classes.", "")
        return super().getChild(suffix)

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
        if isinstance(level, str):
            level = level.upper()
            if level not in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(f"Logging level <{level}> invalid!")
        super().setLevel(level)


def setup_logger(
    name,
    console_level="WARNING",
    logfile=None,
    file_level="WARNING",
    capture_warnings=True,
) -> CustomLogger:
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
    CustomLogger
        logger
    """
    logging.setLoggerClass(CustomLogger)
    log = logging.getLogger(name)
    cfmt = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s", "%H:%M:%S"
    )
    ffmt = logging.Formatter(
        "%(asctime)s: %(name)s: %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    # allow all messages to be captured, and then printed only if desired
    log.setLevel("DEBUG")
    # add the console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(cfmt)
    log.addHandler(console_handler)

    # add a file handler if desired
    if logfile is not None:
        file_handler = handlers.TimedRotatingFileHandler(
            logfile, when="midnight", interval=1, backupCount=10
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(ffmt)
        log.addHandler(file_handler)
    logging.captureWarnings(capture_warnings)
    # prevent duplicate messages
    log.propagate = False
    return log

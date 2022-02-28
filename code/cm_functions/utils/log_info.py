import logging
import contextlib
import sys

__all__ = ["OutputLogger"]

class OutputLogger:
    def __init__(self, name="my_log.log", level="INFO") -> None:
        """
        A wrapper class to redirect print() statements to a log file
        """
        logging.basicConfig(filename=name, format="%(asctime)s: %(levelname)s: %(message)s", datefmt="%Y/%m/%d %I:%M:%S %p", level=15)
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)
        self._redirector = contextlib.redirect_stdout(self)
    
    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)
    
    def flush(self):
        pass

    def __enter__(self):
        self._redirector.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._redirector.__exit__(exc_type, exc_value, traceback)

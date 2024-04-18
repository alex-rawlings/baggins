from .Logging import setup_logger

__all__ = ["setup_logger"]


# Top level classes and functions that are designed to only be used internally
# within the baggins package, and not called from external scripts.
# The .py files in this directory should not contain any references to other
# modules within the baggins package.

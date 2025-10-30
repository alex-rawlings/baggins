import warnings
import functools
import inspect

__all__ = ["ExperimentalAPIWarning", "experimental_api"]


class ExperimentalAPIWarning(UserWarning):
    """
    Warning raised when using an experimental API feature.
    """

    pass


def experimental_api(message=None):
    """
    Decorator to mark functions or classes as experimental (warn once per API).
    """
    warned_names = set()

    def decorator(obj):
        name = obj.__name__
        warning_message = f"The API '{name}' is experimental and may change or be removed in future releases."
        if message:
            warning_message += f" {message}"

        def issue_warning():
            if name not in warned_names:
                warned_names.add(name)
                # Dynamically estimate how deep we are in the call stack
                # to make the warning point to user code.
                frame_depth = len(inspect.stack())
                # Cap stacklevel to a reasonable value (avoid huge numbers)
                stacklevel = max(22, min(frame_depth - 1, 6))
                # Force the warning to be shown, even if filters suppress it
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        warning_message, ExperimentalAPIWarning, stacklevel=stacklevel
                    )

        if isinstance(obj, type):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                issue_warning()
                return orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj
        else:

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                issue_warning()
                return obj(*args, **kwargs)

            return wrapper

    return decorator

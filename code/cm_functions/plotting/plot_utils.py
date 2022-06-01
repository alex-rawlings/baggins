from datetime import datetime
import inspect
from matplotlib.pyplot import gcf
from PIL import Image
from ..env_config import git_hash, username, date_format

__all__ = ["savefig", "get_meta"]

def savefig(fname, fig=None, save_kwargs={}):
    """
    Wrapper arounf pyplot's savefig() that adds some more useful meta data

    Parameters
    ----------
    fname : str or path-like
        file name to save the figure as
    fig : matplotlib.Figure, optional
        figure object to save, by default None (gets current figure)
    save_kwargs : dict, optional
        optional kwargs to pass to pyplot.savefig(), by default {}
    """
    if fig is None:
        fig = gcf()
    f = inspect.stack()[-1] # get outermost caller on stack
    now = datetime.now()
    # ensure things are deterministic
    try:
        meta_data = dict(
                         user = username,
                         script = f.filename,
                         created = now.strftime(date_format),
                         git_hash = git_hash
        )
        fig.savefig(fname, metadata=meta_data, **save_kwargs)
    finally:
        del f


def get_meta(fname):
    """
    Return the meta data of an image
    TODO is this only working for png's or also other file types?

    Parameters
    ----------
    fname : str, path-like
        image to print meta data for
    """
    with Image.open(fname) as img:
        for k,v in img.text.items():
            print(f"{k}: {v}")

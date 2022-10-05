from datetime import datetime
import inspect
from matplotlib.pyplot import gcf
from PIL import Image
import os.path
from ..env_config import git_hash, username, date_format, fig_ext

__all__ = ["savefig", "get_meta"]

def savefig(fname, fig=None, save_kwargs={}, force_ext=False):
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
    force_ext : bool
        override config file figure extension, by default False
    """
    if fig is None:
        fig = gcf()
    f = inspect.stack()[-1] # get outermost caller on stack
    now = datetime.now()
    if fig_ext == "png" and not force_ext:
        now = now.strftime(date_format)
    # ensure things are deterministic
    try:
        meta_data = dict(
                         Author = username,
                         Creator = f.filename,
                         CreationDate = now,
                         Keywords = git_hash
        )
        # save to the correct format
        if force_ext:
            fig.savefig(fname, metadata=meta_data, **save_kwargs)
        else:
            fname_name, fname_ext = os.path.splitext(fname)
            # protect against cases where no extension is specified, and the 
            # file name has a "." in it
            _fname = fname_name if fname_ext in (".png", ".pdf") else fname
            fig.savefig(f"{_fname}.{fig_ext}", metadata=meta_data, **save_kwargs)
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

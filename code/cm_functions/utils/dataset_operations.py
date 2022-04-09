import numpy as np
import pandas as pd

__all__ = ["create_error_col", "add_time_column"]


def create_error_col(dat, col, splitat="+/-"):
    """
    Split a pandas dataframe column with an error into two 
    columns

    Parameters
    ----------
    dat : pandas.core.frame.DataFrame
        data frame to operate on
    col : str
        column to split
    splitat : str, optional
        character combination to split the entry at, by default "+/-"
    """
    val_err = np.array([d.split(splitat) for d in dat.loc[:,col]], dtype="float")
    dat.loc[:,col] = val_err[:,0]
    dat.insert(len(dat.columns), col+"_ERR", val_err[:,1])


def add_time_column(df, unit="s", colname="time", newcolname="time_s"):
    """
    Convert a column in a dataframe from HH:MM:SS format to the 
    corresponding time in h, m, or s, and add to the dataframe.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        data frame to operate on
    unit : str, optional
        unit to convert to - h(ours), m(inutes), s(econds), by default "s"
    colname : str, optional
        name of column to convert, by default "time"
    newcolname : str, optional
        name of new column to be added to df, by default "time_s"

    Raises
    ------
    ValueError
        invalid unit input
    """
    if unit == "h":
        multipliers = [1, 1/60, 1/3600]
    elif unit == "m":
        multipliers = [60, 1, 1/60]
    elif unit == "s":
        multipliers = [3600, 60, 1]
    else:
        raise ValueError("unit must be one of 'h', 'm', or 's'!")
    times = pd.DatetimeIndex(df[colname])
    df[newcolname] = times.hour * multipliers[0] + times.minute * multipliers[1] + times.second * multipliers[2]
import numpy as np
import pandas as pd

__all__ = ['create_error_col']


def create_error_col(dat, col, splitat='+/-'):
    """
    Split a pandas dataframe column with an error into two columns

    Parameters
    ----------
    dat: data frame object
    col: column to split
    splitat: the character combination to split the entry at

    Return
    ------
    None: new data column is appended to dataframe with column name [col]_ERR
    """
    val_err = np.array([d.split(splitat) for d in dat.loc[:,col]], dtype='float')
    dat.loc[:,col] = val_err[:,0]
    dat.insert(len(dat.columns), col+'_ERR', val_err[:,1])

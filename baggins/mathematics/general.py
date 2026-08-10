import math

__all__ = ["next_square_root", "next_square"]


def next_square_root(n):
    """
    Generate the next integer square root number x such that x**2 > n.

    Parameters
    ----------
    n : int
        given number

    Returns
    -------
    root : int
        next perfect square root number
    """
    root = math.isqrt(n)
    if root * root <= n:
        root += 1
    return root


def next_square(n):
    """
    Generate the next square number above n.

    Parameters
    ----------
    n : int
        given number

    Returns
    -------
    : int
        next perfect square number
    """
    return next_square_root(n) ** 2

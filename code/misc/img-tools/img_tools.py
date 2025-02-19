import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.filters import gaussian
from  baggins.env_config import _cmlogger


__all__ = ["rescale_image", "fit_circle_to_image"]


_logger = _cmlogger.getChild(__name__)


class FittedCircle:
    def __init__(self, a, cx, cy, r) -> None:
        self.accum = a
        self.cx = cx
        self.cy = cy
        self.r = r

    def __repr__(self):
        return f"{type(self)}: radius={self.r:.1e} centred at ({self.cx:.1e}, {self.cy:.1e})"


def rescale_image(img):
    alpha = 0.3
    img = np.clip(img, np.quantile(img, alpha), np.quantile(img, 1-alpha))
    img =  (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def fit_circle_to_image(img, num_circles, sigma_blur=0.2, sigma_canny=4, hough_radii=np.arange(5, 25, 1), debug=False):
    """
    Fit a circle to an image.

    Parameters
    ----------
    img : np.array
        image to fit the circle to
    num_circles : int
        expected number of circles
    sigma_blur : float, optional
        blurring std dev, by default 2
    sigma_canny : float, optional
        edge detection std dev, by default 10
    hough_radii : array-like, optional
        expected range of circle radii, by default np.arange(20, 100, 2)

    Returns
    -------
    circles : list
        fitted circles, in order of increasing radius
    """
    # apply Gaussian smoothing to reduce noise
    img = rescale_image(img)
    blurred_image = gaussian(img, sigma=sigma_blur)
    if debug:
        _logger.debug("Plotting the blurred image...")
        plt.imshow(blurred_image)
        plt.show()
    # apply Canny edge detection
    edges = feature.canny(blurred_image, sigma=sigma_canny)

    # TODO make configurable
    dist_threshold = 5
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    approx_centre_x = int(img.shape[0]/2)
    approx_centre_y = int(img.shape[1]/2)
    print(f"Approx centre is at ({approx_centre_x}, {approx_centre_y})")
    dist_from_approx_centre = np.sqrt((X - approx_centre_x)**2 + (Y - approx_centre_y)**2)
    edges = edges & (dist_from_approx_centre <= dist_threshold)

    # detect circles using Hough Transform
    hough_res = hough_circle(edges, hough_radii)
    # extract the most prominent circles
    res = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=num_circles)
    try:
        assert len(res[0]) > 0
    except AssertionError:
        _logger.exception("Circle detection algorithm failed! Try adjusting input parameters.", exc_info=True)
        raise
    circles = []
    for a, cx, cy, r in zip(*res):
        circles.append(FittedCircle(a=a, cx=cx, cy=cy, r=r))
    # sort detected circles by radius
    circles.sort(key = lambda x: x.r)
    return circles

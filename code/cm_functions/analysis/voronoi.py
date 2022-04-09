import numpy as np
import scipy.optimize, scipy.ndimage
from scipy.stats import binned_statistic_2d
from voronoi_binning import voronoi_binned_image

__all__ = ["voronoi_grid", "gauss_hermite_function", "fit_gauss_hermite_distribution", "voronoi_binned_los_V_statistics", "lambda_R"]


def voronoi_grid(x, y, Npx=100, extent=None, part_per_bin=500):
    """
    Create the voronoi grid by assigning particles to correct bins.
    Original form by Matias Mannerkoski

    Parameters
    ----------
    x : np.ndarray
        x coordinates of particles
    y : np.ndarray
        y coordinates of particles
    Npx : int, optional
        number of bins, by default 100
    extent : float, optional
        image extent, by default None (allows for full image to be plotted)
    part_per_bin : int, optional
        target number of particles per bin, by default 500

    Returns
    -------
    particle_vor_bin_num : np.ndarray
        voronoi bin number of particles
    pixel_vor_bin_num : np.ndarray
        voronoi bin number of pixels
    extent : tuple
        range to bin dimensions over
    x_bin : np.ndarray
        x coordinate of bin centres
    y_bin : np.ndarray
        y coordinate of bin centres
    """
    #bin in the x-y plane
    nimg, xedges, yedges, grid_bin_num = binned_statistic_2d(x, y, values=None, statistic='count', bins=Npx, range=extent, expand_binnumbers=True)
    
    #determine image extent
    w = xedges[-1] - xedges[0]
    h = yedges[-1] - yedges[0]
    
    #assign particles to voronoi bin
    pixel_vor_bin_num = voronoi_binned_image(nimg, part_per_bin, w, h)
    particle_vor_bin_num = np.full(x.shape, -1, dtype=int)
    valid_grid_bin_mask = np.logical_and(np.all(grid_bin_num > 0, axis=0), np.all(grid_bin_num < Npx+1, axis=0))
    indx, indy = grid_bin_num[:, valid_grid_bin_mask]-1
    particle_vor_bin_num[valid_grid_bin_mask] = pixel_vor_bin_num[indy, indx]
    
    if extent is None:
        extent = (*xedges[[0, -1]], *yedges[[0, -1]])
    
    #create mesh
    X, Y = np.meshgrid((xedges[1:] + xedges[-1:])/2, (yedges[1:] + yedges[-1:])/2)
    index = np.unique(pixel_vor_bin_num)
    bin_sums = scipy.ndimage.sum(nimg, labels=pixel_vor_bin_num, index=index)
    x_bin = scipy.ndimage.sum(nimg * X, labels=pixel_vor_bin_num, index=index) / bin_sums
    y_bin = scipy.ndimage.sum(nimg * Y, labels=pixel_vor_bin_num, index=index) / bin_sums
    
    return particle_vor_bin_num, pixel_vor_bin_num, extent, x_bin, y_bin


def gauss_hermite_function(x, mu, sigma, h3, h4):
    """
    The normalised (when non-negative) function corresponding to the first
    three terms in the expansion used by van der Marel & Franx 
    (1993ApJ...407..525V) and others following their methods. 
    Original form by Matias Mannerkoski.

    Parameters
    ----------
    x : np.ndarray
        data values
    mu : float
        mean of x
    sigma : float
        standard deviation of x
    h3 : float
        3rd moment
    h4 : float
        4th moment

    Returns
    -------
    : np.ndarray
        function value of x
    """
    w = (x-mu)/sigma
    a = np.exp(-.5*w**2)/np.sqrt(2*np.pi)
    H3 = (2*w**3 - 3*w)/np.sqrt(3)
    H4 = (4*w**4 - 12*w**2 + 3)/np.sqrt(24)
    N = np.sqrt(6)*h4*sigma/4 + sigma #normalization when the function is non-negative
    #TODO for fitting the function should be always normalized
    return np.clip(a * (1 + h3*H3 + h4*H4)/N, 1e-300, None)


def fit_gauss_hermite_distribution(data):
    """
    Fit a Gauss-Hermite distribution to the data.
    Original form by Matias Mannerkoski.

    Parameters
    ----------
    data : np.ndarray
         data to fit function for

    Returns
    -------
    mu0 : float
        fit mean
    sigma0 : float
        fit standard deviation
    h3 : float
        3rd moment 
    h4 : float
        4th moment
    """
    if len(data) == 0:
        return 0.,0.,0.,0.
    # the gauss hermite function is made to have the same mean and sigma as the
    # plain gaussian so compute them with faster estimates
    mu0 = np.mean(data)
    sigma0 = np.std(data)

    def log_likelihood(pars):
        return -np.sum(np.log(gauss_hermite_function(data, mu0, sigma0, *pars)))
    
    res = scipy.optimize.minimize(log_likelihood, (0., 0.))
    
    # assume that this just worked
    h3, h4 =  res.x
    return mu0, sigma0, h3, h4


def voronoi_binned_los_V_statistics(x,y,V,m, Npx, **kwargs):
    """
    Determine the statistics of each voronoi bin.

    Parameters
    ----------
    x : np.ndarray
        x coordinates of bins
    y : np.ndarray
        y coordinates of bins
    V : np.ndarray
        LOS velocity
    m : np.ndarray
        masses
    Npx : int
        number of pixels per voronoi bin

    Returns
    -------
    : dict
        binned quantitites convereted to CoM frame
    """
    M = np.sum(m)
    xcom = np.sum(m * x)/M
    ycom = np.sum(m * y)/M
    Vcom = np.sum(m * V)/M
    x = x-xcom
    y = y-ycom
    vz = V-Vcom

    print(f"Binning {len(x)} particles...")
    particle_vor_bin_num, pixel_vor_bin_num, extent, xBar, yBar = voronoi_grid(x,y, Npx=Npx, **kwargs)
    bin_index = list(range(int(np.max(particle_vor_bin_num)+1)))
    
    bin_mass = scipy.ndimage.sum(m,
                    labels=particle_vor_bin_num, index=bin_index)

    fits = []
    for i in bin_index:
        print("Fitting bin:", i, end='\r')
        fits.append(fit_gauss_hermite_distribution(vz[particle_vor_bin_num==i]))
    print()
    bin_stats = np.array(fits)
    img_stats = bin_stats[pixel_vor_bin_num]

    return dict(
        xBar=xBar, yBar=yBar,
        bin_V=bin_stats[:,0], bin_sigma=bin_stats[:,1],
        bin_h3=bin_stats[:,2], bin_h4=bin_stats[:,3],
        bin_mass=bin_mass,
        img_V=img_stats[...,0], img_sigma=img_stats[...,1],
        img_h3=img_stats[...,2], img_h4=img_stats[...,3],
        extent=extent, xcom = xcom, ycom=ycom, Vcom=Vcom,
        )


def lambda_R(vorbin_stats, re):
    """
    Determine the lambda(R) spin parameter.
    Original form by Matias Mannerkoski

    Parameters
    ----------
    vorbin_stats : dict
        output of voronoi_binned_los_V_statistics() method
    re : float
        projected half mass (or half light) radius

    Returns
    -------
    : np.ndarray
        radial values in units of Re
    : np.ndarray
        lambda(R) value
    """
    R = np.sqrt(vorbin_stats['xBar']**2 + vorbin_stats['yBar']**2)
    inds = np.argsort(R)
    R = R[inds]
    F = vorbin_stats['bin_mass'][inds]
    V = vorbin_stats['bin_V'][inds]
    s = vorbin_stats['bin_sigma'][inds]
    return R/re, np.cumsum(F*R*np.abs(V))/np.cumsum(F*R*np.sqrt(V**2 + s**2))


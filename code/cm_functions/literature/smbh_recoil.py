import numpy as np
from ..env_config import _cmlogger

__all__ = ["ketju_calculate_bh_merger_remnant_properties"]


_logger = _cmlogger.getChild(__name__)

# set the unit system
# mass: 1e10 Msol
# distance: kpc
# velocity: km/s
G = 4.300917e4
c = 299792458.0 / 1e3

def ketju_calculate_bh_merger_remnant_properties(m1, m2, s1, s2, x1, x2, v1, v2):
    """
    Determine the properties of a merger BH binary following the Zlochower & 
    Lousto 2015 prescription.
    This code has been directly copied from the implementation in the Ketju 
    integrator, from the repo `ketju-integrator-dev', commit 7a43ef1, and 
    transformed from C into python code. The original code has the same function
    name as this one. 

    Parameters
    ----------
    m1, m2 : float
        mass of BHs in Msol
    s1, s2 : array-like
        dimensionless spin vector of BHs
    x1, x2 : array-like
        position vector of BHs
    v1, v2 : array-like
        velocity vector of BHs

    Returns
    -------
    remnant : dict
        remnant properties (mass, spin, velocity). Note spin is converted to 
        dimensional units.
    """
    s1 = np.atleast_1d(s1)
    s2 = np.atleast_1d(s2)
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    v1 = np.atleast_1d(v1)
    v2 = np.atleast_1d(v2)

    try:
        assert np.linalg.norm(s1) <= 1.0 and np.linalg.norm(s2) <= 1.0
    except AssertionError:
        _logger.exception("Spin input must have a magnitude less than or equal to 1!", exc_info=True)
        raise

    # set up an output dict
    remnant = {"m":np.nan, "s":[np.nan, np.nan, np.nan], "v":[np.nan, np.nan, np.nan]}

    # total mass
    m = m1 + m2;

    # relative position and velocity
    rv = x1 - x2
    vv = v1 - v2

    # orbital angular momentum (Newtonian) and corresponding unit vector
    L = np.cross(rv, vv) # TODO check that this is giving the right dimensions
    L *= m1 * m2  / m
    Lhat = L / np.linalg.norm(L)

    # Check that L is non-zero to handle the corner case of a head-on merger
    # separately. This shouldn't really ever happen, but don't want it to
    # crash then anyway.
    if np.all(L==0):
        _logger.warning("Angular momentum is 0! No significant GW emission.")
        # In this case there won't be significant GW emission,
        # so just use Newtonian values and assume spin conservation.
        remnant["m"] = m
        remnant["v"] = np.zeros(3)
        remnant["s"] = s1 + s2
        return remnant

    # set n0 equal to separation vector. vertical direction is given
    # by L_hat. right handed orthogonal basis given by n0, m0 and Lhat
    n0 = rv / np.linalg.norm(rv)
    m0 = np.cross(Lhat, n0)
    # rotate n0 by -59 degrees to get m59
    angle = -59 * np.pi / 180.0;
    m59 = n0 * np.cos(angle) + m0 * np.sin(angle)

    # we need direction of total angular momentum for the direction of remnant
    # spin
    J = s1 + s2 + L
    Jhat = J / np.linalg.norm(J)

    # the paper uses spins in units of M^2
    s1 *= m1**2
    s2 *= m2**2

    delta_m = (m1 - m2) / m
    eta = (1 - delta_m * delta_m) / 4.
    S_tilde  = (s1 + s2) / m**2
    Delta_tilde = (s2 / m2 - s1 / m1) / m
    S0_tilde = S_tilde + 0.5 * delta_m * Delta_tilde

    # quantities are decomposed into para (parallel to orbital angular
    # momentum) and ortho components
    Delta_tilde_para = np.dot(Delta_tilde, Lhat)
    Deltax = np.dot(Delta_tilde, n0)
    Deltay = np.dot(Delta_tilde, m0)
    Delta_tilde_ortho = np.linalg.norm(np.array([Deltax, Deltay]))

    S_tilde_para = np.dot(S_tilde, Lhat)
    Sx = np.dot(S_tilde, n0)
    Sy = np.dot(S_tilde, m0)
    S_tilde_ortho = np.linalg.norm(np.array([Sx, Sy]))

    S0_tilde_para = np.dot(S0_tilde, Lhat)

    # Calculate the components of recoil velocity, in km/s
    # Eq. (37)
    # Note that this equation appears to have a typo in the leading eta term
    # in the paper. There it reads 4*pow(eta,2), but using that form gives
    # far too low maximum kick values compared to the simulation results
    # in e.g. table XI. All the other formulae use pow(4*eta,2) as well,
    # including the closely related eq 19, so it seems that this is the correct
    # form. The maximal kick values from table XI are also reproduced within
    # the quoted tolerance using this formula.
    v_para = (4*eta)**2 * (np.dot(Delta_tilde, n0)
               * (3678. - 2475. * delta_m**2 + 4962. * S0_tilde_para
                  + 7170. * S0_tilde_para**2
                  + 12050. * S0_tilde_para**3)
           + np.dot(S0_tilde, m59) * (Delta_tilde_para
                        * (4315. - 1262. * delta_m**2
                           + 15970. * S0_tilde_para)
                    - 2256. * delta_m - 2231. * delta_m * S0_tilde_para))

    # Eq. (39)
    v_ortho2 = (4 * eta)**4 * (2.106e5 * Delta_tilde_para**2
               + 4.967e5 * Delta_tilde_para * delta_m
               - 2.116e5 * Delta_tilde_para**3 * delta_m
               - 5.037e5 * Delta_tilde_para**2 * delta_m**2
               - 1.269e5 * Delta_tilde_para * delta_m**3
               - 3.384e5 * Delta_tilde_para**2 * S0_tilde_para
               - 6.440e5 * delta_m**2 * S0_tilde_para
               + 2.138e6 * Delta_tilde_para**2 * S0_tilde_para**2
               - 4.905e6 * Delta_tilde_para * delta_m * S0_tilde_para**2
               - 1.100e6 * delta_m**2 * S0_tilde_para**2
               - 1.024e7 * Delta_tilde_para**2 * S0_tilde_para**3) \
        + (1.2e4 * eta**2 * delta_m * (1 - 0.93 * eta))**2
    if v_ortho2 < 0:
        v_ortho2 = 0
    v_ortho = np.sqrt(v_ortho2)

    # Conversion between the internal unit system defined by the given value of
    # c and the SI units. The model gives kick velocities in km/s.
    km_per_s = c / 2.99792458e5;

    # XXX: we will set the parallel component along the direction of
    # original r, since the PN dynamics can't follow the binary until the
    # initial separation of the simulations.
    # This makes the direction essentially random.
    remnant["v"] = km_per_s * (v_para * Lhat + v_ortho * n0)


    # For mass loss and spin, we need (dimensionless) energy and angular
    # momentum evaluated at ISCO of Kerr metric, with spin = S_tilde_para.
    # Formulas are standard, mostly given after Eq. (35) in the corrected arXiv
    # version.
    chi = S_tilde_para
    sign_chi = np.sign(chi)
    Z1 = 1 + pow(1 - chi * chi, 1. / 3) * (pow(1 + chi, 1. / 3) + pow(1 - chi, 1. / 3))
    Z2 = np.sqrt(3 * chi * chi + Z1 * Z1);
    # also dimensionless / in units of system mass
    r_isco = 3 + Z2 - sign_chi * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    L_isco = 2. / np.sqrt(3 * r_isco) * (3 * np.sqrt(r_isco) - 2 * chi)
    # E_isco not given in the paper, but is a standard result.
    E_isco = np.sqrt(1 - 2. / (3 * r_isco))

    # Calculate mass loss.
    # Only use the E_c term in Eq. (30), the other term should be small and
    # can't be modelled so that it would improve accuracy.
    k2a = -0.024;
    k2d = 0;
    k3a = -0.055;
    k3b = 0.0;
    k3d = -0.019;
    k4a = -0.119;
    k4b = 0.005;
    k4e = 0;
    k4f = 0.035;
    k4g = 0.022;
    E_A = 0;
    E_B = 0.59;
    E_D = 0;
    E_E = -0.51;
    E_F = 0.056;
    E_G = -0.073;
    E_H = 0;
    e1 = 0.0356;
    e2 = 0.096;
    e3 = 0.122;
    eps1 = 0.0043;
    eps2 = 0.0050;
    eps3 = -0.009;

    # Eq. (33)
    E_HU = 0.0025829 - 0.0773079 / (2 * S0_tilde_para - 1.693959)
    # Eq. (32)
    Ec_para = pow(4 * eta, 2) * (E_HU + k2a * delta_m * Delta_tilde_para
                + 0.000743 * pow(Delta_tilde_para, 2) + k2d * pow(delta_m, 2)
                + k3a * delta_m * Delta_tilde_para * S_tilde_para
                + k3b * S_tilde_para * pow(Delta_tilde_para, 2)
                + k3d * pow(delta_m, 2) * S_tilde_para
                + k4a * delta_m * Delta_tilde_para * pow(S_tilde_para, 2)
                + k4b * delta_m * pow(Delta_tilde_para, 3)
                + 0.000124 * pow(Delta_tilde_para, 4)
                + k4e * pow(Delta_tilde_para, 2) * pow(S_tilde_para, 2)
                + k4f * pow(delta_m, 4)
                + k4g * pow(delta_m, 3) * Delta_tilde_para)
    + pow(delta_m, 6) * eta * (1 - E_isco)

    # Eq. (34)
    Ec = Ec_para + pow(4 * eta, 2) * (pow(S_tilde_ortho, 2)
                        * (e1 + e2 * S_tilde_para + e3 * pow(S_tilde_para, 2))
                    + pow(Delta_tilde_ortho, 2)
                        * (eps1 + eps2 * S_tilde_para
                            + eps3 * pow(S_tilde_para, 2))
                    + pow(delta_m, 2) * pow(S_tilde_ortho, 2)
                        * (E_A + S_tilde_para * E_B)
                    + pow(delta_m, 2) * pow(Delta_tilde_ortho, 2)
                        * (E_D + S_tilde_para * E_E)
                    + E_F * delta_m * Delta_tilde_ortho * S_tilde_ortho
                    + E_G * pow(Delta_tilde_para, 2)
                        * pow(Delta_tilde_ortho, 2)
                    + E_H * pow(Delta_tilde_para, 2) * pow(S_tilde_ortho, 2));

    # take delta M = (M1 + M2 - M_remnant)/(M1 + M2) = Ec
    remnant["m"] = m * (1 - Ec);

    # Calculate square of remnant spin magnitude, using only the A_c term in
    # Eq. (31). Spin direction is  assumed to be equal to total angular
    # momentum, which the paper assures is accurate to within 20 degrees.

    L0 = 0.686710;
    L1 = 0.613247;
    L2a = -0.145427;
    L2b = -0.115689;
    L2c = -0.005254;
    L2d = 0.801838;
    L3a = -0.073839;
    L3b = 0.004759;
    L3c = -0.078377;
    L3d = 1.585809;
    L4a = -0.003050;
    L4b = -0.002968;
    L4c = 0.004364;
    L4d = -0.047204;
    L4e = -0.053099;
    L4f = 0.953458;
    L4g = -0.067998;
    L4h = 0.001629;
    L4i = -0.066693;

    # The A_* coefficients have large uncertainties, so no need to bother
    # with the last decimals.
    A_A = 3.0;
    A_B = -7.3;
    A_D = -2.0;
    A_E = 5.1;
    A_F = 0;
    A_G = -2.8;
    A_H = 5.1;
    a1 = 0.8401;
    a2 = -0.328;
    a3 = -0.61;
    zeta1 = -0.0209;
    zeta2 = -0.038;
    zeta3 = 0.04;

    # Eq. (35)
    alpha_align = pow(4 * eta, 2) * (L0 + L1 * S_tilde_para + L2a * delta_m * Delta_tilde_para
                   + L2b * pow(S_tilde_para, 2) + L2c * pow(Delta_tilde_para, 2)
                   + L2d * pow(delta_m, 2)
                   + L3a * Delta_tilde_para * S_tilde_para * delta_m
                   + L3b * S_tilde_para * pow(Delta_tilde_para, 2)
                   + L3c * pow(S_tilde_para, 3)
                   + L3d * S_tilde_para * pow(delta_m, 2)
                   + L4a * Delta_tilde_para * pow(S_tilde_para, 2) * delta_m
                   + L4b * pow(Delta_tilde_para, 3) * delta_m
                   + L4c * pow(Delta_tilde_para, 4) + L4d * pow(S_tilde_para, 4)
                   + L4e * pow(Delta_tilde_para, 2) * pow(S_tilde_para, 2)
                   + L4f * pow(delta_m, 4)
                   + L4g * Delta_tilde_para * pow(delta_m, 3)
                   + L4h * pow(Delta_tilde_para * delta_m, 2)
                   + L4i * pow(S_tilde_para * delta_m, 2)) 
    + S_tilde_para * (1 + 8 * eta) * pow(delta_m, 4) 
    + eta * L_isco * pow(delta_m, 6)

    # Eq. (36)
    Ac = pow(alpha_align, 2) + pow(4 * eta, 2) * (pow(S_tilde_ortho, 2)
                         * (a1 + a2 * S_tilde_para + a3 * pow(S_tilde_para, 2))
                     + pow(Delta_tilde_ortho, 2)
                           * (zeta1 + zeta2 * S_tilde_para
                              + zeta3 * pow(S_tilde_para, 2))
                     + pow(delta_m, 2) * pow(S_tilde_ortho, 2)
                           * (A_A + A_B * S_tilde_para)
                     + pow(delta_m, 2) * pow(Delta_tilde_ortho, 2)
                           * (A_D + A_E * S_tilde_para)
                     + A_F * delta_m * Delta_tilde_ortho * S_tilde_ortho
                     + A_G * pow(Delta_tilde_para, 2)
                           * pow(Delta_tilde_ortho, 2)
                     + A_H * pow(Delta_tilde_para, 2) * pow(S_tilde_ortho, 2)) 
    + pow(delta_m, 6) * (1 + 12 * eta) * pow(S_tilde_ortho, 2);

    # square of dimensionless spin magnitude now given by Ac. multiply in
    # G/c*remnant_mass^2 to get simulation units
    spin_mag = np.sqrt(Ac);

    # Safety check if the fits don't extrapolate to extreme cases nicely.
    if spin_mag > 1.:
        spin_mag = 1.

    rm2 = remnant["m"]**2
    remnant["s"] = (G / c) * rm2 * spin_mag * Jhat

    return remnant

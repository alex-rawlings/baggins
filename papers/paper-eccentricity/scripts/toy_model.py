import itertools
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# units Gyr, 1e10 Msol, kpc
G = 44900
km_per_s = 1.02201216 #kpc/Gyr


class System:
    
    def __init__(self,
                 M_BH=3e-1,
                 e_spheroid=0.9, rho=1.0,
                 stellar_sigma=310*km_per_s, df_fudge_factor=0.5,
                 ):
        
        self.M_BH = M_BH # single BH mass
         
        e2s = e_spheroid**2
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
        A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)
        self.Avec = np.array([[A3],[A1]])
        self.rho = rho

        self.stellar_sigma = stellar_sigma
        self.df_fudge_factor = df_fudge_factor
        self.logL = 10

        self.a_hard = G * M_BH / (8 * stellar_sigma**2)
        self.r_infl = np.cbrt(2*M_BH/(rho*(4/3*np.pi)))


# The actual potential in the sims is fairly similar to a spheroidal potential
# with e~0.7-0.9
    def ellipsoid_accel(self, x):
        return -2 * np.pi * G * self.rho * self.Avec * x


    def dynamical_friction(self, x, v):

        v_single = v/2
        v_single_norm = np.linalg.norm(v_single,axis=0)
        if v_single_norm/self.stellar_sigma < 1e-3: # taylor expansion for small values
            df_single = -(v_single * 8*np.sqrt(np.pi) * self.M_BH * self.rho 
                          * self.logL / (3*np.sqrt(2) * self.stellar_sigma**3)
                         )
        else:
            X = v_single_norm/(np.sqrt(2)*self.stellar_sigma) 
            df_single = -(4*np.pi * G**2 * self.M_BH * self.rho * self.logL 
                          * (erf(X) - 2*X/np.sqrt(np.pi)*np.exp(-X**2))
                          * v_single/v_single_norm**3)

        return 2*df_single * self.df_fudge_factor

    def df_decoupling_factor(self, x):
        r = np.linalg.norm(x,axis=0)
        cutoff_point = self.a_hard * 2
        cutoff_scale = self.a_hard * 0.5
        return 1/(1+np.exp(-(r-cutoff_point)/cutoff_scale))


# x is the relative distance vector between BHs
    def accel(self, x, v):
        return (-G * 2 * self.M_BH/np.linalg.norm(x,axis=0)**3 * x 
                + 2 * self.ellipsoid_accel(x/2)
                + self.dynamical_friction(x,v)*self.df_decoupling_factor(x)
                )
        

    def integrate(self, x0, v0, ts):
        def dydt(t,y):
            dvdt = self.accel(y[:2], y[2:])
            return np.append(y[2:], dvdt, axis=0)

        def min_E(t,y):
            E = self.orbital_energy(y[:2], y[2:])
            return E + G*self.M_BH/self.a_hard
        min_E.terminal=True

        res = solve_ivp(dydt, (ts[0], ts[-1]), np.append(x0,v0),
                        t_eval=ts, vectorized=True,
                        method='DOP853',
                        events=[min_E],
                        rtol=1e-8, atol=1e-12)
        t = res.t
        x = res.y[:2]
        v = res.y[2:]

        return x,v,t

    def orbital_energy(self, x,v):
        r = np.linalg.norm(x,axis=0)
        v2 = np.sum(v*v,axis=0)
        return .5 * v2 - G * 2 * self.M_BH/r
        

    def eccentricity(self, x, v):
        h2 = np.cross(x,v, axis=0)**2
        E = self.orbital_energy(x,v)
        return np.sqrt(1 + .5*E*h2/(G * self.M_BH)**2)
    
    def semimajor_axis(self, x, v):
        return -G*self.M_BH/self.orbital_energy(x,v)

# helper for multiprocessing
def compute_res(args):
    sys, b, v, r = args
    x0 = [r,b]
    v0 = [-v,0]
    tmax = 200 * r/v
    ts = np.linspace(0,tmax, 500)

    x,v,t = sys.integrate(x0,v0,ts)
    return x,v,t

def compute_deflection_angle(x,v,rmin):
    r = np.linalg.norm(x,axis=0)
    rdot = np.sum(x*v,axis=0)/r
    i = np.nonzero((rdot>0)&(r>rmin))[0][0]
    return np.pi + np.arctan2(v[1,i],v[0,i])

def main_plots():
    fig, axes = plt.subplots(3,1,sharex='col')
    fig2, ax_orbit = plt.subplots(1,1)
#   ax_orbit.set_aspect('equal')

# The system gives identical behavior when scaled in a certain way, these are
# the individual free parameters:
    M_BH_ref = 1e-2 # single BH mass
    rho_ref = 40 # stellar density
    sigma_ref = 200*km_per_s # stellar velocity dispersion
    gal_e = 0.9

    v0_per_sigma = 2.5 # initial BH velocity / stellar sigma
    r0_per_rinfl = 3 # initial BH separation/single BH influence radius
    bmin_per_r0,bmax_per_r0 = 1e-3, 2e-1 # impact parameter limits / initial separation

    Mscale = 1 # free scale parameter for BH mass
    vscale = 1 # free scale parameter for velocities

    rscale = Mscale/vscale**2 # scaling of distances for identical behaviour
    rhoscale = Mscale/rscale**3 # scaling of density for identical behaviour

    sys = System(M_BH=M_BH_ref*Mscale,
                 rho=rho_ref*rhoscale,
                 e_spheroid=gal_e,
                 stellar_sigma=sigma_ref*vscale,
                 df_fudge_factor=0.5)


    v0 = v0_per_sigma * sigma_ref * vscale
    r0 = r0_per_rinfl * sys.r_infl

    efin = []
    deflection_angle = []
    bs = np.linspace(bmin_per_r0,bmax_per_r0, 30)*r0

    with multiprocessing.Pool() as pool:
        for b,(x,v,t) in zip(bs, 
                              pool.imap(compute_res,
                                        zip(itertools.repeat(sys),
                                            bs,
                                            itertools.repeat(v0),
                                            itertools.repeat(r0)))
                             ):
            color = plt.cm.viridis((b-bs[0])/(bs[-1]-bs[0]))
            ax_orbit.plot(*x,color=color)
            axes[0].plot(t, np.linalg.norm(x,axis=0),color=color)
            axes[1].plot(t, sys.semimajor_axis(x,v), color=color)
            e = sys. eccentricity(x,v)
            axes[2].plot(t, e,color=color)
            efin.append(e[-1])
            #deflection_angle.append(compute_deflection_angle(x,v,sys.r_infl*4))
            print(b, "done")

    axes[0].set_ylabel("R/kpc")
    axes[1].set_ylabel("a/kpc")
    axes[2].set_ylabel("e")

    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-3,20)
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-3,20)
    axes[2].set_ylim(0,1.5)


    plt.figure()
    plt.plot(bs*1e3, efin, '-')
    plt.xlabel('Impact parameter/pc')
    plt.ylabel('Final e')

    #plt.figure()
    #plt.plot(np.degrees(deflection_angle), efin, '-')
    #plt.xlabel('Deflection angle/deg')
    #plt.ylabel('Final e')

    plt.show()

# TODO use some global optimization algo (shgo?) to find all the local minima in
# a single pass
def find_e_min(args):
    sys, v0, b0, r0, minimum_index = args
    def f(b):
        x,v,t = compute_res((sys, b, v0, r0))
        return sys.eccentricity(x,v)[-1]
    if minimum_index == 0:
        bounds = [0,2*b0]
    else:
        bounds = [.5*b0, 5*b0]
    res = minimize_scalar(f, method='bounded', bounds=bounds, options=dict(xatol=1e-2*b0))
    b = res.x
    x,v,t = compute_res((sys, b, v0, r0))
    e = sys.eccentricity(x,v)[-1]
    angle = compute_deflection_angle(x,v,sys.r_infl)
    return b, angle, e


def ecc_minimum_plot():
    fig, axes = plt.subplots(3,1, sharex='col')

    M_BH_ref = 1e-2 # single BH mass
    rho_ref = 40 # stellar density
    sigma_ref = 200*km_per_s # stellar velocity dispersion
    gal_e = 0.9

    v0_per_sigma = 2.5 # initial BH velocity / stellar sigma
    r0_per_rinfl = 3 # initial BH separation/single BH influence radius

    Mscale = 1 # free scale parameter for BH mass
    vscale = 1 # free scale parameter for velocities

    rscale = Mscale/vscale**2 # scaling of distances for identical behaviour
    rhoscale = Mscale/rscale**3 # scaling of density for identical behaviour

    sys = System(M_BH=M_BH_ref*Mscale,
                 rho=rho_ref*rhoscale,
                 e_spheroid=gal_e,
                 stellar_sigma=sigma_ref*vscale,
                 df_fudge_factor=0.5)

    v0 = v0_per_sigma * sigma_ref * vscale
    r0 = r0_per_rinfl * sys.r_infl

    v0s = v0*(1 + np.linspace(-.2,.25,50))
    
    for b_index,b_guess in enumerate([5e-3*rscale, 15e-3*rscale]):
        bs, angles, emins = [],[],[]
        with multiprocessing.Pool() as pool:
            for v0, (b, angle, emin) in zip(v0s,
                                            pool.imap(find_e_min, 
                                                      zip(itertools.repeat(sys),
                                                          v0s,
                                                          itertools.repeat(b_guess),
                                                          itertools.repeat(r0),
                                                          itertools.repeat(b_index)))
                                            ):
                bs.append(b)
                angles.append(angle)
                emins.append(emin)
                print(v0, "done")


        axes[0].plot(v0s/km_per_s, emins, marker='.')
        axes[1].plot(v0s/km_per_s, np.array(bs)*1e3, marker='.')
        axes[2].plot(v0s/km_per_s, np.degrees(angles), marker='.')

    axes[0].set_ylabel('Min e')
    axes[1].set_ylabel('Min e impact parameter/pc')
    axes[2].set_ylabel('Min e deflection angle/deg')
    axes[-1].set_xlabel('Initial velocity / km/s')

    plt.show()



if __name__ == '__main__':
    main_plots()
    #ecc_minimum_plot()



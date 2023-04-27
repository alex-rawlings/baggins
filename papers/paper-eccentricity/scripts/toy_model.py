import itertools
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# units Gyr, 1e10 Msol, kpc
G = 44900
km_per_s = 1.02201216 #kpc/Gyr

BH_rel_vel0 = 800*km_per_s # reference relative vel of BHs
#BH_rel_vel0 = 760*km_per_s

class System:
    
    def __init__(self,
                 M_BH=3e-1,
                 e_spheroid=0.9, rho=1.0,
                 stellar_sigma=310*km_per_s, df_fudge_factor=0.5,
                 ):
        
        self.M_BH = 3e-1 # single BH mass
         
        e2s = e_spheroid**2
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
        A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)
        self.Avec = np.array([[A3],[A1]])
        self.rho = rho

        self.stellar_sigma = stellar_sigma
        self.df_fudge_factor = df_fudge_factor
        self.logL = 10


# The actual potential in the sims is fairly similar to a spheroidal potential
# with e~0.7-0.9
    def ellipsoid_accel(self, x):
        return -2*np.pi*G*self.rho * self.Avec * x


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
        return 1/(1+np.exp(-(r-0.02)/0.005))


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
            min_a = 2e-2
            E = self.orbital_energy(y[:2], y[2:])
            return E + G*self.M_BH/min_a
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
    sys, b, v = args
    x0 = [0.6,b]
    v0 = [-v,0]
    ts = np.linspace(0,0.05, 500)

    x,v,t = sys.integrate(x0,v0,ts)
    return x,v,t

def compute_deflection_angle(x,v):
    r = np.linalg.norm(x,axis=0)
    rdot = np.sum(x*v,axis=0)/r
    i = np.nonzero((rdot>0)&(r>0.2))[0][0]
    return np.pi + np.arctan2(v[1,i],v[0,i])

def main_plots():
    fig, axes = plt.subplots(3,1,sharex='col')
    fig2, ax_orbit = plt.subplots(1,1)
#   ax_orbit.set_aspect('equal')

    sys = System()

    efin = []
    deflection_angle = []
    bmin,bmax = 1e-3, 200e-3
# sample linearly in impact param probability, ~area
    #bs = np.linspace(bmin**2,bmax**2,400)**.5
    bs = np.linspace(bmin,bmax,100)
    v0 = BH_rel_vel0
    with multiprocessing.Pool() as pool:
        for y0,(x,v,t) in zip(bs, 
                              pool.imap(compute_res,
                                        zip(itertools.repeat(sys),
                                            bs,
                                            itertools.repeat(v0)))
                             ):
            color = plt.cm.viridis((y0-bmin)/(bmax-bmin))
            ax_orbit.plot(*x,color=color)
            axes[0].plot(t, np.linalg.norm(x,axis=0),color=color)
            axes[1].plot(t, sys.semimajor_axis(x,v), color=color)
            e = sys. eccentricity(x,v)
            axes[2].plot(t, e,color=color)
            efin.append(e[-1])
            deflection_angle.append(compute_deflection_angle(x,v))
            print(y0, "done")

    axes[0].set_ylabel("R/kpc")
    axes[1].set_ylabel("a/kpc")
    axes[2].set_ylabel("e")

    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-3,20)
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-3,20)
    axes[2].set_ylim(0,1.5)

    """
    plt.figure()
    plt.hist(efin, bins=np.linspace(0,1,61))
    plt.xlabel('Final e')
    plt.ylabel('Count')
    """

    plt.figure()
    plt.plot(bs*1e3, efin, '-')
    plt.xlabel('Impact parameter/pc')
    plt.ylabel('Final e')

    plt.figure()
    plt.plot(np.degrees(deflection_angle), efin, '-')
    plt.xlabel('Deflection angle/deg')
    plt.ylabel('Final e')

    plt.show()

def find_e_min(args):
    sys, v0, b0 = args
    def f(b):
        x,v,t = compute_res((sys, b[0], v0))
        return sys.eccentricity(x,v)[-1]
    res = minimize(f, b0, tol=1e-3, method='Powell', bounds=([0.5*b0, 2*b0],))
    b = res.x[0]
    x,v,t = compute_res((sys, b, v0))
    e = sys.eccentricity(x,v)[-1]
    angle = compute_deflection_angle(x,v)
    return b, angle, e


def ecc_minimum_plot():
    sys = System()
    fig, axes = plt.subplots(3,1, sharex='col')

    v0s = BH_rel_vel0 + np.linspace(-150,200,20) * km_per_s
    for b_guess in [30e-3, 100e-3]:
        bs, angles, emins = [],[],[]
        with multiprocessing.Pool() as pool:
            for v0, (b, angle, emin) in zip(v0s,
                                            pool.imap(find_e_min, 
                                                      zip(itertools.repeat(sys),
                                                          v0s,
                                                          itertools.repeat(b_guess)))
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



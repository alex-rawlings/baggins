import itertools
import multiprocessing
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, shgo

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

    def df_decoupling_factor(self, x, v):
        a = self.semimajor_axis(x,v)
        a[a<0] = np.inf
        cutoff_point = self.a_hard * 2
        cutoff_scale = self.a_hard * 0.5
        return 1/(1+np.exp(-(a-cutoff_point)/cutoff_scale))


# x is the relative distance vector between BHs
    def accel(self, x, v):
        return (-G * 2 * self.M_BH/np.linalg.norm(x,axis=0)**3 * x 
                + 2 * self.ellipsoid_accel(x/2)
                + self.dynamical_friction(x,v)*self.df_decoupling_factor(x,v)
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
    tmax = 150 * r/v
    ts = np.concatenate(([0], np.linspace(0.5*r/v,1.5*r/v, 50), np.linspace(1.5*r/v,tmax, 500)[1:]))

    x,v,t = sys.integrate(x0,v0,ts)
    return x,v,t

def compute_deflection_angle(sys,x,v):
    r = np.linalg.norm(x,axis=0)
    rdot = np.sum(x*v,axis=0)/r
    i = np.nonzero(rdot>0)[0][0]
    E = sys.orbital_energy(x[:,i],v[:,i])
    L = np.cross(x[:,i], v[:,i])
    return 2*np.arctan(G*2*sys.M_BH/(L*np.sqrt(2*E)))

def compute_argument_of_periapsis(sys, x, v):
    r = np.linalg.norm(x,axis=0)
    rdot = np.sum(x*v,axis=0)/r
    i = np.nonzero(rdot>0)[0][0]
    h = np.cross(x[:,i], v[:,i])
    evec = np.cross(v[:,i], [0,0,h])[:2]/(2*G*sys.M_BH) - x[:,i]/r[i]
    return np.arctan2(evec[1],evec[0])
    


def calculate_b_theta_e_curves(sys, r0, v0s, bs):
    Bs, V0s = np.meshgrid(bs, v0s)
    efin = []
    deflection_angle = []
    n_tot = np.prod(Bs.shape)
    with multiprocessing.Pool() as pool:
        for i, (x,v,t) in enumerate(pool.imap(compute_res,
                                              zip(itertools.repeat(sys),
                                                  Bs.ravel(),
                                                  V0s.ravel(),
                                                  itertools.repeat(r0))
                                              )):
            efin.append(sys.eccentricity(x[:,-1],v[:,-1]))
            deflection_angle.append(compute_deflection_angle(sys,x,v))
            print(f"{i+1}/{n_tot}")

    return np.array(deflection_angle).reshape(V0s.shape), np.array(efin).reshape(Bs.shape)
    

def parameter_space_scan_hernquist_conf():
# params matching Hernquist 11-0.825 high_e_no_stars runs fairly well
    for gal_e, df_fudge in itertools.product([0.8,0.9],[0.3, 0.5]):
        sys = System(M_BH=3e-1,
                     rho=1,
                     e_spheroid=gal_e,
                     stellar_sigma=310*km_per_s,
                     df_fudge_factor=df_fudge)


        r0 = sys.r_infl

        v0s = np.linspace(650, 950, 60) * km_per_s
        bs = np.linspace(0.1, 150, 200) * 1e-3
        
        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        
        res = dict(bs=bs*1e3, v0s=v0s/km_per_s, theta=theta, e=efin,
                   e_shperoid=gal_e, df_fudge=df_fudge)

        with open(f'data/hernquist_b_v_scan_es_{gal_e:.2f}_df_{df_fudge:.1f}.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=-1)

def parameter_space_scan_g05_conf():
# params matching gamma=0.5 e=0.99 runs fairly well
    for gal_e, df_fudge in itertools.product([0.85,0.9],[0.3, 0.5]):
        sys = System(M_BH=1e-2,
                     rho=40,
                     e_spheroid=gal_e,
                     stellar_sigma=200*km_per_s,
                     df_fudge_factor=df_fudge)


        r0 = sys.r_infl

        v0s = np.linspace(350, 600, 60) * km_per_s
        bs = np.linspace(0.01, 25, 250) * 1e-3
        
        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        
        res = dict(bs=bs*1e3, v0s=v0s/km_per_s, theta=theta, e=efin,
                   e_shperoid=gal_e, df_fudge=df_fudge)

        with open(f'data/g05_b_v_scan_es_{gal_e:.2f}_df_{df_fudge:.1f}.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=-1)
    
def parameter_space_scan_g05_conf2():
# params matching gamma=0.5 e=0.99 runs fairly well
    for gal_e, df_fudge in itertools.product([0.85,0.9],[0.3]):
        sys = System(M_BH=1e-2,
                     rho=70,
                     e_spheroid=gal_e,
                     stellar_sigma=225*km_per_s,
                     df_fudge_factor=df_fudge)

        r0 = sys.r_infl

        v0s = np.linspace(400, 650, 40) * km_per_s
        bs = np.linspace(0.01, 25, 250) * 1e-3
        
        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        
        res = dict(bs=bs*1e3, v0s=v0s/km_per_s, theta=theta, e=efin,
                   e_shperoid=gal_e, df_fudge=df_fudge)

        with open(f'data/g05_2_b_v_scan_es_{gal_e:.2f}_df_{df_fudge:.1f}.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=-1)
    
def parameter_space_scan_g05_conf3():
# params matching gamma=0.5 e=0.99 runs fairly well
    for gal_e, df_fudge in itertools.product([0.9],[0.5]):
        sys = System(M_BH=1e-2,
                     rho=30,
                     e_spheroid=gal_e,
                     stellar_sigma=200*km_per_s,
                     df_fudge_factor=df_fudge)

        r0 = 0.025

        v0s = np.linspace(400, 650, 40) * km_per_s
        bs = np.linspace(0.01, 25, 200) * 1e-3
        
        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        
        res = dict(bs=bs*1e3, v0s=v0s/km_per_s, theta=theta, e=efin,
                   e_shperoid=gal_e, df_fudge=df_fudge)

        with open(f'data/g05_3_b_v_scan_es_{gal_e:.2f}_df_{df_fudge:.1f}.pkl', 'wb') as f:
            pickle.dump(res, f, protocol=-1)

# main plot data generated with this function
def high_res_well_fitting_models():
    r0 = 25e-3
    bs = np.geomspace(0.1, 20, 500) * 1e-3
    e_spheroids = [0.895, 0.9, 0.905, 0.91]

    v0s = [450 * km_per_s, 560 * km_per_s]
    res090 = []
    res099 = []
    for e_s in e_spheroids:
    #e=0.9, 0.99 mergers are well fitted by this system, and the parameters are pretty similar to the sims as well
        sys = System(M_BH=1e-2,
                     rho=30,
                     e_spheroid=e_s,
                     stellar_sigma=200*km_per_s,
                     df_fudge_factor=0.5)

        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        res090.append(dict(theta=theta[0], e=efin[0], b=bs))
        res099.append(dict(theta=theta[1], e=efin[1], b=bs))

    with open(f'data/well_fitting_e_0.90_model_curve.pkl', 'wb') as f:
        pickle.dump(dict(e_spheroids=e_spheroids, res=res090), f, protocol=-1)

    with open(f'data/well_fitting_e_0.99_model_curve.pkl', 'wb') as f:
        pickle.dump(dict(e_spheroids=e_spheroids, res=res099), f, protocol=-1)

def gen_orbit_plot_data():
# Some orbits for paper for the e0=0.9 well fitting config + ~spherical system
    # for comparison
    r0 = 25e-3
    v0 = 450 * km_per_s
    bs = np.array([3.2e-3, 6e-3, 9.8e-3])
    for e_s in [0.2, 0.9]:
        sys = System(M_BH=1e-2,
                     rho=30,
                     e_spheroid=e_s,
                     stellar_sigma=200*km_per_s,
                     df_fudge_factor=0.5)
        res = []
        for b in bs:
            x = [r0,b]
            v = [-v0,0]
            tmax = 150 * r0/v0
            ts = np.linspace(0,tmax, 10000)

            x,v,t = sys.integrate(x,v,ts)
            e = sys.eccentricity(x,v)
            th = compute_deflection_angle(sys,x,v)
            res.append(dict(x=x,v=v,e=e,t=t,th=th,b=b))
            print(e_s, b, 'done')

        with open(f'data/sample_orbits_e_s_{e_s:.1f}.pkl', 'wb') as f:
            pickle.dump(res, f)





def test_plots():
    fig, axes = plt.subplots(3,1,sharex='col')
    fig2, ax_orbit = plt.subplots(1,1)
#   ax_orbit.set_aspect('equal')

# The system gives identical behavior when scaled in a certain way, these are
# the individual free parameters:
# Params matching gamma=0.5 e=.99/.97 runs fairly well
    M_BH_ref = 1e-2 # single BH mass
    rho_ref = 25 # stellar density
    sigma_ref = 200*km_per_s # stellar velocity dispersion
    gal_e = 0.905

    v0_per_sigma = 4.5/2 # initial BH velocity / stellar sigma
    r0_per_rinfl = .5 # initial BH separation/single BH influence radius
    #v0_per_sigma = 2.2 # initial BH velocity / stellar sigma
    #r0_per_rinfl = 2 # initial BH separation/single BH influence radius
    bmin_per_r0,bmax_per_r0 = 5e-3, 1e-1 # impact parameter limits / initial separation

# params matching Hernquist 11-0.825 high_e_no_stars runs fairly well
    #M_BH_ref = 3e-1 # single BH mass
    #rho_ref = 1 # stellar density
    #sigma_ref = 310*km_per_s # stellar velocity dispersion
    #gal_e = 0.91
    #v0_per_sigma = 2.41 # initial BH velocity / stellar sigma
    #r0_per_rinfl = 1 # initial BH separation/single BH influence radius
    #bmin_per_r0,bmax_per_r0 = 1e-3, 1.5e-1 # impact parameter limits / initial separation

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
    #r0 = r0_per_rinfl * sys.r_infl
    r0 = 25e-3
    print(r0*1e3, v0/km_per_s, 2*sys.a_hard*1e3)
    bs = np.linspace(bmin_per_r0, bmax_per_r0, 15)*r0



    efin = []
    deflection_angle = []
    periapsis_angle = []

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
            deflection_angle.append(compute_deflection_angle(sys,x,v))
            periapsis_angle.append(compute_argument_of_periapsis(sys,x,v))
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

    plt.figure()
    plt.plot(np.degrees(deflection_angle), efin, '-')
    plt.xlabel('Deflection angle/deg')
    plt.ylabel('Final e')

    plt.figure()
    plt.plot(np.degrees(deflection_angle), np.degrees(periapsis_angle), '-')
    plt.xlabel('Deflection angle/deg')
    plt.ylabel('Arg periapsis/deg')

    plt.show()




if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("running", sys.argv[1])
        if sys.argv[1] == 'h':
            parameter_space_scan_hernquist_conf()
        if sys.argv[1] == 'g':
            parameter_space_scan_g05_conf()
        if sys.argv[1] == 'g2':
            parameter_space_scan_g05_conf2()
        if sys.argv[1] == 'g3':
            parameter_space_scan_g05_conf3()
    else:
        #high_res_well_fitting_models()
        gen_orbit_plot_data()
        #test_plots()



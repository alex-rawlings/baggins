import argparse
import itertools
import multiprocessing
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import erf

# units Gyr, 1e10 Msol, kpc
G = 44900
km_per_s = 1.02201216 #kpc/Gyr


class System:
    def __init__(self,
                 m1=3e-1,
                 m2=None,
                 e_spheroid=0.9, rho=1.0,
                 stellar_sigma=310,
                 system_rmax = 30,
                 df_fudge_factor=None # TODO remove this
                 ):
        """
        Analytical representation of galaxy potential.

        Parameters
        ----------
        m1 : _type_, optional
            _description_, by default 3e-1
        m2 : _type_, optional
            _description_, by default None
        e_spheroid : float, optional
            _description_, by default 0.9
        rho : float, optional
            _description_, by default 1.0
        stellar_sigma : int, optional
            _description_, by default 310
        system_rmax : int, optional
            _description_, by default 30
        """
        if m2 is None:
            m2 = m1
        self.m1 = max(m1, m2)
        self.m2 = min(m1, m2)
        self.mbin = m1 + m2 # single BH mass
        self.reduced_mass = self.m1 * self.m2 / self.mbin
        self.stellar_sigma = stellar_sigma * km_per_s
         
        e2s = e_spheroid**2
        A1 = (1-e2s)/e2s*(1/(1-e2s) - 1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)))
        A3 = 2*(1-e2s)/e2s*(1/(2*e_spheroid)*np.log((1+e_spheroid)/(1-e_spheroid)) - 1)
        self.Avec = np.array([[A3],[A1]])
        self.rho = rho
        self.b90 = 2 * G * self.m1 / self.stellar_sigma**2
        if df_fudge_factor is not None:
            self.logL = 10 * df_fudge_factor
        else:
            self.logL = np.log(system_rmax / self.b90)
        self.a_hard = G * self.reduced_mass / (4 * self.stellar_sigma**2)
        self.r_infl = np.cbrt(self.mbin/(rho*(4/3*np.pi)))

    def ellipsoid_accel(self, x):
        """
        Potential well modelled by spheroidal potential with some ellipticity

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return -2 * np.pi * G * self.rho * self.Avec * x


    def dynamical_friction(self, x, v):
        """
        Determine binary dynamical friction

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            dynamical friction contribution to acceleration
        """
        # TODO set fraction from mass
        v_single = v * m / self.mbin
        v_single_norm = np.linalg.norm(v_single,axis=0)

        def _df_taylor(m):
            return -(v_single * 8*np.sqrt(np.pi) * m * self.rho * self.logL / (3*np.sqrt(2) * self.stellar_sigma**3))

        def _df_general(m):
            X = v_single_norm/(np.sqrt(2)*self.stellar_sigma) 
            return -(4*np.pi * G**2 * m * self.rho * self.logL * (erf(X) - 2*X/np.sqrt(np.pi)*np.exp(-X**2)) * v_single/v_single_norm**3)

        if v_single_norm/self.stellar_sigma < 1e-3: # taylor expansion for small values
            return _df_taylor(self.m1) + _df_taylor(self.m2)
        else:
            return _df_general(self.m1) + _df_general(self.m2)

    def df_decoupling_factor(self, x, v):
        """
        Decoupling factor as SMBH separation decreases

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            cut-off factor
        """
        a = self.semimajor_axis(x,v)
        a[a<0] = np.inf
        cutoff_point = self.a_hard * 2
        cutoff_scale = self.a_hard * 0.5
        return 1/(1+np.exp(-(a-cutoff_point)/cutoff_scale))


    def accel(self, x, v):
        """
        Determine acceleration

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            acceleration vector
        """
        return (-G * self.mbin/np.linalg.norm(x,axis=0)**3 * x 
                + 2 * self.ellipsoid_accel(x/2)
                + self.dynamical_friction(x,v)*self.df_decoupling_factor(x,v)
                )

    def integrate(self, x0, v0, ts):
        """
        Integrate system acceleration

        Parameters
        ----------
        x0 : array-like
            initial separation
        v0 : array-like
            initial velocity
        ts : array-like
            times for integration sampling

        Returns
        -------
        t : array-like
            times of sampled integrals
        x : array-like
            separations
        v : array-like
            velocities
        """
        def dydt(t,y):
            dvdt = self.accel(y[:2], y[2:])
            return np.append(y[2:], dvdt, axis=0)

        def min_E(t,y):
            E = self.orbital_energy(y[:2], y[2:])
            return E + G*self.mbin/(2*self.a_hard)
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
        """
        Determine orbital energy of system.

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            energy
        """
        r = np.linalg.norm(x,axis=0)
        v2 = np.sum(v*v,axis=0)
        # TODO mbin or M1 or something else?
        return .5 * v2 - G * self.mbin/r
        

    def eccentricity(self, x, v):
        """
        Determine eccentricity of binary.

        Parameters
        ----------
        x : array-like
            relative distance vector between BHs
        v : array-like
            velocity vector

        Returns
        -------
        : array-like
            eccentricities
        """
        # h = L / mu
        h2 = np.cross(x,v, axis=0)**2
        E = self.orbital_energy(x,v)
        return np.sqrt(1 + 2 * h2 * E / (G * self.mbin)**2)
    
    def semimajor_axis(self, x, v):
        return -G*self.mbin/(2*self.orbital_energy(x,v))

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
    return 2*np.arctan(G*sys.mbin/(L*np.sqrt(2*E)))

def compute_argument_of_periapsis(sys, x, v):
    r = np.linalg.norm(x,axis=0)
    rdot = np.sum(x*v,axis=0)/r
    i = np.nonzero(rdot>0)[0][0]
    h = np.cross(x[:,i], v[:,i])
    evec = np.cross(v[:,i], [0,0,h])[:2]/(G*sys.mbin) - x[:,i]/r[i]
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
        sys = System(m1=3e-1,
                     m2=3e-1,
                     rho=1,
                     e_spheroid=gal_e,
                     stellar_sigma=310,
        )


        r0 = sys.r_infl

        v0s = np.linspace(650, 950, 10) * km_per_s
        bs = np.linspace(0.1, 150, 10) * 1e-3
        
        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        
        res = dict(bs=bs*1e3, v0s=v0s/km_per_s, theta=theta, e=efin,
                   e_shperoid=gal_e, df_fudge=df_fudge)

        '''with open(f'toy_data_hernquist.pickle', 'wb') as f:
            pickle.dump(res, f, protocol=-1)'''
        return res

def parameter_space_scan_g05_conf():
# params matching gamma=0.5 e=0.99 runs fairly well
    for gal_e, df_fudge in itertools.product([0.85,0.9],[0.3, 0.5]):
        sys = System(m1=1e-2,
                     m2=1e-2,
                     rho=40,
                     e_spheroid=gal_e,
                     stellar_sigma=200,
                     df_fudge_factor=df_fudge)


        r0 = sys.r_infl

        v0s = np.linspace(350, 600, 10) * km_per_s
        bs = np.linspace(1, 25, 10) * 1e-3
        
        theta, efin = calculate_b_theta_e_curves(sys, r0, v0s, bs)
        
        res = dict(bs=bs*1e3, v0s=v0s/km_per_s, theta=theta, e=efin,
                   e_shperoid=gal_e, df_fudge=df_fudge)

        with open(f'data/g05_b_v_scan_es_{gal_e:.2f}_df_{df_fudge:.1f}_general.pkl', 'wb') as f:
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
    bs = np.append(np.geomspace(1e-3,0.1,5)[:-1], np.geomspace(0.1, 20, 500)) * 1e-3
    e_spheroids = [0.9, 0.91, 0.92]


    v0s = [450 * km_per_s, 560 * km_per_s]
    res090 = []
    res099 = []
    for e_s in e_spheroids:
    #e=0.9, 0.99 mergers are well fitted by this system, and the parameters are pretty similar to the sims as well
        sys = System(M_BH=1e-2,
                     rho=30,
                     e_spheroid=e_s,
                     stellar_sigma=200*km_per_s,
                     df_fudge_factor=0.47)

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
                     df_fudge_factor=0.47)
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
    rho_ref = 30 # stellar density
    sigma_ref = 200*km_per_s # stellar velocity dispersion
    gal_e = 0.905

    v0_per_sigma = 5.5/2 # initial BH velocity / stellar sigma
    r0_per_rinfl = .5 # initial BH separation/single BH influence radius
    #v0_per_sigma = 2.2 # initial BH velocity / stellar sigma
    #r0_per_rinfl = 2 # initial BH separation/single BH influence radius
    bmin_per_r0,bmax_per_r0 = 3e-2, 8e-2 # impact parameter limits / initial separation

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
                 df_fudge_factor=0.47)


    v0 = v0_per_sigma * sigma_ref * vscale
    #r0 = r0_per_rinfl * sys.r_infl
    r0 = 25e-3
    print(r0*1e3, v0/km_per_s, 2*sys.a_hard*1e3)
    bs = np.linspace(bmin_per_r0, bmax_per_r0, 25)*r0



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


def th_e_curves(data):
    from matplotlib.collections import LineCollection

    fig = plt.figure()
    lines = LineCollection(list(map(np.column_stack, zip(np.degrees(data['theta']), data['e']))), array=data['v0s'])
    plt.gca().add_collection(lines)
    plt.xlim(0,180)
    cb = fig.colorbar(lines)
    cb.set_label('$v_0/\mathrm{km\,s^{-1}}$')

    plt.ylabel(r'$e_\mathrm{h}$')
    plt.xlabel(r'$\theta$')
    plt.gca().xaxis.set_major_formatter('{x:.0f}°')
    plt.savefig("~/Desktop/th_e_curve.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute theoretical relation between impact parameter and eccentricity", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--scan", type=str, choices=["h", "g1", "g2", "g3"], default=None, help="perform parameter scan", dest="scan")
    parser.add_argument("-p", "--plot", action="store_true", help="plot well fitting models", dest="plot")
    args = parser.parse_args()

    if args.scan is not None:
            if args.scan == "h":
                data = parameter_space_scan_hernquist_conf()
            elif args.scan == "g1":
                data = parameter_space_scan_g05_conf()
            elif args.scan == "g2":
                data = parameter_space_scan_g05_conf2()
            elif args.scan == "g3":
                data = parameter_space_scan_g05_conf3()
            th_e_curves(data)
    if args.plot:
        #high_res_well_fitting_models()
        gen_orbit_plot_data()


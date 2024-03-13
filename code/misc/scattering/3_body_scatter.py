import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.stats
import ketjugw
import baggins as bgs
from ketjugw.units import pc, yr, km_per_s



class AnalyticalGalaxy:
    def __init__(self, gamma, ey, ez, Mgal, r_tilde=3e4*pc) -> None:
        self.gamma = gamma
        self.ey = ey
        self.ez = ez
        self.r_tilde = r_tilde * pc
        self.rho_tilde = (3-self.gamma)*Mgal /(4*np.pi*self.r_tilde**3) # set mass in r_tilde to be Mg for spherical system
    
    def accel_from_gal(self, pos):
        r = np.linalg.norm(pos)
        #units where G=1
        return -4*np.pi/(3-self.gamma) * self.rho_tilde * self.r_tilde * (r/self.r_tilde)**(1-self.gamma) / r * pos * (
            1 - self.gamma*(self.ez*pos[2]**2+self.ey*pos[1]**2)/(r**2 * (2-self.gamma))
            + 2/(2-self.gamma) * np.array([0, self.ey, self.ez])
        )
    
    def dyn_fric(self, v, sep):
        # TODO this may need some work...
        #return -v / (5e6*yr) * 1/(1+np.exp(-sep/(200*pc)))
        return -v / (5e8*yr) * 1/(1+np.exp(-sep/(200*pc)))
    
    def total_accel(self, pos, vel, use_df=False, use_galacc=False):
        a = 0
        if use_df:
            # TODO r is sep between BHs, or sep from centre??
            a += self.dyn_fric(vel, np.linalg.norm(pos))
        if use_galacc:
            a += self.accel_from_gal(pos)
        return a
    

class GeneralSystem(AnalyticalGalaxy):
    def __init__(self, gamma, ey, ez, Mgal, r_tilde=3e3) -> None:
        super().__init__(gamma, ey, ez, Mgal, r_tilde)
        self.bh_list = []
        self.initial_kepler_params = {}
    
    def _add_particle(self, m, x, v):
        return ketjugw.Particle(0, m, x*pc, v*km_per_s)
    
    def add_BH(self, m, x, v):
        assert len(self.bh_list)<3
        self.bh_list.append(
            self._add_particle(m=m, x=x, v=v)
        )
    
    def setup_kepler_orbit_bhs(self, ic):
        ic.setdefault("e0", 0)
        ic.setdefault("l0", np.pi/2)
        self.initial_kepler_params = ic
        bh1, bh2 = ketjugw.keplerian_orbit(m1=ic["m1"], m2=ic["m2"], a=ic["a0"]*pc, e=ic["e0"], l=ic["l0"], ts=0)
        self.bh_list.extend([bh1, bh2])
    
    def save(self):
        raise NotImplementedError
    
    def _plot_binary(self, bh1, bh2, figax1=None, figax2=None, label=None):
        alpha = 0.7
        suptitle = f"Potential: e$_{{y}}$={self.ey:.1f} e$_{{z}}$={self.ez:.1f}"
        # set up the figures
        if figax1 is None:
            bind_figax1 = True
            fig1, ax1 = plt.subplots(1,2,sharex="all")
            for axi, l in zip(ax1, ("y", "z")): 
                axi.set_xlabel("x/pc")
                axi.set_ylabel(f"{l}/pc")
            fig1.suptitle(suptitle)
        else:
            bind_figax1 = False
            ax1 = figax1[1]
        if figax2 is None:
            bind_figax2 = True
            fig2, ax2 = plt.subplots(3,1,sharex="all")
            ax2[-1].set_xlabel("t/yr")
            ax2[0].set_ylabel("a/pc")
            ax2[1].set_ylabel("e")
            ax2[-1].set_ylabel("1-e")
            ax2[1].set_ylim(0,1)
            ax2[-1].set_ylim(1e-8,1)
            fig2.suptitle(suptitle)
        else:
            bind_figax2 = False
            ax2 = figax2[1]
        
        # plot the orbits
        l = ax1[0].plot(bh1.x[:,0]/pc, bh1.x[:,1]/pc, markevery=[-1], marker="o", alpha=alpha)
        ax1[1].plot(bh1.x[:,0]/pc, bh1.x[:,2]/pc, markevery=[-1], marker="o", alpha=alpha)
        ax1[0].plot(bh2.x[:,0]/pc, bh2.x[:,1]/pc, markevery=[-1], marker="o", alpha=alpha, c=l[-1].get_color(), ls="--")
        ax1[1].plot(bh2.x[:,0]/pc, bh2.x[:,2]/pc, markevery=[-1], marker="o", alpha=alpha, c=l[-1].get_color(), ls="--")

        # determine and plot the orbit parameters
        bbh = ketjugw.find_binaries([bh1,bh2], remove_unbound_gaps=True)
        try:
            pars = ketjugw.orbital_parameters(*bbh[(0,1)])
            ax2[0].semilogy(pars["t"]/yr, pars["a_R"]/pc, alpha=alpha)
            ax2[1].plot(pars["t"]/yr, pars["e_t"], alpha=alpha, label=label)
            ax2[-1].semilogy(pars["t"]/yr, 1-pars["e_t"], alpha=alpha, label=label)
        except KeyError:
            print("Binary not found")
            print(bbh)
            pass
            pars = None
        if bind_figax1: figax1 = [fig1, ax1]
        if bind_figax2: figax2 = [fig2, ax2]
        return figax1, figax2



class TwoBodySystem(GeneralSystem):
    def __init__(self, gamma, ey, ez, Mgal, r_tilde=3000) -> None:
        super().__init__(gamma, ey, ez, Mgal, r_tilde)
        self._bh_list_original = []
    
    @property
    def times(self):
        return self._times
    
    @times.setter
    def times(self, dt_tf):
        assert len(dt_tf)==2
        self._times = np.arange(dt_tf[0]*yr, dt_tf[1]*yr, dt_tf[0]*yr)
    
    def _make_ketjugw_particles(self, i, ts, vals):
        xs = vals[:, 3*i:3*(i+1)]
        vs = vals[:, 6+3*i:6+3*(i+1)]
        return ketjugw.Particle(ts, self.bh_list[i].m[0], xs, vs)
    
    def add_BH(self, m, x, v):
        super().add_BH(m, x, v)
        self._bh_list_original.append(self.bh_list[-1])
    
    def setup_kepler_orbit_bhs(self, ic):
        super().setup_kepler_orbit_bhs(ic)
        self._bh_list_original.extend(self.bh_list)

    def _motion_derivatives(self, t, posvel):
        x = posvel[:6]
        v = posvel[6:]
        r_vec = x[:3] - x[3:]
        r = np.linalg.norm(r_vec)
        a0_point_mass = -self.bh_list[0].m * r_vec / r**3
        a1_point_mass = self.bh_list[1].m * r_vec / r**3
        a0 = a0_point_mass + self.total_accel(x[:3], v[3:])
        a1 = a1_point_mass + self.total_accel(x[3:], v[:3])
        return np.append(v, np.array([a0, a1]))

    def integrate_system(self):
        init_conds = np.concatenate((self.bh_list[0].x, self.bh_list[1].x, self.bh_list[0].v, self.bh_list[1].v)).flatten()
        print("Integrating system...")
        res = solve_ivp(self._motion_derivatives, (0, self.times[-1]), y0=init_conds, t_eval=self.times, rtol=1e-8, atol=1e-20)
        print("Integration complete")
        return res.t, res.y.T
    
    def plot_system(self, figax1=None, figax2=None, label=None):
        ts, vals = self.integrate_system()
        bh1 = self._make_ketjugw_particles(0, ts, vals)
        bh2 = self._make_ketjugw_particles(1, ts, vals)
        return self._plot_binary(bh1, bh2, figax1=figax1, figax2=figax2, label=label)
    
    def restore_all_particles(self):
        self.bh_list = []
        self.bh_list = copy.deepcopy(self._bh_list_original)
    
    def print(self, i):
        print(f"    {self.bh_list[i].x[0]/pc}")
        print(f"    {self.bh_list[i].v[0]/km_per_s}")
    
    def perturb_particle(self, p, c, val, add=False, verbose=False):
        if verbose:
            print("\nPerturbation\n------------")
            print("  Original:")
            self.print(p)
        if not add:
            self.restore_all_particles()
        crds = {"x":0, "y":1, "z":2, "vx":0, "vy":1, "vz":2}
        if "v" in c:
            self.bh_list[p].v[0][crds[c]] += (val*km_per_s)
        else:
            self.bh_list[p].x[0][crds[c]] += (val*pc)
        if verbose:
            print("  New:")
            self.print(p)


class ThreeBodySystem(GeneralSystem):
    def __init__(self, gamma, ey, ez, Mgal, r_tilde=3000, rng=None) -> None:
        super().__init__(gamma, ey, ez, Mgal, r_tilde)
        self.star_list = []
        self.times = []
        self.rng = rng
        # TODO may need thinking, good first approximation
        self.v_Vc_dist = scipy.stats.maxwell()
        self.rp_dist = scipy.stats.uniform(
                            loc=0,
                            scale=5*self.initial_kepler_params["a0"]
                    )
        self.orientation_dist = scipy.stats.uniform(loc=0, scale=2*np.pi)
    
    def add_star(self, mstar):
        # set up following
        # https://iopscience.iop.org/article/10.1086/507596/pdf
        if self.times == []:
            # no integration has been done yet
            a = self.initial_kepler_params["a0"]
            m1, m2 = self.bh_list[0].m, self.bh_list[1].m
        else:
            # TODO this may need some updating once the integration part is 
            # written
            orbit_pars = ketjugw.orbital_parameters(*self.bh_list)
            a = orbit_pars["a_R"]
            m1, m2 = orbit_pars["m1"], orbit_pars["m2"]
        M = m1 + m2
        mu = m1*m2/M
        Vc = np.sqrt(M/a)
        rp = self.rp_dist.rvs(random_state=self.rng)
        v0 = self.v_Vc_dist.rvs(random_state=self.rng)
        # impact parameter
        b = rp * np.sqrt(1 + 2 * M / (rp * (v0*Vc)**2))
        # velocity angles
        theta, phi = bgs.mathematics.uniform_sample_sphere(1, rng=rng)
        # initial displacement of star
        ri = (1e10 * mu/M)**0.25 * a
        

    
    def add_star(self, m, x, v):
        pass
        """self.star_list.append(
            self._add_particle(m=m, x=x, v=v)
        )"""
    
    def integrate_with_ketju(self):
        raise NotImplementedError




if __name__ == "__main__":
    # Give all quantities in terms of Msol, yr, pc, km/s -> conversions done 
    # internally
    Mgal = 1e11
    ey = 0.0
    ez = 0.0
    gamma = 1.2
    ic_orbit = {"m1":1e9, "m2":1e9, "a0":100, "e0":0.2}#[1e9, 1e9, 100, 0.2]

    rng = np.random.default_rng(42)

    galsystem = TwoBodySystem(gamma=gamma, ey=ey, ez=ez, Mgal=Mgal)
    galsystem.times = [1e4, 1e8]
    galsystem.setup_kepler_orbit_bhs(*ic_orbit)
    figax1, figax2 = galsystem.plot_system()
    figax1[1][0].set_aspect("equal")
    if False:
        # perturb
        for i in range(3):
            galsystem.restore_all_particles()
            for j in range(2):
                for c in ("x", "y", "z", "vx", "vy", "vz"):
                    galsystem.perturb_particle(j, c, rng.normal(0, 10), add=True)
                galsystem.print(j)
            galsystem.plot_system(figax1=figax1, figax2=figax2)
    if False:
        for ezi in np.arange(0.4, 0.9, 0.1):
            galsystemi = TwoBodySystem(gamma=gamma, ey=ezi, ez=ez, Mgal=Mgal)
            galsystemi.times = [1e4, 1e8]
            galsystemi.setup_kepler_orbit_bhs(*ic_orbit)
            galsystemi.plot_system(figax1=figax1, figax2=figax2, label=f"$e_{{y}}$={ezi:.1f}")
    figax2[1][-1].legend()
    plt.show()

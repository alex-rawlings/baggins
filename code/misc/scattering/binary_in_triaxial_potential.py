import copy
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ketjugw
from ketjugw.units import pc, yr, km_per_s
import baggins as bgs


bgs.plotting.check_backend()

class FrozenParticle:
    def __init__(self, mass, pos, vel, ptype) -> None:
        self.mass = mass
        self.pos = np.array(pos)*pc
        self.vel = np.array(vel)*km_per_s
        self.ptype = ptype
    
    @classmethod
    def centre_mass(cls, p1, p2):
        def _mass_wgt_mean(q1, m1, q2, m2):
            return (q1*m1 + q2*m2) / (m1+m2)
        p = cls(
                p1.mass + p2.mass,
                _mass_wgt_mean(p1.pos, p1.mass, p2.pos, p2.mass),
                _mass_wgt_mean(p1.vel, p1.mass, p2.vel, p2.mass),
                f"{p1.ptype}+{p2.ptype}"
        )
        return p

    def print(self):
        print(f"FrozenParticle Object {self.ptype}")
        print(f"  Mass: {self.mass:.2e}")
        print(f"  Pos: {self.pos/pc}")
        print(f"  Vel: {self.vel/km_per_s}")


class StaticPotential:
    def __init__(self, gamma, ey, ez, Mgal, r_tilde=3e3*pc) -> None:
        self.gamma = gamma
        self.ey = ey
        self.ez = ez
        self.r_tilde = r_tilde
        self.rho_tilde = (3-self.gamma)*Mgal /(4*np.pi*self.r_tilde**3) # set mass in r_tilde to be Mg for spherical system
        self.particles = []
        self._particles_original = []
        self.particle_count = 0
    
    @property
    def times(self):
        return self._times
    
    @times.setter
    def times(self, dt_tf):
        try:
            if self._time_set:
                raise ValueError("System integration time already set")
        except AttributeError:
            self._time_set = True
        self._times = np.arange(dt_tf[0]*yr, dt_tf[1]*yr, dt_tf[0]*yr)

    def add_particle(self, mass, pos, vel, ptype="BH"):
        self.particles.append(
            FrozenParticle(mass=mass, pos=pos, vel=vel, ptype=ptype)
        )
        self._particles_original.append(
            FrozenParticle(mass=mass, pos=pos, vel=vel, ptype=ptype)
        )
        self.particle_count += 1
    
    def delete_all_particles(self):
        self.particle_count = 0
        self.particles = []
    
    def perturb_particle(self, p, i, v):
        self.delete_all_particles()
        self.particles = copy.deepcopy(self._particles_original)
        self.particle_count = len(self.particles)
        if i < 3:
            self.particles[p].pos[i] += (v*pc)
        else:
            self.particles[p].vel[i-3] += (v*km_per_s)
        self.particles[p].print()
    
    def set_up_kepler(self, m1, m2, a0, e0, l0=np.pi):
        p1, p2 = ketjugw.keplerian_orbit(m1, m2, a0, e0, l0, 0)
        self.add_particle(m1, p1.x.flatten()/pc, p1.v.flatten()/km_per_s)
        self.add_particle(m2, p2.x.flatten()/pc, p2.v.flatten()/km_per_s)
        for p in self.particles: p.print()
    
    def accel_from_gal(self, pos):
        r = np.linalg.norm(pos)
        #units where G=1
        return -4*np.pi/(3-self.gamma) * self.rho_tilde * self.r_tilde * (r/self.r_tilde)**(1-self.gamma) / r * pos * (
            1 - self.gamma*(self.ez*pos[2]**2+self.ey*pos[1]**2)/(r**2 * (2-self.gamma))
            + 2/(2-self.gamma) * np.array([0, self.ey, self.ez])
        )
    
    def dyn_fric(self, v, sep):
        # TODO this may need some work...
        #return -v / (5e8*yr) * 1/(1+np.exp(-sep/(1200*pc)))
        return -v / (5e7*yr) * 1/(1+np.exp(-sep/(400*pc)))
    
    def accel(self, pos, vel, df=False, gf=False):
        assert self.particle_count == 2
        rv = pos[:3] - pos[3:]
        r = np.linalg.norm(rv)
        accel0 = -self.particles[0].mass * rv/r**3
        accel1 = self.particles[1].mass * rv/r**3
        if df:
            # use dynamical friction
            accel0 += self.dyn_fric(vel[:3], np.linalg.norm(pos[:3]))
            accel1 += self.dyn_fric(vel[3:], np.linalg.norm(pos[3:]))
        if gf:
            # use background acceleration from stars
            accel0 += self.accel_from_gal(pos[:3])
            accel1 += self.accel_from_gal(pos[3:])
        return np.append(accel0, accel1)
    
    def motion_derivative(self, t, posvel):
        idx = self.particle_count * 3
        return np.append(posvel[idx:], self.accel(pos=posvel[:idx], vel=posvel[idx:]))
    
    def integrate_orbit(self):
        print("Integrating orbits...")
        ics = np.concatenate((self.particles[0].pos, self.particles[1].pos, self.particles[0].vel, self.particles[1].vel))
        res = solve_ivp(self.motion_derivative, (0, self.times[-1]), y0=ics, method="DOP853", t_eval=self.times, rtol=1e-8, atol=1e-20)
        print("Integration complete")
        return res.t, res.y.T
    
    def integrate_with_ketju(self):
        raise NotImplementedError
        prop_kwargs = {"use_spin":False, "progress_report_interval":1000, "PN_level":7, "tol":1e-9}
        ketjugw.chain_bindings.propagate_bh_binary_chain(
            self.particles[0], self.particles[1], 
        )
    
    def _make_ketju_particle(self, i, ts, vals):
        idx0p = 3*i
        idx1p = 3*(i+1)
        idx0v = 3*self.particle_count+idx0p
        idx1v = 3*self.particle_count+idx1p
        #print(f"Pos: {idx0p}-{idx1p}, Vel: {idx0v}-{idx1v}")
        xs = vals[:, idx0p:idx1p]
        vs = vals[:, idx0v:idx1v]
        return ketjugw.Particle(ts, np.full_like(ts, self.particles[i].mass),
                                xs, vs
                                )
    
    def plot(self, figax1=None, figax2=None, figax3=None, label=None, pn=0):
        times, vals = self.integrate_orbit()
        bh1 = self._make_ketju_particle(0, times, vals)
        bh2 = self._make_ketju_particle(1, times, vals)
        if figax1 is None:
            bind_figax1 = True
            fig1, ax1 = plt.subplots(1,2,sharex="all")
            for axi, l in zip(ax1, ("y", "z")): 
                axi.set_xlabel("x/pc")
                axi.set_ylabel(f"{l}/pc")
        else:
            bind_figax1 = False
            ax1 = figax1[1]
        l = ax1[0].plot(bh1.x[:,0]/pc, bh1.x[:,1]/pc, markevery=[-1], marker="o", alpha=0.8)
        ax1[1].plot(bh1.x[:,0]/pc, bh1.x[:,2]/pc, markevery=[-1], marker="o", alpha=0.8)
        ax1[0].plot(bh2.x[:,0]/pc, bh2.x[:,1]/pc, markevery=[-1], marker="o", alpha=0.8, c=l[-1].get_color(), ls="--")
        ax1[1].plot(bh2.x[:,0]/pc, bh2.x[:,2]/pc, markevery=[-1], marker="o", alpha=0.8, c=l[-1].get_color(), ls="--")
        if figax2 is None:
            bind_figax2 = True
            fig2, ax2 = plt.subplots(3,1,sharex="all")
            ax2[-1].set_xlabel("t/yr")
            ax2[0].set_ylabel("a/pc")
            ax2[1].set_ylabel("e")
            ax2[-1].set_ylabel("1-e")
            ax2[1].set_ylim(0,1)
            ax2[-1].set_ylim(1e-8,1)
        else:
            bind_figax2 = False
            ax2 = figax2[1]
        if figax3 is None:
            bind_figax3 = True
            fig3, ax3 = plt.subplots(1,1)
            ax3.set_xlabel("t/yr")
            ax3.set_ylabel("E")
            ax3.axhline(0, c="k", ls=":", alpha=0.7)
        else:
            bind_figax3 = False
            ax3 = figax3[1]
        bbh = ketjugw.find_binaries([bh1,bh2], remove_unbound_gaps=False)
        try:
            energy = ketjugw.orbital_energy(bh1, bh2)
            #energy = ketjugw.orbital_energy(*bbh[(0,1)])
            pars = ketjugw.orbital_parameters(*bbh[(0,1)], PN_level=pn)
            ax2[0].semilogy(pars["t"]/yr, pars["a_R"]/pc, alpha=0.7)
            ax2[1].plot(pars["t"]/yr, pars["e_t"], alpha=0.7, label=label)
            ax2[-1].semilogy(pars["t"]/yr, 1-pars["e_t"], alpha=0.7, label=label)
            ax3.plot(bh1.t/yr, energy)
            #ax3.plot(bbh[(0,1)][0].t/yr, energy)
            energy_positive = energy > 0
            #ax3.scatter(bbh[(0,1)][0].t[energy_positive]/yr, energy[energy_positive])
        except KeyError:
            print("Binary not found")
            print(bbh)
            pass
            pars = None
        if bind_figax1: figax1 = [fig1, ax1]
        if bind_figax2: figax2 = [fig2, ax2]
        if bind_figax3: figax3 = [fig3, ax3]
        return figax1, figax2, figax3


if __name__ == "__main__":
    Mgal = 1e11
    ey = 0.2
    ez = 0.9
    gamma = 1.2
    kepler_initial_e = 0.8
    perturb_crd = 0

    gal = StaticPotential(gamma, ey, ez, Mgal)
    gal.times = [1e4, 1.0e8]
    if True:
        # set up BHs on Kepler orbit
        # pos in pc, mass in Msun
        m1 = 1e9
        m2 = 1e9
        Rs = 2*(m1+m2)
        gal.set_up_kepler(m1, m2, 100*pc, kepler_initial_e, np.pi/2)
    else:
        gal.add_particle(1e9, [200, 100, 10], [20, 10, 0])
        gal.add_particle(1e9, [-200, -100, -10], [-20, -10, 0])
    figax1, figax2, figax3 = gal.plot(label="0.00")

    if False:
        # perturb
        perturbs = (-10.0, -5.0, 5.1, 10.1)
        """if perturb_crd < 3:
            perturbs = (-10.0, -5.0, 5.1, 10.1)
        else:
            perturbs = (-5.0, -2.0, 2.1, 5.1)"""
        for i, xperturb in enumerate(perturbs):
            print(f"\nPerturbation: {i}\n---------------")
            gal.perturb_particle(0, perturb_crd, xperturb)
            gal.plot(figax1=figax1, figax2=figax2, figax3=figax3, label=f"{xperturb:.1f}")
        figax2[1][1].legend()
    perturb_crd_str = ["x", "y", "z", "vx", "vy", "vz"]
    fig_name_base = f"scatter/S-{gal.ey:.1f}-{gal.ez:.1f}-e-{kepler_initial_e:.1f}-crd-{perturb_crd_str[perturb_crd]}"
    figax1[0].suptitle(fig_name_base.split("/")[-1])
    figax2[0].suptitle(fig_name_base.split("/")[-1])
    #figax1[0].savefig(os.path.join(bgs.FIGDIR, f"{fig_name_base}-orbit.png"))
    #figax2[0].savefig(os.path.join(bgs.FIGDIR, f"{fig_name_base}-pars.png"))
    bgs.plotting.savefig("orbit.png", figax1[0])
    plt.show()




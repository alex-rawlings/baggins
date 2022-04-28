import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import ketjugw
from ketjugw.units import pc, yr, km_per_s

#switch to turn on/off ad-hoc dynamical friction
use_df = True

M1 = 1e9
M2 = 1e9
# y is [x1,y1,x2,y2,v1,u1,v2,u2]

# from Nasim+2021
g = 1
r_tilde = 3e3*pc
Mg = 1e11
rho_tilde = (3-g)*Mg /(4*np.pi*r_tilde**3) # set mass in r_tilde to be Mg for spherical system
e_x = 0.5
e_y = 0.5

def gal_acc(x):
    # x: array of [xpos, ypos]
    r = np.linalg.norm(x)
    return -4*np.pi/(3-g)* rho_tilde * r_tilde * (r/r_tilde)**(1-g) * (
                 x/r * (1 + (e_x * x[0]**2 + e_y * x[1]**2)/r**2)
                + r/(2-g) * (-2*x*(e_x * x[0]**2 + e_y * x[1]**2)/r**4 + 2*np.array([e_x*x[0], e_y*x[1]])/r**2))

def df(m, v,r):
    return -v / (5e6*yr) * 1/(1+np.exp(-r/(200*pc)))

def acc(y, use_df=True):
    x1 = y[:2]
    x2 = y[2:4]
    rv = x1-x2
    r = np.linalg.norm(rv)

    if use_df:
        return np.append(-M2*rv/r**3 + gal_acc(x1) + df(M1, y[4:6], r), M1*rv/r**3 + gal_acc(x2) + df(M2, y[6:8], r))
    else:
        return np.append(-M2*rv/r**3 + gal_acc(x1), M1*rv/r**3 + gal_acc(x2))

def der(t,y):
    return np.append(y[4:], acc(y, use_df=use_df))

def integrate(y0, t_target, dt):
    res = solve_ivp(der, (0,t_target), y0, method='DOP853',
                    t_eval=np.arange(dt,t_target, dt), rtol=1e-8, atol=1e-20)
    return res.t, res.y.T



def plot_res(ts, ys, label):
    x1 = ys[:,:2]
    x2 = ys[:,2:4]
    v1 = ys[:,4:6]
    v2 = ys[:,6:8]
    plt.figure(1)
    plt.plot(*x1.T/pc, label=label[0])
    plt.plot(*x2.T/pc, label=label[1])

    plt.gca().set_aspect('equal')
    plt.xlabel('x/pc')
    plt.ylabel('y/pc')
    plt.legend()

    plt.figure(2)
    L = np.cross(x1-x2, v1-v2)
    plt.plot(ts/yr, L, label=label[0])
    plt.xlabel('t/yr')
    plt.ylabel('L')
    plt.legend()

    plt.figure(3)
    bh1 = ketjugw.Particle(ts, np.full_like(ts, M1),
                           np.concatenate((x1, np.zeros_like(ts)[:,np.newaxis]), axis=-1),
                           np.concatenate((v1, np.zeros_like(ts)[:,np.newaxis]), axis=-1),
                           )
    bh2 = ketjugw.Particle(ts, np.full_like(ts, M2),
                           np.concatenate((x2, np.zeros_like(ts)[:,np.newaxis]), axis=-1),
                           np.concatenate((v2, np.zeros_like(ts)[:,np.newaxis]), axis=-1),
                           )
    bbh = ketjugw.find_binaries([bh1,bh2])
    pars = ketjugw.orbital_parameters(*bbh[(0,1)], PN_level=0)
    plt.plot(pars['t']/yr, pars['e_t'], label=label[0])
    plt.xlabel('t/yr')
    plt.ylabel('e')
    plt.legend()


ic = [100*pc,10*pc, -100*pc, -10*pc, -400*km_per_s, 0*km_per_s, 400*km_per_s, 0*km_per_s]

tspan = 2e7 * yr
dt = 1e4 * yr

plot_res(*integrate(ic, tspan, dt), label=["Orig1", "Orig2"])
print("System 1 done")

ic[-1] += 10*km_per_s
plot_res(*integrate(ic, tspan, dt), label=["Perturb1_1", "Perturb1_2"])
print("System 2 done")

#ic[-1] -= 10*km_per_s
#ic[0] += 10*pc
#plot_res(*integrate(ic, tspan, dt), label=["Perturb2_1", "Perturb2_2"])

plt.show()

    
    
    

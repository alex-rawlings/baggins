import numpy as np
from scipy.special import erf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# assume Dehnen profile
G = 43007  # km/s, kpc, 1e10Msol, Gyr


def enclosed_mass(r, g, a, M):
    return M * (r / (r + a)) ** (3 - g)

def density(r, g, a, M):
    return M * (3 - g) * a / (4 * np.pi * r**g * (r + a) ** (4 - g))

def eff_rad(g, a):
    return 0.75 * a / (2**(1/(3-g)) - 1)

def sigma(g, a, M):
    def _sigma(r, a, M, aa, bb):
        return G * M / a * ((r/a)**aa) / (1 + (r/a))**bb
    if g==0.5:
        aa = 0.8
        bb = 1.7
    elif g==1:
        aa = 1
        bb = 2
    elif g==1.2:
        aa = 1.1
        bb = 2.2
    elif g==1.5:
        aa = 1.25
        bb = 2.5
    elif g==1.75:
        aa = 1.35
        bb = 2.8
    else:
        raise RuntimeError("Invalid gamma")
    return lambda r: _sigma(r, a=a, M=M, aa=aa, bb=bb)


def solve_motion(x0, v0, t_eval, pars, Mbh):
    lnL = 1
    def system(t, y):
        x1, x2 = y
        xx = x2 / (np.sqrt(2)*sigma(*pars)(x1))
        dx1dt = x2
        dx2dt = -G * enclosed_mass(x1, *pars) / x1**2 - (
            4 * np.pi * G**2 * lnL * density(x1, *pars) * Mbh / x2**2 * (erf(xx) - 2*xx/np.sqrt(np.pi) * np.exp(-xx**2))
        )
        return [dx1dt, dx2dt]

    y0 = [x0, v0]
    t_span = (t_eval[0], t_eval[-1])
    
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45', vectorized=False)
    
    return sol.t, sol.y[0]


if __name__ == "__main__":
    t_points = np.linspace(0, 1, 1000)
    Mtot = 1.38e11*2/1e10
    Mbh = 1e-2
    a = 3.9
    for g in (0.5, 1.0, 1.2):
        print(f"Doing {g}")
        t, X = solve_motion(1e-3, 800, t_points, [g, a, Mtot], Mbh)
        plt.semilogy(t, X, label=f"{g:.1f}")
        reff = eff_rad(g=g, a=a)
        print(f"Eff. radius: {reff:.2e}")
        print(f"Apo: {np.max(X):.2e}")
        print(f"Apo / Eff. radius: {np.max(X)/reff:.2e}")
    plt.ylim(0.1, None)
    plt.legend(title=r"$\gamma$")
    plt.grid()
    plt.savefig("apo_ics.png", dpi=300)

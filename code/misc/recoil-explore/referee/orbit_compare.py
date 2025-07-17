import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pygad
import baggins as bgs

# use kpc, km/s, 1e10Msol
G = 43007

def solve_motion(A, B, X0, V0, t_eval):
    """
    Solves the second-order ODE: X'' + A*X' + B*X = 0
    Args:
        A (float): damping coefficient
        B (float): stiffness coefficient
        X0 (float): initial position X(0)
        V0 (float): initial velocity X'(0)
        t_eval (array-like): times at which to evaluate X(t)
    Returns:
        t (np.ndarray): times
        X (np.ndarray): positions at those times
    """
    def system(t, y):
        x1, x2 = y
        dx1dt = x2
        dx2dt = -A * x2 - B * x1
        return [dx1dt, dx2dt]

    y0 = [X0, V0]
    t_span = (t_eval[0], t_eval[-1])
    
    sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45', vectorized=False)
    
    return sol.t, sol.y[0]

def Tdf(dens, sig, lnL, Mbh):
    return 3 / 8 * np.sqrt(2 / np.pi) * sig**3 / (G**2 * dens * Mbh * lnL)

def wc(dens):
    return np.sqrt(4 * np.pi / 3 * G * dens)

# Example usage
if __name__ == "__main__":
    core_rad = 0.58 # kpc
    kickvel = 60 # km/s
    snap = pygad.Snapshot("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0000/output/snap_003.hdf5", physical=True)
    bgs.analysis.basic_snapshot_centring(snap)
    core_snap = snap[pygad.BallMask(core_rad)]
    core_sig = np.nanstd(core_snap["vel"])
    print(f"Core sigma; {core_sig}")
    dens = bgs.mathematics.density_sphere(np.sum(core_snap.stars["mass"].view(np.ndarray))/1e10 + np.sum(core_snap.dm["mass"].view(np.ndarray))/1e10, core_rad)
    print(f"Dens is: {dens}")
    wc_val = wc(dens)

    for lnL in (0.1, 0.5, 0.8, 1, 1.5, 2):
        print(f"Doing coulomb logarithm {lnL}")
        Tdf_val = Tdf(dens, core_sig, lnL, snap.bh["mass"][0].view(np.ndarray)/1e10)
        X0 = 0.0  # initial position
        t_points = np.linspace(0, 0.5, 500)

        t, X = solve_motion(1/Tdf_val, wc_val**2, X0, kickvel, t_points)

        # Optional: plot the result
        plt.plot(t, X, label=rf"ln$\Lambda$={lnL:.1f}", zorder=0.5)
    plt.xlabel('t/Gyr')
    plt.ylabel('X(t)/kpc')
    plt.title('Damped Oscillator: $X\'\' + A X\' + B X = 0$')
    #plt.grid(zorder=0.1)

    # add ketju data
    kf = bgs.utils.get_ketjubhs_in_dir(f"/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-{kickvel:04d}/")[0]
    bh = bgs.analysis.get_bh_after_merger(kf)
    bh.t /= bgs.general.units.Gyr
    bh.t -= bh.t[0]
    bh.x /= bgs.general.units.kpc
    bh.x -= bh.x[0,:]
    plt.plot(bh.t, bh.x[:,0], label="simulation", lw=2, zorder=0.2)
    plt.axhline(core_rad, c="k", label="core radius")
    plt.legend()
    bgs.plotting.savefig("analytical_orbit.png")

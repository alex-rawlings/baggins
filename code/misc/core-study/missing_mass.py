from tqdm import tqdm
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import baggins as bgs


datafile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"

'''data = bgs.utils.load_data(datafile)


def data_yielder():
    for k in data["rb"].keys():
        d = {
            "rb": data["rb"][k].flatten(),
            "Re": data["Re"][k].flatten(),
            "n": data["n"][k].flatten(),
            "log10densb": data["log10densb"][k].flatten(),
            "g": data["g"][k].flatten(),
            "a": data["a"][k].flatten()
        }
        yield k, d'''

def sersic_b(n):
    return 2.0 * n - 0.33333333 + 0.009876 / n

def sersic_fit(r, Re, n):
    """
    :param r: radius
    :param Re: effective radius
    :param n: Sersic index
    :return: projected surface mass density
    """
    b = sersic_b(n)
    mu = np.exp(-b * ((r / Re) ** (1 / n) - 1))
    return mu

def core_sersic(r, rb, Re, n, log10densb, g, a, edit=False):
    b = sersic_b(n)
    preterm = - g / a * np.log(2.0) + b * pow((pow(2.0, 1/a) * rb / Re), 1/n)
    if edit:
        rb=0
    dens = log10densb + (preterm + g / a * np.log(pow(r, a) + pow(rb, a)) - g * np.log(r) - b * pow(Re, -1/n) * pow((pow(r, a) + pow(rb, a)), (1/(a*n)))) / np.log(10.0)
    return 10**dens.flatten()


#yield_data = data_yielder()
rng = np.random.default_rng(87899)
N_samples = 2000
min_r = 1e-1

if True:
    fig, ax = plt.subplot_mosaic(
    """
    A
    A
    B
    """
    )
    ax["B"].sharex(ax["A"])
    r = np.geomspace(0.2, 30, 31)
    rb = 1.3
    Re = 6
    n = 3
    log10densb = 9
    gamma = 0.5
    alpha = 2
    mu_cs = core_sersic(r, rb, Re, n, log10densb, gamma, alpha)
    mu_s = core_sersic(r, rb, Re, n, log10densb, gamma, alpha, edit=True)
    ax["A"].loglog(r, mu_cs)
    ax["A"].loglog(r, mu_s)
    ax["B"].loglog(r, np.abs(mu_cs-mu_s)/mu_cs*100)
    for i in range(1, 5):
        for k in "AB":
            ax[k].axvline(i * rb, c="k", ls=":", label=(r"$n \times r_\mathrm{b}$" if i==1 else ""))
            ax[k].axvline(i * Re, c="k", ls="--", label=(r"$n \times R_\mathrm{e}$" if i==1 else ""))
    ax["B"].axhline(1, c="k", ls="-")
    ax["A"].set_ylabel("Surface density")
    ax["B"].set_xlabel("r/kpc")
    ax["B"].set_ylabel("|CS - S| / CS * 100 (Rel. Err.)")
    ax["A"].legend()
    plt.savefig("mass_240913.pdf")
    plt.show()
    quit()

fig, ax = plt.subplots(1,1)

while True:
    try:
        kv, params = next(yield_data)
    except StopIteration:
        break
    miss_mass = np.full(N_samples, np.nan)
    for i in tqdm(range(N_samples), desc=f"MC sampling parameters for {kv}"):
        # set this iteration's parameters
        _pars = {}
        for k, v in params.items():
            '''vv = v[np.logical_and(
                v > np.nanquantile(v, 0.25),
                v < np.nanquantile(v, 0.75)
            )]'''
            _pars[k] = rng.choice(v)
        b = _pars["rb"]
        #print(_pars)
        dens_cs = core_sersic(b, **_pars)
        # integrate core fit
        r = np.geomspace(min_r, b, 1000)
        '''ax.plot(r, core_sersic(r, **_pars), label="core", c="tab:blue")
        ax.scatter(b, core_sersic(b, **_pars), marker="o")'''
        int_cs, *_ = scipy.integrate.quad(lambda x: 2*np.pi * x * 10**core_sersic(x, **_pars), a=min_r, b=b)
        #int_cs = scipy.integrate.trapezoid(2*np.pi*r*10**core_sersic(r, **_pars), r)
        # integrate sersic fit
        _pars["rb"] = 0
        _pars["g"] = 0
        _pars["a"] = 1
        offset = dens_cs - core_sersic(b, **_pars)
        '''ax.plot(r, core_sersic(r, **_pars)+offset, label="sersic", c="tab:orange")
        ax.scatter(b, core_sersic(b, **_pars)+offset, marker="o")
        ax.set_xscale("log")
        ax.legend()
        plt.show()
        quit()'''
        int_s, *_ = scipy.integrate.quad(lambda x: 2*np.pi * x * 10**(core_sersic(x, **_pars) + offset), a=min_r, b=b)
        #int_s = scipy.integrate.trapezoid(2*np.pi*r*10**core_sersic(r, **_pars), r)
        miss_mass[i] = int_s - int_cs
    med, err = bgs.mathematics.quantiles_relative_to_median(miss_mass)
    #err[0] = err[0] if err[0] > 0 else 1e6
    ax.errorbar(float(kv), med, yerr=err, fmt=".", ls="", color="tab:blue")
    #ax.plot(float(kv), med, c="tab:blue", ls="", marker="o")

ax.set_yscale("log")
ax.set_xlabel(r"$v_\mathrm{kick}/\mathrm{kms}^{-1}$")
#ax.set_ylabel(r"$\log_{10}(M/\mathrm{M}_\odot)$")
ax.set_ylabel(r"$M_\mathrm{def}$")

bgs.plotting.savefig("missing_mass.png")
plt.show()
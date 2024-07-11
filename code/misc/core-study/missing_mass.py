from tqdm import tqdm
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import baggins as bgs


datafile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/core-paper-data/core-kick.pickle"

data = bgs.utils.load_data(datafile)


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
        yield k, d

def sersic_b(n):
    return 2.0 * n - 0.33333333 + 0.009876 / n

def core_sersic(r, rb, Re, n, log10densb, g, a):
    b = sersic_b(n)
    preterm = - g / a * np.log(2.0) + b * pow((pow(2.0, 1/a) * rb / Re), 1/n)
    dens = log10densb + (preterm + g / a * np.log(pow(r, a) + pow(rb, a)) - g * np.log(r) - b * pow(Re, -1/n) * pow((pow(r, a) + pow(rb, a)), (1/(a*n)))) / np.log(10.0)
    return dens.flatten()


yield_data = data_yielder()
rng = np.random.default_rng(87899)
N_samples = 2000
min_r = 1e-1


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
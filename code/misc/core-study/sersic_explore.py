from copy import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import baggins as bgs

def sersic_b(n):
    return 2.0 * n - 0.33333333 + 0.009876 / n

def core_sersic_log10(r, rb, Re, n, log10densb, g, a):
    b = sersic_b(n)
    preterm = - g / a * np.log(2.0) + b * pow((pow(2.0, 1/a) * rb / Re), 1/n)
    dens = log10densb + (preterm + g / a * np.log(pow(r, a) + pow(rb, a)) - g * np.log(r) - b * pow(Re, -1/n) * pow((pow(r, a) + pow(rb, a)), (1/(a*n)))) / np.log(10.0)
    return dens

def core_sersic(r, rb, Re, n, log10densb, g, a):
    return 10**core_sersic_log10(r, rb, Re, n, log10densb, g, a)


pars_v = dict(
    rb = np.geomspace(0.1, 1.5, 10),
    Re = np.linspace(2, 15, 10),
    n = np.linspace(1.5, 6, 10),
    log10densb = np.linspace(8, 12, 10),
    g = np.geomspace(0.01, 1, 15),
    a = 20
)

pars = dict(
    rb = np.median(pars_v["rb"]),
    Re = np.median(pars_v["Re"]),
    n = np.median(pars_v["n"]),
    log10densb = np.median(pars_v["log10densb"]),
    g = np.median(pars_v["g"]),
    a = pars_v["a"]
)

# choose which parameter to vary
chosen = "g"

cmapper, sm = bgs.plotting.create_normed_colours(np.min(pars_v[chosen]), np.max(pars_v[chosen]))


r = np.geomspace(0.1, 20, 500)
rb_dens = []
for p in tqdm(pars_v[chosen]):
    pars[chosen] = p
    plt.loglog(r, core_sersic(r, **pars), c=cmapper(p))
    rb_dens.append(core_sersic_log10(pars["rb"], **pars))

sersic_pars = copy(pars)
sersic_pars["g"] = 0
sersic_pars["a"] = 1
sersic_pars["rb"] = 0

offset = np.median(rb_dens) - core_sersic_log10(pars["rb"], **sersic_pars)
print(f"Offset is {offset:.2e}")

plt.loglog(r, core_sersic(r, **sersic_pars)*10**offset, c="tab:red")
y = core_sersic(r, **sersic_pars)*10**offset
plt.loglog(pars["rb"], core_sersic(pars["rb"], **sersic_pars)*10**offset, c="tab:red", marker="o", ls="")

plt.colorbar(sm, ax=plt.gca())
plt.show()
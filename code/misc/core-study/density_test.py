import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import cm_functions as cmf


if False:
    # extract
    main_path = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/HMQcubes/core_study_2/H_2M-H_2M-30.000-2.000"

    hmqfiles = cmf.utils.get_files_in_dir(main_path)

    # note the argument order here is different to the function 
    # core_Sersic_profile
    log_core_sersic = lambda x, rb, Re, Ib, gamma, n, a: np.log10(cmf.literature.core_Sersic_profile(x, Re=Re, rb=rb, Ib=Ib, n=n, gamma=gamma, alpha=a))
    p_bounds = ([0, 0, 0, 0, 0, 0], [30, 30, np.inf, 20, 20, 30])

    N = len(hmqfiles)
    rb = np.full(N, np.nan)
    vk = np.full(N, np.nan)

    for i, f in enumerate(hmqfiles):
        hmq = cmf.analysis.HMQuantitiesSingleData.load_from_file(f)
        dens = np.log10(list(hmq.projected_mass_density.values())[0])
        dens_m = np.mean(dens, axis=0)
        dens_s = np.std(dens, axis=0)
        r = cmf.mathematics.get_histogram_bin_centres(hmq.radial_edges)
        popt, pcov = scipy.optimize.curve_fit(log_core_sersic, r, dens_m, sigma=dens_s, bounds=p_bounds, maxfev=int(1e4))
        rb[i] = popt[0]
        vk[i] = hmq.merger_remnant["kick"]
        print(popt)
    data = {"vk":vk, "rb":rb}
    cmf.utils.save_data(data, "data.pickle")
else:
    data = cmf.utils.load_data("data.pickle")
    vk = data["vk"]
    rb = data["rb"]

# normalise, can use the 0km/s kick data
vesc = 1800
vk /= vesc
rb /= rb[0]
min_rb = min(rb)

if True:
    vk = np.delete(vk,2)
    rb = np.delete(rb,2)

# Modelling
class ModelBase:
    def __init__(self, f, name) -> None:
        self.f = f
        self.name = name
        self.popt=None
        self.r2 = None
        self.fit()

    def fit(self):
        self.popt, *_ = scipy.optimize.curve_fit(self.f, vk, rb)
        self.r2 = r2_score(rb, self.f(vk, *self.popt))
        print(f"{self.name} model: {self.r2:.3f}")

    def plot(self, v, ax, **kwargs):
        ax.plot(v, self.f(v, *self.popt), label=f"{self.name}: r2={self.r2:.3f}", **kwargs)

# Models
Linear = ModelBase(lambda x, a, b: a*x + b, "linear")
Power = ModelBase(lambda x, a, b: a*x**b, "power")
Combined = ModelBase(lambda x, a, b, c, d: a*(c*x+d)**b, "combined")
Nasim = ModelBase(lambda x, a, b: a*x**b+1, "Nasim")
Poly4 = ModelBase(lambda x, a, b, c, d, e: a + b*x +c*x**2 + d*x**3 +e*x**4, "Poly4")
Rayleigh = ModelBase(lambda x,s,c: x/s**2 * np.exp(-x**2/(2*s**2))+c, "Rayleigh")


vkseq = np.linspace(0.9*min(vk), 1.1*max(vk), 1000)

fig, ax = plt.subplots(1,1)
axt = cmf.plotting.twin_axes_from_samples(ax, vk, vk*vesc)
ax.set_xlabel(r"$v_k/v_\mathrm{esc}$")
axt.set_xlabel(r"$v_k/\mathrm{kms}^{-1}$")
ax.set_ylabel(r"$r_b/r_{b,0}$")
ax.plot(vk, rb, marker="o", ls="", mew=0.5, mec="k", ms=10, zorder=1)
for m in [Linear, Power, Combined, Nasim, Poly4, Rayleigh]:
    m.plot(vkseq, ax, zorder=0.5)
plt.legend()
cmf.plotting.savefig("vk-rb.png")
plt.show()

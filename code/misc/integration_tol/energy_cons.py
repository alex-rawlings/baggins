import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os.path
import pygad
import cm_functions as cmf


def calculate_H(s, chunk=1e5):
    chunk = int(chunk)
    total_N = s["pos"].shape[0]
    KE = 0
    PE = 0
    for start in range(0, total_N, chunk):
        end = min(start+chunk, total_N)
        vel_mag = cmf.analysis.radial_separation(s["vel"][start:end])
        vel_mag = pygad.UnitArr(vel_mag, "km/s")
        KE += np.sum(0.5 * s["mass"][start:end]*vel_mag**2)
        PE += np.sum(s["pot"][start:end]*s["mass"][start:end])
    return KE+PE

intacc = True

if intacc:
    subfigdir = "errtolintacc"
    snapdirs = [
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.002-0.0035",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.002-0.0200",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.002-0.0350",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.010-0.0035",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.020-0.0035",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.030-0.0035",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.050-0.0035"
    ]
else:
    subfigdir = "errtolforceacc"
    snapdirs = [
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolintacc-test/0.020-0.0035",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolforceacc-test/0.01",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolforceacc-test/0.02",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolforceacc-test/0.03",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolforceacc-test/0.04",
        "/scratch/pjohanss/arawling/collisionless_merger/acc-params/errtolforceacc-test/0.05"
    ]

fig, ax = plt.subplots(1,1, sharex="all")
ax = [ax]
for j, snapdir in enumerate(snapdirs):
    labval = snapdir.split("/")[-1]
    snapdir = os.path.join(snapdir, "output")
    snapfiles = cmf.utils.get_snapshots_in_dir(snapdir)
    rel_energy_error = np.full_like(snapfiles, np.nan, dtype=float)
    snaptimes = np.full_like(snapfiles, np.nan, dtype=float)
    numfiles = len(snapfiles)
    for i, snapfile in enumerate(snapfiles):
        print("Running {:.2f}%                    ".format(i/(numfiles-1)*100), end="\r")
        snap = pygad.Snapshot(snapfile, physical=True)
        snaptimes[i] = cmf.general.convert_gadget_time(snap)
        H = calculate_H(snap)
        if i == 0:
            H0 = H
        rel_energy_error[i] = np.abs((H-H0)/H0)
    if j == 0:
        REE_interp = scipy.interpolate.interp1d(snaptimes, rel_energy_error)
    if intacc:
        eta, epsilon = labval.split("-")
        label = r"$\eta=${}, $\varepsilon_\star=${}".format(eta, epsilon)
    else:
        if j == 0:
            label = r"$\alpha=$0.005"
        else:
            label = r"$\alpha=${}".format(labval)
    ax[0].plot(snaptimes, rel_energy_error, markevery=[-1], marker="o", label=label, alpha=0.7, markersize=10-j)
    #ax[1].plot(snaptimes, rel_energy_error-REE_interp(snaptimes))

    print("Complete                                   ")
ax[0].set_xlabel("t/Gyr")
ax[0].set_ylabel(r"$|(H-H_0)/H_0|$")
#ax[1].set_ylabel("Residual (to fiducial)")
ax[0].legend()
ax[0].set_yscale("log")
plt.savefig("{}/{}/energy_cons.png".format(cmf.FIGDIR, subfigdir))
plt.show()
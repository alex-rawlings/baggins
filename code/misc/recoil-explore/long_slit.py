import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.ndimage import uniform_filter1d
import pygad
import baggins as bgs

bgs.plotting.check_backend()

# set up instrument
micado = bgs.analysis.MICADO_NFM()
micado.redshift = 0.6
fors2 = bgs.analysis.VLT_FORS2()
fors2.redshift = 0.6
fors2.max_extent = micado.extent


# read in snapshot, centre on BH
snapfile = bgs.utils.get_snapshots_in_dir("/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/kick-vel-0600/output")[5]
snap = pygad.Snapshot(snapfile, physical=True)
bgs.analysis.basic_snapshot_centring(snap)
bhr = snap.bh['r'].flatten()
print(f"BH pos {bhr}")
pygad.Translation(-snap.bh["pos"].flatten()).apply(snap, total=True)
pygad.Boost(-snap.bh["vel"].flatten()).apply(snap, total=True)

for instr in (fors2, micado):
    print(f"Doing {instr.name}")
    bin_centres, vel_disp = instr.get_LOS_velocity_dispersion_profile(snap)

    lp = plt.plot(bin_centres, vel_disp, alpha=0.5)
    plt.plot(bin_centres, np.sqrt(uniform_filter1d(vel_disp**2, 20, mode="nearest")), c=lp[-1].get_color(), label=instr.name)
plt.axvline(-bhr, ls=":", c="k", label="gal centre")
plt.legend()
plt.xlabel("pos/kpc (rel. BH)")
plt.ylabel("vel disp [km/s]")
plt.grid(alpha=0.2)
bgs.plotting.savefig("longslit.png")
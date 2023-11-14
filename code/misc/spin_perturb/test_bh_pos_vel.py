import numpy as np
import matplotlib.pyplot as plt
import ketjugw

pc = ketjugw.units.pc
myr = ketjugw.units.yr * 1e6

ts_in = (0, 100000*myr, 4)


bh1, bh2 = ketjugw.keplerian_orbit(1e9, 1e9, 2*pc, 0.9, l=0, ts=0)

for i, bh in enumerate((bh1, bh2), start=1):
    print(f"BH{i}:")
    print(f"  pos: {bh.x / ketjugw.units.pc}")
    print(f"  vel: {bh.v / ketjugw.units.km_per_s}")
orbit_pars = ketjugw.orbital_parameters(bh1, bh2)

a, e, n, l, ts = ketjugw.peters_evolution(
                        orbit_pars["a_R"],
                        orbit_pars["e_t"],
                        orbit_pars["m0"],
                        orbit_pars["m1"],
                        ts_in
)


fig, ax = plt.subplots(2,1,sharex="all")
ax[0].semilogy(ts/myr, a/pc)
ax[1].plot(ts/myr, e)

ax[-1].set_xlabel("t/Myr")
ax[0].set_ylabel("a/pc")
ax[1].set_ylabel("e")

plt.show()

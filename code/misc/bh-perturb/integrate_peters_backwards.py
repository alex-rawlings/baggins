import numpy as np
import os.path
import matplotlib.pyplot as plt
import ketjugw
import baggins as bgs


main_path = "/scratch/pjohanss/arawling/collisionless_merger/merger-test/D-E-3.0-0.001/perturbations/"
data_file = "output/ketju_bhs_cp.hdf5"

myr = ketjugw.units.yr * 1e6
err_level = 0.05

fig, ax = plt.subplots(2,2, sharex="all")

for i in (2,):
    label = "{:03d}".format(i)
    bhfile = os.path.join(main_path, label, data_file)
    bh1, bh2, merged = bgs.analysis.get_bound_binary(bhfile)
    op = ketjugw.orbital_parameters(bh1, bh2)
    ax[0,0].semilogy(op["t"]/myr, op["a_R"]/ketjugw.units.pc, label=label)
    ax[0,1].plot(op["t"]/myr, op["e_t"])
    peter_a, peter_e, peter_n, peter_l = ketjugw.peters_evolution(op["a_R"][-1], op["e_t"][-1], op["m0"][0], op["m1"][0], -op["t"])
    peter_a = peter_a[::-1]
    peter_e = peter_e[::-1]
    idx, gwtime = bgs.analysis.find_where_gw_dominate(op, err_level)
    ax[0,0].plot(op["t"]/myr, peter_a/ketjugw.units.pc)
    ax[0,1].plot(op["t"]/myr, peter_e)
    ax[1,0].semilogy(op["t"]/myr, np.abs((op["a_R"]-peter_a)/op["a_R"]))
    ax[1,1].semilogy(op["t"]/myr, np.abs((op["e_t"]-peter_e)/op["e_t"]))
    for axi in np.concatenate(ax).flat:
        axi.axhline(err_level, c="tab:red", alpha=0.4)
        axi.axvline(gwtime/myr, c="tab:red")
plt.show()
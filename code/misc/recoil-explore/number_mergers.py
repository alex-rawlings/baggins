import numpy as np
import matplotlib.pyplot as plt
from astropy.units import Unit
from astropy.cosmology import Planck18
from scipy.integrate import trapezoid
import baggins as bgs


if False:
    # merger rate per galaxy from O'Leary 2021
    '''data = np.array([
        [0.019383064739436233, 0.06062345524743358],
        [0.08583928670321761, 0.06069307019818064],
        [0.1554284448452915, 0.05671018000599599],
        [0.22504133735235246, 0.05580628626031328],
        [0.2914817364061425, 0.05397360488757415],
        [0.36425129945648305, 0.052206814801666244],
        [0.4275192050570021, 0.04962513560278291],
        [0.4749721121211402, 0.047979708302644614],
        [0.5287620946368247, 0.04720206821393754],
        [0.5825441656975134, 0.04564197131622578],
        [0.6299970727616516, 0.044128614330423706],
        [0.6837554094573534, 0.0405157122500458],
        [0.7438903788795799, 0.04126430076851746],
        [0.8198324353831913, 0.04061103479477571],
        [0.8894215935252652, 0.03794599755657991],
        [0.9779745092920039, 0.0336767459946601],
        [1.0064320129114945, 0.031992159220944796],
    ])'''

    data = np.array([
        [0.03214285714285714, 0.0042470210699944],
        [0.06071428571428572, 0.0034922123941341972],
        [0.10357142857142854, 0.003231283031522575],
        [0.15357142857142847, 0.002932158886771794],
        [0.1785714285714285, 0.002364016852402281],
        [0.22142857142857142, 0.002022246114943171],
        [0.2857142857142857, 0.0015383803272786987],
        [0.35, 0.0015708486184699666],
        [0.4035714285714286, 0.0014255329913998335],
        [0.45000000000000007, 0.001105547778480822],
        [0.5107142857142856, 0.0008746257115902645],
        [0.5928571428571429, 0.0007199379195354823],
        [0.6571428571428571, 0.0006931025294115104],
        [0.7642857142857141, 0.0006548509340346287],
        [0.7999999999999997, 0.0005940639492788239],
        [0.8714285714285716, 0.0006066870035106917],
        [0.9214285714285714, 0.0005954812627589828],
        [0.957142857142857, 0.0006077083603297764],
        [1.0035714285714286, 0.0005734858771250684],
    ])


    z = data[:,0]
    mask = z < 1.01

    z = z[mask]
    Rz = data[mask,1] * Unit("1/Gyr")


    N_merger_per_gal = trapezoid(Rz / ((1+z)*bgs.cosmology.Hubble_parameter(z)))

    print(f"Number of mergers per galaxy: {N_merger_per_gal:.3e}")

    N_gal = 960 #108891#960# 1827761  # from SDSS
    print(f"There are {N_gal:.2e} galaxies")
    N_mergers = N_merger_per_gal * N_gal

    print(f"There are {N_mergers:.3e} expected mergers")

if True:
    redshift = 0.6
    halo_mass = 10**bgs.literature.Moster10(1.38e11 * 2)
    print(f"Halo mass: {halo_mass:.3e} Msun (log={np.log10(halo_mass):.3f})")

    '''redshifts = np.linspace(0, 1, 10)
    num_mergers_per_halo = np.full_like(redshifts, np.nan)
    for i, z in enumerate(redshifts):
        num_mergers_per_halo[i] = bgs.literature.Fakhouri2010_cumulative_mergers(halo_mass, 1/10, redshift=4, redshift0=2)
    plt.plot(redshifts, num_mergers_per_halo)
    plt.show()
'''
    '''cosmo = Planck18
    Mvir = np.geomspace(2e10, 1e15, 10)
    n = np.full_like(Mvir, np.nan)
    for i, Mv in enumerate(Mvir):
        n[i] = bgs.literature.RodriguezPuebla2016_halo_mass_function(Mv, z=0)
    print(n)
    # TODO about an order of magnitude off
    plt.loglog(Mvir, n/cosmo.h**3, label="alex")

    
    idx = np.argsort(data[:,0])
    plt.loglog(data[idx,0], data[idx,1], label="true")
    plt.legend()
    plt.close()'''

    def cumulative_mass_func(M):
        data = np.array([
            [489223707625289.94, 0.0000015130598777438696],
            [346429255922580.06, 0.0000033510418846662216],
            [187046078790514, 0.000010685060324987081],
            [103511664769066.88, 0.00002792821412001957],
            [68073243716148.93, 0.00004905095666006939],
            [43677418228435.9, 0.00008905130491118541],
            [24774464880539.48, 0.0001616713606901079],
            [18889969790492.43, 0.00024870439112267313],
            [10714664743263.68, 0.0004225706024204134],
            [7046374783288.996, 0.0006500543072854863],
            [4303610678422.3, 0.0011044993576334568],
            [2830219562454.6636, 0.001590152765702566],
            [1771715580090.1821, 0.002446184492404595],
            [933300316865.9075, 0.004296289372597643],
            [613774110117.0841, 0.006609123767732891],
            [384222188262.76807, 0.009835710907082526],
            [223375944873.22467, 0.01564028290154816],
            [133105806788.81677, 0.02657421422234941],
            [73661013052.14282, 0.0395477975386091],
            [48442309425.35684, 0.06083768247572094],
            [32652664762.939087, 0.09358861528011182],
            [19942787756.051838, 0.1347399565218501],
            [13442487997.815092, 0.21425711451386326],
            [8010142771.7011, 0.2886899429409495],
            [10000000000, 0.24461844924045936],
        ])
        idx = np.argsort(data[:, 0])
        return np.interp(M, data[idx,0], data[idx, 1])

    number_mergers_per_Mpc3 = cumulative_mass_func(halo_mass * Planck18.h)/Planck18.h**3 * bgs.literature.Fakhouri2010_cumulative_mergers(halo_mass, 1/3, redshift=redshift) * Unit("1/Mpc^3")
    print(f"There are {number_mergers_per_Mpc3:.3e}")

    vol = Planck18.comoving_volume(z=redshift).to(Unit("Mpc^3"))
    print(f"volume: {vol:.3e}")
    number_mergers = number_mergers_per_Mpc3 * vol
    print(f"There are {number_mergers:.3e} mergers out to z={redshift}")

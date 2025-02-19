import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from astropy.cosmology import Planck18
from astropy.units import parsec, jansky, erg, cm, Hz

# XXX THIS IS A MOCK PIPELINE FOR ONE CELL IN THE LUMINOSITY MAP

# set the grid directory and file
grid_dir = "/scratch/pjohanss/arawling/ssp_models/synthesizer_data"
grid_name = "bpass-2.2.1-bin_chabrier03-0.1,100.0.hdf5"
#grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c23.01-sps_INTRINSIC_LINES.hdf5"
grid = Grid(grid_name, grid_dir=grid_dir, read_lines=False)

# extract the spectra at a target age and metallicity
log10age = 10.0 # log10(age/yr)
metallicity = 1e-2
spectra_id = "incident"

grid_point = grid.get_grid_point(log10ages=log10age, metallicity=metallicity)
sed = grid.get_spectra(grid_point, spectra_id=spectra_id)
sed.lnu *= 1e4  # let's say there are 10000 stars in this cell

print(f"Lnu is {sed.lnu}")

# plot the rest frame SED
fig, ax = plt.subplots()
ax.set_xlabel(r"$\lambda / \mathrm{Angstrom}$")
ax.set_ylabel(r"$\log(\mathrm{L}_\nu)$")
ax.loglog(sed.wavelength, sed.luminosity_nu, label="rest")

plt.savefig("synthesizer_sed.png", dpi=300)
plt.close()

# determine the transmission
z = 0.1
sed.get_fnu(cosmo=Planck18, z=z)
euclid_filters = FilterCollection(
    filter_codes = [f"Euclid/NISP.{b}" for b in ("Y", "J", "H")],
    new_lam=grid.lam
)
euclid_filters.resample_filters(lam_size=1000)

chosen_filter = euclid_filters[euclid_filters.filter_codes[1]]

Lvo = 3631 * jansky * 4*np.pi * (10*parsec)**2
Lvo=Lvo.to("erg/(Hz*s)")

print(f"Lvo is {Lvo}")
abs_mag = -2.5 * np.log10(chosen_filter.apply_filter(sed.lnu, nu=sed.obsnu) / Lvo)
print(f"Abs. magntiude is: {abs_mag:.2f}")

app_mag = abs_mag + 5 * np.log10(Planck18.luminosity_distance(z).to("pc") / (10 * parsec)) - 2.5 * np.log10(1 + z)
print(f"App. magnitude is {app_mag:.2f}")



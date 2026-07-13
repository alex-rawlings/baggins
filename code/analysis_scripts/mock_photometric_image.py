import pygad
import baggins as bgs


snapfile = "/orion/ptmp/arawling/recoil-sims/k0540_snap_009.hdf5"
redshift = 0.6


snap = pygad.Snapshot(snapfile, physical=True)
sed = bgs.analysis.get_spectrum_ssp(1e9, 0.03396)[1]
bgs.analysis.set_luminosity(snap, sed=sed, z=redshift)
inst = bgs.analysis.HSTWFC3(z=redshift)

inst.flux_zeropoint = 1e-2  # erg/s/cm^2 for 1 e-/s -- calibrate for real use
mask = inst.get_fov_mask(0, 1)
stars = snap.stars[mask]
img = inst.image_from_snapshot(
    stars,
    xaxis="x",
    yaxis="y",
    weight_type="luminosity",
)
fig, ax = inst.plot_image(img, title="Snapshot mock observation", stretch="log")
bgs.plotting.savefig("test.png")

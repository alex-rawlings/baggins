import pygad
import baggins as bgs


snapfile = "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/minor_merger/parents/H_2M_a-H_2M_minor_c-30.000-1.900/output/snap_023.hdf5"
galaxy_star_age = 3.645e9  # yr
galaxy_metallicity = 0.03396487304923489
galaxy_redshift = 0.1
xaxis = 0
yaxis = 2

snap = pygad.Snapshot(snapfile, physical=True)
synth_grid, synth_SED = bgs.analysis.get_spectrum_ssp(
            age=galaxy_star_age, metallicity=galaxy_metallicity
    )
bgs.analysis.set_luminosity(snap, synth_SED, z=galaxy_redshift)

euclid = bgs.analysis.Euclid_VIS(z=galaxy_redshift)
print(euclid)
fov_mask = euclid.get_fov_mask(xaxis, yaxis)

fig, ax, *_ = pygad.plotting.image(snap.stars[fov_mask], "lum", cbartitle="", showcbar=False, cmap="copper", Npx=euclid.number_pixels, xaxis=xaxis, yaxis=yaxis, scaleind="none", logscale=False)
ax.set_facecolor("k")
bgs.plotting.savefig("thesis_merger.png")
import baggins as bgs


mergemask = bgs.analysis.MergeMask.make_default_mask()

classifier_pre = bgs.analysis.OrbitClassifier(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis/premerger/classification/premerger.cl",
    mergemask=mergemask
)

classifier_2000 = bgs.analysis.OrbitClassifier(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis/kick-vel-2000/classification/kick-vel-2000.cl",
    mergemask=mergemask
)

# where did the pi-box orbits in the 2000 km/s case come from?
classifier_2000.compare_class_change(classifier_pre, "pi-box")

print()

# where did the rosette orbits in the premerger case go?
classifier_pre.compare_class_change(classifier_2000, "rosette", other_is_earlier=False)

print()

# let's look at rosettes in the 0 km/s case
classifier_0000 = bgs.analysis.OrbitClassifier(
    "/scratch/pjohanss/arawling/collisionless_merger/mergers/core-study/vary_vkick/orbit_analysis/kick-vel-0000/classification/kick-vel-0000.cl",
    mergemask=mergemask
)

# do rosettes decrease with time?
classifier_0000.compare_class_change(classifier_pre, "rosette", other_is_earlier=True)
import baggins as bgs


filename = "/scratch/pjohanss/arawling/collisionless_merger/mergers/processed_data/kicksurvey-paper-data/perfect_obs/perf_obs_0600.pickle"

data = bgs.utils.load_data(filename)

for i in range(len(data["cluster_props"])):
    try:
        data["cluster_props"][i]["cluster_Re_pc"] = data["cluster_props"][i][
            "cluster_Re_pc"
        ].in_units_of("pc")
    except AttributeError:
        continue

bgs.utils.save_data(data, filename, exist_ok=True)

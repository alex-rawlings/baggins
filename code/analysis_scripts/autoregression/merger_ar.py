import baggins as bgs

bgs.plotting.check_backend()

mar = bgs.analysis.MergerAutoRegression("autoregress/merger", thin=1000)
mar.extract_data(
    "/scratch/project_2007917/rubywrig/ketju_stellarfeedback/merger_runs_ketju/merger_m9p50_mu1_z0_v2/r1/merger_m9p50_mu1_z0_vsn4000_ketju/output"
)
print(f"Number of observations: {mar.num_obs_collapsed}")
mar.set_stan_data()
mar.sample_model()
mar.sample_diagnosis
mar.all_posterior_pred_plots()

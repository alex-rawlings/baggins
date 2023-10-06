from abc import ABC, abstractmethod
import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, collections, patches
from datetime import datetime
import cmdstanpy
import arviz as az
import yaml

from ...plotting import savefig, create_normed_colours
from ...env_config import figure_dir, data_dir, tmp_dir, _cmlogger

__all__ = ["HierarchicalModel_1D", "HierarchicalModel_2D", "FactorModel_2D"]

_logger = _cmlogger.copy(__file__)



class _StanModel(ABC):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        """
        Class to set up, run, and plot key plots of a stan model.

        Parameters
        ----------
        model_file : str
            path to .stan file specifying the likelihood model
        prior_file : str
            path to .stan file specifying the prior model
        figname_base : str
            path-like base name that all plots will share
        num_OOS : int
            number of out-of-sample points for posterior modelled quantity
        rng :  np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        self._model_file = model_file
        self._prior_file = prior_file
        self.figname_base = figname_base
        self._num_OOS = num_OOS
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng
        self._num_obs = None
        self._stan_data = {}
        self._model = None
        self._fit = None
        self._fit_for_az = None
        self._prior_model = None
        self._prior_fit = None
        self._prior_fit_for_az = None
        self._parameter_diagnostic_plots_counter = 0
        self._gq_distribution_plot_counter = 0
        self._group_par_counter = 0
        # corner plot method doesn't save figure --> ensures first plot index 0
        self._parameter_corner_plot_counter = -1 
        self._trace_plot_cols = None
        self._observation_mask = True
        self._plot_obs_data_kwargs = {"marker":"o", "linewidth":0.5, "edgecolor":"k", "label":"Sims.", "cmap":"PuRd"}
        self._default_hdi_levels = [99, 75, 50, 25]
        self._num_groups = 0
        self._loaded_from_file = False
        self._generated_quantities = None
        self._obs_collapsed = {}
        self._obs_collapsed_names = []
        self._input_data_file_count = 0
        self._input_data_files = {}


    @property
    def num_OOS(self):
        return self._num_OOS

    @property
    def model_file(self):
        return self._model_file

    @property
    def prior_file(self):
        return self._prior_file
    
    @property
    def observation_mask(self):
        return self._observation_mask

    @observation_mask.setter
    def observation_mask(self, m):
        self._observation_mask = m
    
    @property
    def obs(self):
        return self._obs
    
    @obs.setter
    def obs(self, d):
        self._check_observation_validity(d, True)
        # set categorical label
        # access the first value of the dict
        v = next(iter(d.values()))
        while v[0].ndim > 1:
            v = next(iter(d.values()))
        self._num_groups = len(v)
        _label = []
        for j, vv in enumerate(v, start=1):
            _label.append(np.repeat(j, vv.shape[-1]))
        self._num_obs = sum(len(sublist) for sublist in _label)
        d["label"] = _label
        self._obs = d

    @property
    def obs_collapsed(self):
        return self._obs_collapsed

    @property
    def num_obs(self):
        return self._num_obs

    @property
    def num_obs_collapsed(self):
        return self._num_obs_collapsed

    @property
    def num_groups(self):
        return self._num_groups

    @property
    def points_per_group(self):
        return [len(o) for o in self.obs["label"]]

    @property
    def generated_quantities(self):
        return self._generated_quantities

    @property
    def figname_base(self):
        return self._figname_base

    @figname_base.setter
    def figname_base(self, f):
        self._figname_base = os.path.join(figure_dir, f)
        os.makedirs(os.path.dirname(self._figname_base), exist_ok=True)

    @property
    def sample_diagnosis(self):
        return self._sample_diagnosis

    @property
    def stan_data(self):
        return self._stan_data

    @stan_data.setter
    def stan_data(self, d):
        try:
            assert isinstance(d, dict)
        except AssertionError:
            _logger.logger.exception("Input to property `stan_data` must be a dict!", exc_info=True)
            raise
        self._stan_data.update(d)


    @abstractmethod
    def set_stan_data(self):
        pass


    @abstractmethod
    def _set_stan_data_OOS(self):
        pass


    def _make_fig_name(self, fname, tag):
        """
        Make figure names by appending a tag to a base name.

        Parameters
        ----------
        fname : str
            base figure name to which a tag will be appended
        tag : str
            tag to append

        Returns
        -------
        str, path-like
            path to save figure as
        """
        fname_parts = list(os.path.splitext(fname))
        if fname_parts[1] == "":
            fname_parts[1] = ".png"
        elif fname_parts[1] not in (".png", ".jpeg", ".jpg", ".eps", ".pdf"):
            # we do not have a valid extension
            fname_parts = [fname, ".png"]
        fittype = "posterior" if self._fit is not None else "prior"
        return f"{fname_parts[0]}_{fittype}_{tag}{fname_parts[1]}"


    def _get_data_dir(self, d):
        """
        Get the observed data directories for a Stan model

        Parameters
        ----------
        d : path-like, list
            (list of) path(s) to data, by default None

        Returns
        -------
        d : path-like, list
            observed data directories
        """
        if d is None:
            try:
                assert self._loaded_from_file
                d = [[f["path"] for f in self._input_data_files.values()]]
            except AssertionError:
                _logger.logger.exception(f"HMQ directory must be given if not loaded from file!", exc_info=True)
                raise
        return d


    def _check_observation_validity(self, d, set_categorical=False):
        """
        Ensure that an observation is numerically valid: it is a dict, no NaN 
        values, and each member of the dict is mutable to the same shape

        Parameters
        ----------
        d : any
            proposed value to set the observations to. Should be a dict, but an
            error is thrown if it is not.
        set_categorical : bool, optional
            set the categorical variable to distinguish groups, by default False
        """
        try:
            assert isinstance(d, dict)
        except AssertionError:
            _logger.logger.exception(f"Observational data must be a dict! Current type is {type(d)}", exc_info=True)
            raise
        for i, (k,v) in enumerate(d.items()):
            if set_categorical:
                try:
                    assert k != "label"
                except AssertionError:
                    _logger.logger.exception(f"Keyword 'label' is reserved!", exc_info=True)
                    raise
            try:
                assert isinstance(v, list)
            except AssertionError:
                _logger.logger.exception(f"Data format must be a list! Currently '{k}' type {type(v)}", exc_info=True)
                raise
            for j, vv in enumerate(v):
                try:
                    assert isinstance(vv, (list, np.ndarray))
                    if isinstance(vv, list):
                        d[k][j] = np.array(vv)
                except AssertionError:
                    _logger.logger.exception(f"Observed variable {k} element {j} is not list-like, but type {type(vv)}", exc_info=True)
                    raise
                try:
                    assert not np.any(np.isnan(d[k][j]))
                except AssertionError:
                    _logger.logger.exception(f"NaN values detected in observed variable {k} item {j}! This can lead to undefined behaviour.", exc_info=True)
                    raise


    def transform_obs(self, key, newkey, func):
        """
        Apply a transformation to an observed quantity, saving the result to the
        data frame.

        Parameters
        ----------
        key : str, tuple, list
            observation dictionary key for transformation of a single variable, 
            or tuple-like for a transformation involving a number of 
            independent observations
        newkey : str
            dictionary key for transformed quantity
        func : function
            transformation

        Raises
        ------
        ValueError
            when proposed dictionary key is a reserved keyword
        """
        for k in self.obs.keys():
            if k == newkey:
                _logger.logger.warning(f"Requested key {newkey} already exists! Transformation will not be reapplied --> Skipping.")
                break
        else:
            _logger.logger.debug(f"Applying transformation designated by {newkey}")
            self.obs[newkey] = []
            if isinstance(key, str):
                for v in self.obs[key]:
                    self.obs[newkey].append(func(v))
            else:
                vs = list(map(self.obs.get, key))
                try:
                    dims = [o.shape for v in vs for o in v]
                    dims = np.reshape(dims, (len(vs), self.num_groups))
                    assert (dims == dims[0]).all()
                except AssertionError:
                    _logger.logger.exception(f"Observation transformation failed for keys {key}: differing dimensions: {dims}", exc_info=True)
                    raise
                extract = lambda i: list(map(itemgetter(i), vs))
                for i in range(len(vs[0])):
                    _vs = extract(i)
                    self.obs[newkey].append(func(*_vs))
            self._check_observation_validity(self.obs)


    def print_obs_summary(self):
        """
        Print a shape summary of the observations in the model
        """
        print("General:")
        print(f"  Total observations: {self.num_obs}")
        print("Observations  [groups]  [points/group]:")
        for k, v in self.obs.items():
            vv_shape = []
            for vv in v:
                vv_shape.append(vv.shape)
            print(f"  {k}:  {len(v)}:  {vv_shape}")
        print("Collapsed Observations:")
        for k, v in self.obs_collapsed.items():
            print(f"  {k}:  {v.shape}")


    def collapse_observations(self, obs_names, order="F"):
        """
        Collapse a 2D observed quantity to a 1D representation.

        Parameters
        ----------
        obs_name : list
            observation(s) to collapse
        include_label : bool
            should key `label` be collapsed (sometimes not desired if 2D arrays 
            have been flattened), by default True
        """
        if "label" not in obs_names: obs_names.append("label")
        dim = {}
        order = order.upper()
        for obs_name in obs_names:
            try:
                assert obs_name not in self._obs_collapsed_names
                self._obs_collapsed_names.append(obs_name)
                dim[obs_name] = self.obs[obs_name][0].ndim
            except AssertionError:
                _logger.logger.exception(f"Observation {obs_name} has already been collapsed! Cannot collapse again!", exc_info=True)
                raise
            except IndexError:
                _logger.logger.exception(f"Error collapsing {obs_name}, {self.obs[obs_name]}")
                raise
        try:
            assert max(dim.values()) < 3
        except AssertionError:
            _logger.logger.exception(f"Error collapsing observation {max(dim, key=dim.get)}: data cannot have more than 2 dimensions", exc_info=True)
            raise
        if max(dim.values()) == 1:
            # all observations are 1D, can just concatenate
            for k, v in self.obs.items():
                if k not in obs_names:
                    _logger.logger.debug(f"Observation {k} will not be collapsed.")
                    continue
                self._obs_collapsed[k] = np.concatenate(v)
                _logger.logger.debug(f"Collapsing variable {k}")
        else:
            # collapse the desired variable
            # need to collapse 2D variables first so we know how many tiles to 
            # make of 1D variables
            dim = dict(sorted(dim.items(), key=lambda item: item[1], reverse=True))
            reps = [v.shape[0 if order=="F" else 1] for v in self.obs[list(dim.keys())[0]]]
            for obs_name, ndim in dim.items():
                if ndim == 2:
                    self._obs_collapsed[obs_name] = np.concatenate([v.flatten(order=order) for v in self.obs[obs_name]])
                else:
                    self._obs_collapsed[obs_name] = np.concatenate([np.tile(vv, r) for vv,r in zip(self.obs[obs_name], reps)])
            self._num_obs_collapsed = len(self.obs_collapsed[obs_name])
        # data consistency checks
        try:
            for k, v in self._obs_collapsed.items():
                assert v.shape == self._obs_collapsed[obs_names[0]].shape
        except AssertionError:
            _logger.logger.exception(f"Data lengths are inconsistent in collapsed observations! Observation '{k}' has shape {v.shape}, must have shape {self._obs_collapsed[obs_names[0]].shape}", exc_info=True)
            raise


    def reset_obs_collapsed(self):
        """
        Reset the obs_collapsed attribute.
        """
        self._obs_collapsed = {}
        self._obs_collapsed_names = []


    def thin_observations(self, spacing):
        """
        Thin the observations for large datasets

        Parameters
        ----------
        spacing : int
            Spacing between successive observations per group
        """
        for k,v in self.obs.items():
            for i in range(len(v)):
                idxs = np.r_[np.arange(0, len(v[i]), spacing, dtype=int)]
                self.obs[k][i] = v[i][idxs]
        self._num_obs = sum(self.points_per_group)
        # need to re-collapse observations that have been previously collapsed
        self._obs_collapsed = {}
        obs_names = self._obs_collapsed_names
        self._obs_collapsed_names = []
        self.collapse_observations(obs_names)


    def _check_num_groups(self, pars):
        """
        Ensure a mininmum number of groups exist

        Parameters
        ----------
        pars : dict
            analysis parameters (with 'stan' block)
        """
        try:
            assert self.num_groups >= pars["stan"]["min_num_samples"]
        except AssertionError:
            _logger.logger.exception(f"There are not enough groups to form a valid hierarchical model. Minimum number of groups is {pars['stan']['min_num_samples']}, and we have {self.num_groups}!", exc_info=True)
            raise


    def _add_input_data_file(self, f):
        """
        Save the path to a HMQ file used in the sampling

        Parameters
        ----------
        f : path-like
            path to HMQ file
        """
        self._input_data_files.update(
            {
                f"file{self._input_data_file_count:03d}" : {
                    "path": f,
                    "created": os.path.getmtime(f)
                }
            }
        )
        self._input_data_file_count += 1


    def _get_timestamp_from_csv(self, csvfile):
        return os.path.basename(csvfile).split("-")[-1].split("_")[0]


    def _write_input_data_yml(self, csvfile):
        """
        Save list of HMQ files used to .yml file

        Parameters
        ----------
        csvfile : path-like
            a stan output csv file
        """
        d = os.path.dirname(csvfile)
        tstamp = self._get_timestamp_from_csv(csvfile)
        with open(os.path.join(d, f"input_data-{tstamp}.yml"), "w") as f:
            yaml.dump(self._input_data_files, f)


    def _determine_num_OOS(self, v):
        """
        Determine number of out-of-sample values given a previously saved run

        Parameters
        ----------
        v : str
            inferred posterior variable name
        """
        q = self._fit_for_az["posterior"][v]
        n = [q.sizes[k] for k in q.sizes.keys() if k not in ("chain", "draw")]
        try:
            assert len(n) == 1
        except AssertionError:
            _logger.logger.exception(f"Dataset can only have three dimensions: chain, draw, and other. Currently has size {len(n)+2}", exc_info=True)
            raise
        self._num_OOS = n[0]


    def _sampler(self, prior=False, sample_kwargs={}):
        """
        Sample a stan model

        Parameters
        ----------
        prior : bool, optional
            run sampler for prior model, by default False
        sample_kwargs : dict, optional
            kwargs to be passed to CmdStanModel.sample(), by default {}

        Returns
        -------
        cmdstanpy.CmdStanMCMC 
            container output from stan sampling
        """
        if self._loaded_from_file:
            _logger.logger.warning("Instance instantiated from file: sampling the model again is not possible --> Skipping.")
            self._sample_diagnosis = self._fit.diagnose()
            return self._fit
        else:
            default_sample_kwargs = {"chains":4, "iter_sampling":2000, "show_progress":True, "show_console":False, "max_treedepth":12}
            if not prior:
                default_sample_kwargs["output_dir"] = os.path.join(data_dir, "stan_files")
            else:
                # protect against inability to parallelise, for prior model
                # this shouldn't be so expensive anyway
                default_sample_kwargs["force_one_process_per_chain"] = True
            # update user given sample kwargs
            for k, v in sample_kwargs.items():
                default_sample_kwargs[k] = v
            if "output_dir" in default_sample_kwargs:
                os.makedirs(default_sample_kwargs["output_dir"], exist_ok=True)
            if prior:
                fit = self._prior_model.sample(self.stan_data, **default_sample_kwargs)
            else:
                _logger.logger.debug(f"exe info: {self._model.exe_info()}")
                fit = self._model.sample(self.stan_data, **default_sample_kwargs)
                self._write_input_data_yml(fit.runset.csv_files[0])
            _logger.logger.info(f"Number of threads used: {os.environ['STAN_NUM_THREADS']}")
            self._sample_diagnosis = fit.diagnose()
            _logger.logger.info(f"\n{fit.summary(sig_figs=4)}")
            _logger.logger.info(f"\n{self.sample_diagnosis}")
            return fit


    @abstractmethod
    def extract_data(self):
        pass


    def build_model(self, prior=False):
        """
        Build the stan model

        Parameters
        ----------
        prior : bool, optional
            build prior model, by default False
        """
        if prior:
            self._prior_model = cmdstanpy.CmdStanModel(stan_file=self._prior_file)
        else:
            self._model = cmdstanpy.CmdStanModel(stan_file=self._model_file)#, cpp_options={"STAN_THREADS":"true", "STAN_CPP_OPTIMS":"true"})


    @abstractmethod
    def sample_model(self, sample_kwargs={}):
        """
        Wrapper function around _sampler() to sample a stan likelihood model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
             kwargs to be passed to CmdStanModel.sample(), by default {}
        """
        if self._model is None and not self._loaded_from_file:
            self.build_model()
        self._fit = self._sampler(sample_kwargs=sample_kwargs)
        # TODO capture arviz warnings about NaN
        self._fit_for_az = az.from_cmdstanpy(posterior=self._fit)


    def sample_prior(self, sample_kwargs={}):
        """
        Wrapper function around _sampler() to sample a stan prior model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
            kwargs to be passed to CmdStanModel.sample(), by default {}
        """
        if self._prior_model is None and not self._loaded_from_file:
            self.build_model(prior=True)
        self._prior_fit = self._sampler(sample_kwargs=sample_kwargs, prior=True)
        self._prior_fit_for_az = az.from_cmdstanpy(prior=self._prior_fit)


    def _get_GQ_indices(self, state):
        return np.r_[0:self.num_obs] if state == "pred" else np.r_[self.num_obs:-1]

    @abstractmethod
    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        """
        Sample the 'generated quantities' block of a Stan model. If the model has had both the prior and posterior distributions sampled, the posterior sample will be used.

        Parameters
        ----------
        gq : str
            stan variable to sample
        force_resample : bool, optional
            run the generate_quantities method() again even if already run, by 
            default False
        state : str, optional
            return generated quantities for predictive checks or out-of-sample 
            quantities, by default "pred"

        Returns
        -------
        np.ndarray
            set of draws for the variable gq
        """
        try:
            assert state in ("pred", "OOS")
        except AssertionError:
            _logger.logger.exception(f"`state` must be one of 'pred' or 'OOS', not {state}!", exc_info=True)
            raise
        if self.generated_quantities is None or force_resample:
            if self._model is None:
                _logger.logger.debug("Generated quantities will be taken from the prior model")
                self._generated_quantities = self._prior_model.generate_quantities(data=self._stan_data, mcmc_sample=self._prior_fit)
            else:
                _logger.logger.debug("Generated quantities will be taken from the posterior model")
                self._generated_quantities = self._model.generate_quantities(data=self._stan_data, mcmc_sample=self._fit, gq_output_dir=None)
        try:
            self.generated_quantities.stan_variable(gq)
        except ValueError:
            _logger.logger.error(f"Value error trying to read generated quantities data: creating temporary directory {tmp_dir} ...")
            os.makedirs(tmp_dir, exist_ok=True)
            self._generated_quantities = self._model.generate_quantities(data=self._stan_data, mcmc_sample=self._fit, gq_output_dir=tmp_dir)
        return self.generated_quantities.stan_variable(gq)


    def calculate_mode(self, v):
        """
        Determine the mode of a Stan variable, following the method defined in
        arviz.plot_utils package.

        Parameters
        ----------
        v : str
            stan variable to determine the mode for

        Returns
        -------
        : float
            mode of variable
        """
        if self._fit is None:
            _fit = self._prior_fit
            _logger.logger.debug("Generated quantities will be taken from the prior model")
        else:
            _fit = self._fit
            _logger.logger.debug("Generated quantities will be taken from the posterior model")
        x, dens = az.kde(self._fit[v])
        return x[np.nanargmax(dens)]


    def _parameter_corner_plot(self, var_names, figsize=None, labeller=None, levels=None, combine_dims=None, backend_kwargs=None):
        """
        Base method to create parameter corner plots. This method should not be
        called directly.

        Parameters
        ----------
        var_names : list
            variable names to plot
        figsize : tuple, optional
            figure size, by default None
        labeller : arviz.MapLabeller, optional
            mapping from variable names to labels, by default None
        levels : list, optional
            HDI intervals to plot, by default None
        combine_dims : set-like, optional
            dimensions to reduce, by default None
        backend_kwargs : dict, optional
            keyword arguments to be passed to pyplot.subplots() as per arviz 
            docs, by default None


        Returns
        -------
        ax : matplotlib.axes.Axes
            corner plot
        """
        if levels is None:
            levels = self._default_hdi_levels
        levels = [l/100 for l in levels]
        # show divergences on plots where no dimension combination has 
        # occurred: combining dimensions changes the length of boolean mask 
        # "diverging_mask" in arviz --> index mismatch error
        divergences = True if combine_dims is None else False
        if self._fit_for_az is None:
            data = self._prior_fit_for_az
            group = "prior"
            divergences = False
        else:
            data = self._fit_for_az
            group = "posterior"
        # first lay down the markers
        ax = az.plot_pair(data, group=group, var_names=var_names, kind="scatter", marginals=True, combine_dims=combine_dims, scatter_kwargs={"marker":".", "markeredgecolor":"k", "markeredgewidth":0.5, "alpha":0.2}, figsize=figsize, labeller=labeller, textsize=rcParams["font.size"], backend_kwargs=backend_kwargs)
        # then add the KDE
        az.plot_pair(data, group=group, var_names=var_names, kind="kde", divergences=divergences, combine_dims=combine_dims, ax=ax, figsize=figsize, marginals=True, kde_kwargs={"contour_kwargs":{"linewidths":0.5}, "hdi_probs":levels, "contourf_kwargs":{"cmap":"cividis"}}, point_estimate_marker_kwargs={"marker":""}, labeller=labeller, textsize=rcParams["font.size"], backend_kwargs=backend_kwargs)
        return ax


    def parameter_corner_plot(self, var_names, figsize=None, labeller=None, levels=None, combine_dims=None, backend_kwargs=None):
        """
        See docs for _parameter_corner_plot()
        """
        ax = self._parameter_corner_plot(var_names, figsize=figsize, labeller=labeller, levels=levels, combine_dims=combine_dims, backend_kwargs=backend_kwargs)
        self._parameter_corner_plot_counter += 1
        return ax


    def parameter_diagnostic_plots(self, var_names, figsize=None, labeller=None, levels=None):
        """
        Plot key pair plots and diagnostics of a stan likelihood model.

        Parameters
        ----------
        var_names : list
            variable names to plot
        figsize : tuple, optional
            size of figures, by default (9.0, 6.75)
        labeller : arviz.MapLabeller, optional
            mapping from variable names to labels, by default None
        levels : list, optional
            HDI intervals to plot, by default None
        """
        # TODO choose good figsize always, also labeller sometimes still not working...
        # seems to be to occur when axis label is longer than the axis
        if figsize is None:
            max_dim = max(rcParams["figure.figsize"])
            figsize = (max_dim, max_dim)
        try:
            assert len(var_names) > 1
        except AssertionError:
            _logger.logger.exception("Pair plot requires at least two variables!", exc_info=True)
            raise
        if len(var_names) > 4:
            _logger.logger.warning("Corner plots with more than 4 variables may not correctly map the labels given by the labeller!")
        
        # plot trace
        if self._parameter_diagnostic_plots_counter == 0:
            vmax = len(self._fit_for_az.posterior["chain"])
            cmapper, sm = create_normed_colours(-vmax/2,vmax, cmap="Blues")
            self._trace_plot_cols = [cmapper(x) for x in self._fit_for_az.posterior["chain"]]
        ax = az.plot_trace(self._fit_for_az, var_names=var_names, figsize=figsize, chain_prop={"color":self._trace_plot_cols}, trace_kwargs={"alpha":0.9}, labeller=labeller)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"trace_{self._parameter_diagnostic_plots_counter}"), fig=fig)
        plt.close(fig)

        # plot rank
        ax = az.plot_rank(self._fit_for_az, var_names=var_names, labeller=labeller)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"rank_{self._parameter_diagnostic_plots_counter}"), fig=fig)
        plt.close(fig)

        # plot pair
        ax = self._parameter_corner_plot(var_names=var_names, figsize=figsize, labeller=labeller, levels=levels)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"pair_{self._parameter_diagnostic_plots_counter}"), fig=fig)
        plt.close(fig)
        self._parameter_diagnostic_plots_counter += 1


    def group_parameter_plot(self, var_names, figsize=None, labeller=None, levels=None):
        # TODO add labeller to axis
        if figsize is None:
            max_dim = max(rcParams["figure.figsize"])
            figsize = (max_dim, max_dim)
        if levels is None:
            levels = self._default_hdi_levels
        levels = [l/100 for l in levels]
        az_group = "posterior" if self._fit_for_az is not None else "prior"
        cmapper, sm = create_normed_colours(-0.1, 0.5)
        # loop through variables
        p = []
        for i, v in enumerate(var_names):
            # loop through HDI levels
            for l in levels:
                hdi = az.hdi(self._fit_for_az[az_group][v], l, skipna=True)
                r = patches.Rectangle(
                    (i-0.5, hdi[0]),
                    1, hdi[1] - hdi[0],
                    fc = cmapper(0.5*(1-l)),
                    alpha = 0.7
                )
                p.append(r)
        fig, ax  =plt.subplots(1,1)
        ax.add_collection(collections.PatchCollection(p))
        ax.autoscale_view()
        savefig(self._make_fig_name(self.figname_base, f"group_par_{self._group_par_counter}"), fig=fig)
        plt.close(fig)
        self._group_par_counter += 1



    @abstractmethod
    def _plot_predictive(self, xmodel, ymodel, state, xobs=None, yobs=None, yobs_err=None, levels=None, ax=None, collapsed=True, show_legend=True):
        pass


    def plot_generated_quantity_dist(self, gq, state="pred", ax=None, xlabels=None, save=True, plot_kwargs={}):
        """
        Plot the 1-D distribution of an arbitrary variable in the generated quantities block of a stan model.

        Parameters
        ----------
        gq : list
            variables to plot
        state : str, optional
            return generated quantities for predictive checks or out-of-sample 
            quantities, by default "pred"
        ax : matplotlib.axes.Axes or np.ndarray of, optional
            axes object to plot to, by default None
        xlabels : list, optional
            labels for the x-axis, by default None
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        matplotlib.axes.Axes
            plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots(len(gq),1)
        elif isinstance(ax, (np.ndarray, list)):
            fig = np.atleast_2d(ax)[0,0].get_figure()
        else:
            fig = ax.get_figure()
        if isinstance(ax, np.ndarray):
            ax_shape = ax.shape
        else:
            ax = np.array(ax)
            ax_shape = (1,)
        ax = ax.flatten()
        if xlabels is None:
            xlabels = gq
        else:
            try:
                assert len(gq) == len(xlabels)
            except AssertionError:
                _logger.logger.exception(f"There are {len(gq)} generated quantity variables to plot, but only {len(xlabels)} labels!", exc_info=True)
                raise
        for i, (_gq, l) in enumerate(zip(gq, xlabels)):
            ys = self.sample_generated_quantity(_gq, state=state)
            try:
                assert len(ys.shape) < 3
            except AssertionError:
                _logger.logger.exception(f"Generated quantity {_gq} must have shape 2, has shape {len(ys.shape)}", exc_info=True)
                raise
            az.plot_dist(ys, ax=ax[i], plot_kwargs=plot_kwargs)
            ax[i].set_xlabel(l)
            ax[i].set_ylabel("PDF")
        ax.reshape(ax_shape)
        if save:
            savefig(self._make_fig_name(self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"), fig=fig)
            self._gq_distribution_plot_counter += 1
        return ax
    

    def print_parameter_percentiles(self, vars):
        """
        Print a simple table with the 5%, 50%, and 95% percentiles for some
        variables.

        Parameters
        ----------
        vars : list
            variables in the CmdStanMCMC object to print
        """
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        if self._fit_for_az is None:
            qvals = self._prior_fit_for_az["prior"][vars].quantile(quantiles).to_dataframe()
        else:
            qvals = self._fit_for_az["posterior"][vars].quantile(quantiles).to_dataframe()
        vars = vars.copy()
        vars.insert(0, "Variable")
        max_str_len = max([len(v) for v in vars]) + 1
        head_str = f"\n{vars[0]:>{max_str_len}}          5%        50%        95%        IQR  "
        print(head_str)
        dashes = ["-" for _ in range(len(head_str))]
        print("".join(dashes))
        for v in vars[1:]:
            _iqr = qvals.loc[0.75, v] - qvals.loc[0.25, v]
            print(f"{v:>{max_str_len}}:  {qvals.loc[0.05,v]:>+.2e}  {qvals.loc[0.50,v]:>+.2e}  {qvals.loc[0.95,v]:>+.2e}  {_iqr:>+.2e}")
        print()

    
    def determine_loo(self, stan_log_lik="log_lik"):
        """
        Determine Leave-One Out Statistics (wrapper for arviz method)

        Parameters
        ----------
        stan_log_lik : str, optional
            name of log-likelihood variable in stan code, by default "log_lik"
        """
        if self.generated_quantities is None:
            self.sample_generated_quantity(stan_log_lik, state="pred")
        if "log_likelihood" not in self._fit_for_az:
            self.sample_generated_quantity(stan_log_lik, state="pred")
            self._fit_for_az.add_groups({"log_likelihood":self.generated_quantities.draws_xr(stan_log_lik)})
        l = az.loo(self._fit_for_az)
        print(l)


    def rename_dimensions(self, dim_map):
        """
        Rename dimensions of arviz InferenceData object

        Parameters
        ----------
        dim_map : dict
            mapping of old dimension names to new names
        """
        if self._fit_for_az is not None:
            self._fit_for_az.rename_dims(dim_map, inplace=True)
        else:
            self._prior_fit_for_az.rename_dims(dim_map, inplace=True)


    def _expand_dimension(self, varnames, dim):
        """
        Expand dimensions of variables to match another dimension

        Parameters
        ----------
        varnames : list, tuple
            variables to expand
        dim : str
            dimension name to expand to
        """
        try:
            assert isinstance(varnames, (list, tuple))
        except AssertionError:
            _logger.logger.exception(f"Expanding variables requires first arugment to be a list or tuple, not {type(varnames)}", exc_info=True)
            raise
        if self._fit is None:
            _fit = self._prior_fit_for_az
            type_str = "prior"
        else:
            _fit = self._fit_for_az
            type_str = "posterior"
        for k in varnames:
            _fit[type_str][k] = _fit[type_str][k].expand_dims({dim:np.arange(_fit[type_str].dims[dim])}, axis=-1)


    @classmethod
    def load_fit(cls, model_file, fit_files, figname_base, rng=None):
        """
        Restore a stan model from a previously-saved set of csv files

        Parameters
        ----------
        fit_files : str, path-like
            path to previously saved csv files
        figname_base : str
            path-like base name that all plots will share
        rng : np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        # initiate a class instance
        C = cls(model_file=model_file, prior_file=None, figname_base=figname_base, num_OOS=None, rng=rng)

        # set up the model, be aware of changes between sampling and loading
        C.build_model()
        C._fit = cmdstanpy.from_csv(fit_files)
        fit_time = datetime.strptime(C._fit.metadata.cmdstan_config["start_datetime"], "%Y-%m-%d %H:%M:%S %Z")
        model_build_time = datetime.utcfromtimestamp(os.path.getmtime(C._model.exe_file))
        if model_build_time.timestamp() > fit_time.timestamp():
            print("==========================================")
            _logger.logger.error(f"Stan executable has been modified since sampling was performed! Proceed with caution!\n  --> Compile time: {model_build_time} UTC\n  --> Sample time:  {fit_time} UTC")
            print("==========================================")

        # load path to observation data
        tstamp = os.path.basename(fit_files).split("-")[-1].split("_")[0]
        with open(os.path.join(os.path.dirname(fit_files), f"input_data-{tstamp}.yml"), "r") as f:
            C._input_data_files = yaml.safe_load(f)
        for v in C._input_data_files.values():
            if os.path.getmtime(v["path"]) > v["created"]:
                _logger.logger.error(f"HMQ file {v['path']} has been modified since the Stan model was run, proceed with caution!")
        C._loaded_from_file = True
        return C




class HierarchicalModel_1D(_StanModel):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)


    @abstractmethod
    def extract_data(self):
        return super().extract_data()


    @abstractmethod
    def set_stan_data(self):
        return super().set_stan_data()


    @abstractmethod
    def _set_stan_data_OOS(self):
        return super()._set_stan_data_OOS()


    @abstractmethod
    def sample_model(self, sample_kwargs={}):
        return super().sample_model(sample_kwargs)


    @abstractmethod
    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        return super().sample_generated_quantity(gq, force_resample, state)


    def _plot_predictive(self, xmodel, state, xobs=None, xobs_err=None, levels=None, ax=None, collapsed=True):
        """
        Plot a predictive check for a 1D Stan model.

        Parameters
        ----------
        xmodel :str
            dictionary key for modelled independent variable
        state : str
            return generated quantities for predictive checks or out-of-sample 
            quantities
        xobs :str
            dictionary key for observed independent variable
        xobs_err : str, optional
             dictionary key for observed independent variable scatter, by 
             default None
        levels : list, optional
            HDI intervals to plot, by default None
        ax : matplotlib.axes.Axes, optional
            axis to plot to, by default None (creates new instance)
        collapsed : bool, optional
            plotting collapsed observations?, by default True

        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        if levels is None:
            levels = self._default_hdi_levels
        quantiles = [0.5 - l/200 for l in levels]
        quantiles.extend([0.5 + l/200 for l in levels])
        quantiles.sort()
        if ax is None:
            fig, ax = plt.subplots(1,1, squeeze=False)
        else:
            # TODO assert 2d axes?
            fig = ax.get_figure()
        obs = self.obs_collapsed if collapsed else self.obs
        xs = self.sample_generated_quantity(xmodel, state=state)
        az.plot_dist(xs, quantiles=quantiles, ax=ax)
        # overlay data
        if xobs_err is None:
            ax.scatter(obs[xobs], np.zeros(len(obs[xobs])), c=obs["label"], **self._plot_obs_data_kwargs)
        else:
            colvals = np.unique(obs["label"])
            ncols = len(colvals)
            cmapper, sm = create_normed_colours(np.min(colvals), np.max(colvals), cmap=self._plot_obs_data_kwargs["cmap"])
            for i, c in enumerate(colvals):
                col = cmapper(c)
                mask = obs["label"]==c
                ys = np.zeros(len(obs[xobs][mask]))
                ax.scatter(obs[xobs][mask], np.zeros(len(obs[xobs][mask])), color=col, **self._plot_obs_data_kwargs)
                ax.errorbar(obs[xobs][mask], ys, xerr=obs[xobs_err][mask], c=col, zorder=20, fmt=".", label=("Sims." if i==ncols-1 else ""))


    def plot_predictive(self, xmodel, xobs, xobs_err=None, levels=None, ax=None, collapsed=True, save=True):
        """
        Predictive plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(xmodel=xmodel, state="pred", xobs=xobs, xobs_err=xobs_err, levels=levels, collapsed=collapsed)
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"pred_{xobs}"), fig=fig)
        return ax


    def posterior_OOS_plot(self, xmodel, levels=None, ax=None, collapsed=True, save=True):
        """
        Posterior out-of-sample plot, observed data is not added to plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(xmodel=xmodel, state="OOS", levels=levels, collapsed=collapsed)
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"posterior_OOS_{xmodel}"), fig=fig)
        return ax





class HierarchicalModel_2D(_StanModel):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)


    @abstractmethod
    def extract_data(self):
        return super().extract_data()


    @abstractmethod
    def set_stan_data(self):
        return super().set_stan_data()


    @abstractmethod
    def _set_stan_data_OOS(self):
        return super()._set_stan_data_OOS()


    @abstractmethod
    def sample_model(self, sample_kwargs={}):
        return super().sample_model(sample_kwargs)


    @abstractmethod
    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        return super().sample_generated_quantity(gq, force_resample, state)


    def _plot_predictive(self, xmodel, ymodel, state, xobs=None, yobs=None, yobs_err=None, levels=None, ax=None, collapsed=True, show_legend=True):
        """
        Plot a predictive check for a regression stan model.

        Parameters
        ----------
        xmodel : str
            dictionary key for modelled independent variable
        ymodel : str
            dictionary key for modelled dependent variable
        xobs : str, optional
            dictionary key for observed independent variable, by default None
        yobs : str, optional
            dictionary key for observed dependent variable, by default None
        yobs_err : str, optional
             dictionary key for observed dependent variable scatter, by default 
             None
        levels : list, optional
            HDI intervals to plot, by default None
        ax : matplotlib.axes.Axes, optional
            axis to plot to, by default None (creates new instance)
        collapsed : bool, optional
            plotting collapsed observations?
        show_legend : bool
            create legend, by default True
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            plotting axis
        """
        if levels is None:
            levels = self._default_hdi_levels
        levels.sort(reverse=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
        ys = self.sample_generated_quantity(ymodel, state=state)
        idxs = self._get_GQ_indices(state)
        cmapper, sm = create_normed_colours(max(0, 0.9*min(levels)), 1.2*max(levels), cmap="Blues_r", normalisation="LogNorm")
        for l in levels:
            _logger.logger.debug(f"Fitting level {l}")
            az.plot_hdi(self.stan_data[xmodel][...,idxs], ys, hdi_prob=l/100, ax=ax, plot_kwargs={"c":cmapper(l)}, fill_kwargs={"color":cmapper(l), "alpha":0.8, "label":f"{l}% HDI", "edgecolor":None}, smooth=False, hdi_kwargs={"skipna":True})
        if xobs is not None and yobs is not None:
            # overlay data
            obs = self.obs_collapsed if collapsed else self.obs
            if self._num_groups < 2:
                self._plot_obs_data_kwargs["cmap"] = "Set1"
            if yobs_err is None:
                ax.scatter(obs[xobs], obs[yobs], c=obs["label"], **self._plot_obs_data_kwargs)
            else:
                colvals = np.unique(obs["label"])
                ncols = len(colvals)
                cmapper, sm = create_normed_colours(np.min(colvals), np.max(colvals), cmap=self._plot_obs_data_kwargs["cmap"])
                for i, c in enumerate(colvals):
                    col = cmapper(c)
                    mask = obs["label"]==c
                    ax.errorbar(obs[xobs][mask], obs[yobs][mask], yerr=obs[yobs_err][mask], c=col, zorder=20, fmt=".", label=("Sims." if i==ncols-1 else ""))
        if show_legend:
            ax.legend()
        return ax


    def plot_predictive(self, xmodel, ymodel, xobs=None, yobs=None, yobs_err=None, levels=None, ax=None, collapsed=True, show_legend=True, save=True):
        """
        Predictive plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(xmodel=xmodel, ymodel=ymodel, state="pred", xobs=xobs, yobs=yobs, yobs_err=yobs_err, levels=levels, ax=ax, collapsed=collapsed, show_legend=show_legend)
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"pred_{yobs}"), fig=fig)


    def posterior_OOS_plot(self, xmodel, ymodel, levels=None, ax=None, collapsed=True, save=True, show_legend=True):
        """
        Posterior out-of-sample plot, observed data is not added to plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(xmodel=xmodel, ymodel=ymodel, state="OOS", levels=levels, ax=ax, collapsed=collapsed, show_legend=show_legend)
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"posterior_OOS_{ymodel}"), fig=fig)




class FactorModel_2D(_StanModel):
    def __init__(self, model_file, prior_file, figname_base, num_OOS, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, num_OOS, rng)


    @abstractmethod
    def extract_data(self):
        return super().extract_data()


    @abstractmethod
    def set_stan_data(self):
        return super().set_stan_data()


    @abstractmethod
    def _set_stan_data_OOS(self):
        return super()._set_stan_data_OOS()


    @abstractmethod
    def sample_model(self, sample_kwargs={}):
        return super().sample_model(sample_kwargs)


    @abstractmethod
    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        return super().sample_generated_quantity(gq, force_resample, state)


    def _plot_predictive(self, xmodel, ymodel, state, xobs=None, yobs=None, yobs_err=None, levels=None, ax=None, collapsed=True, show_legend=True):
        if levels is None:
            levels = self._default_hdi_levels
        levels.sort(reverse=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
        idxs = self._get_GQ_indices(state)
        ys = self.sample_generated_quantity(ymodel, state=state)
        print(ys)
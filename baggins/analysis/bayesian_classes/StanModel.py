from abc import ABC, abstractmethod
import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, collections, patches, ticker
from datetime import datetime, timezone
import cmdstanpy
import arviz as az
import yaml
from baggins.plotting import savefig, create_normed_colours
from baggins.env_config import figure_dir, data_dir, TMPDIRs, _cmlogger
from baggins.utils import get_mod_time

__all__ = [
    "_StanModel",
    "HierarchicalModel_1D",
    "HierarchicalModel_2D",
    "FactorModel_2D",
]

_logger = _cmlogger.getChild(__name__)


class _StanModel(ABC):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
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
        rng :  np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        self._model_file = model_file
        self._prior_file = prior_file
        self.figname_base = figname_base
        self._num_OOS = None
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
        self._exec_file = None
        self._parameter_diagnostic_plots_counter = 0
        self._gq_distribution_plot_counter = 0
        self._group_par_counter = 0
        # corner plot method doesn't save figure --> ensures first plot index 0
        self._parameter_corner_plot_counter = -1
        self._trace_plot_cols = None
        self._observation_mask = True
        self._plot_obs_data_kwargs = {
            "marker": "o",
            "linewidth": 0.5,
            "edgecolor": "k",
            "label": "Sims.",
            "cmap": "PuRd",
        }
        self._default_hdi_levels = [99, 75, 50, 25]
        self._num_groups = 0
        self._loaded_from_file = False
        self._generated_quantities = None
        self._obs_collapsed = {}
        self._obs_collapsed_names = []
        self._input_data_file_count = 0
        self._input_data_files = {}

        # properties which are defined in child classes
        self._latent_qtys = None
        self._folded_qtys = None

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
            _logger.exception(
                "Input to property `stan_data` must be a dict!", exc_info=True
            )
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
                _logger.exception(
                    "HMQ directory must be given if not loaded from file!",
                    exc_info=True,
                )
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
            _logger.exception(
                f"Observational data must be a dict! Current type is {type(d)}",
                exc_info=True,
            )
            raise
        for i, (k, v) in enumerate(d.items()):
            if set_categorical:
                try:
                    assert k != "label"
                except AssertionError:
                    _logger.exception("Keyword 'label' is reserved!", exc_info=True)
                    raise
            try:
                assert isinstance(v, list)
            except AssertionError:
                _logger.exception(
                    f"Data format must be a list! Currently '{k}' type {type(v)}",
                    exc_info=True,
                )
                raise
            for j, vv in enumerate(v):
                try:
                    assert isinstance(vv, (list, np.ndarray))
                    if isinstance(vv, list):
                        d[k][j] = np.array(vv)
                except AssertionError:
                    _logger.exception(
                        f"Observed variable {k} element {j} is not list-like, but type {type(vv)}",
                        exc_info=True,
                    )
                    raise
                try:
                    assert not np.any(np.isnan(d[k][j]))
                except AssertionError:
                    _logger.exception(
                        f"NaN values detected in observed variable {k} item {j}! This can lead to undefined behaviour.",
                        exc_info=True,
                    )
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
        if newkey in self.obs.keys():
            _logger.warning(
                f"Requested key {newkey} already exists! Transformation will not be reapplied --> Skipping."
            )
        else:
            _logger.debug(f"Applying transformation designated by {newkey}")
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
                    _logger.exception(
                        f"Observation transformation failed for keys {key}: differing dimensions: {dims}",
                        exc_info=True,
                    )
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
        order : str
            concatenation order for numpy, by default "F"
        """
        if "label" not in obs_names:
            obs_names.append("label")
        dim = {}
        order = order.upper()
        for obs_name in obs_names:
            try:
                assert obs_name not in self._obs_collapsed_names
                self._obs_collapsed_names.append(obs_name)
                dim[obs_name] = self.obs[obs_name][0].ndim
            except AssertionError:
                _logger.exception(
                    f"Observation {obs_name} has already been collapsed! Cannot collapse again!",
                    exc_info=True,
                )
                raise
            except IndexError:
                _logger.exception(f"Error collapsing {obs_name}, {self.obs[obs_name]}")
                raise
        try:
            assert max(dim.values()) < 3
        except AssertionError:
            _logger.exception(
                f"Error collapsing observation {max(dim, key=dim.get)}: data cannot have more than 2 dimensions",
                exc_info=True,
            )
            raise
        if max(dim.values()) == 1:
            # all observations are 1D, can just concatenate
            for k, v in self.obs.items():
                if k not in obs_names:
                    _logger.debug(f"Observation {k} will not be collapsed.")
                    continue
                self._obs_collapsed[k] = np.concatenate(v)
                _logger.debug(f"Collapsing variable {k}")
        else:
            # collapse the desired variable
            # need to collapse 2D variables first so we know how many tiles to
            # make of 1D variables
            dim = dict(sorted(dim.items(), key=lambda item: item[1], reverse=True))
            reps = [
                v.shape[0 if order == "F" else 1] for v in self.obs[list(dim.keys())[0]]
            ]
            for obs_name, ndim in dim.items():
                if ndim == 2:
                    self._obs_collapsed[obs_name] = np.concatenate(
                        [v.flatten(order=order) for v in self.obs[obs_name]]
                    )
                else:
                    self._obs_collapsed[obs_name] = np.concatenate(
                        [np.tile(vv, r) for vv, r in zip(self.obs[obs_name], reps)]
                    )
                _logger.debug(f"Collapsing variable {obs_name}")
        self._num_obs_collapsed = len(self.obs_collapsed[obs_name])
        # data consistency checks
        try:
            for k, v in self._obs_collapsed.items():
                assert v.shape == self._obs_collapsed[obs_names[0]].shape
        except AssertionError:
            _logger.exception(
                f"Data lengths are inconsistent in collapsed observations! Observation '{k}' has shape {v.shape}, must have shape {self._obs_collapsed[obs_names[0]].shape}",
                exc_info=True,
            )
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
        for k, v in self.obs.items():
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
            _logger.exception(
                f"There are not enough groups to form a valid hierarchical model. Minimum number of groups is {pars['stan']['min_num_samples']}, and we have {self.num_groups}!",
                exc_info=True,
            )
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
                f"file{self._input_data_file_count:03d}": {
                    "path": f,
                    "created": os.path.getmtime(f),
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
            _logger.exception(
                f"Dataset can only have three dimensions: chain, draw, and other. Currently has size {len(n)+2}",
                exc_info=True,
            )
            raise
        self._num_OOS = n[0]

    def _sampler(self, prior=False, sample_kwargs={}, diagnose=True, pathfinder=True):
        """
        Sample a stan model

        Parameters
        ----------
        prior : bool, optional
            run sampler for prior model, by default False
        sample_kwargs : dict, optional
            kwargs to be passed to CmdStanModel.sample(), by default {}
        diagnose : bool, optional
            diagnose the fit (should always be done), by default True
        pathfinder : bool, optional
            use pathfinder algorithm to optimise chain initialisation, by
            default True

        Returns
        -------
        cmdstanpy.CmdStanMCMC
            container output from stan sampling
        """
        if self._loaded_from_file:
            _logger.warning(
                "Instance instantiated from file: sampling the model again is not possible --> Skipping."
            )
            self._sample_diagnosis = self._fit.diagnose()
            return self._fit
        else:
            default_sample_kwargs = {
                "chains": 4,
                "iter_sampling": 2000,
                "show_progress": True,
                "max_treedepth": 12,
            }
            if not prior:
                default_sample_kwargs["output_dir"] = os.path.join(
                    data_dir, "stan_files"
                )
                default_sample_kwargs["threads_per_chain"] = 4
            else:
                # protect against inability to parallelise, for prior model
                # this shouldn't be so expensive anyway
                default_sample_kwargs["force_one_process_per_chain"] = True
            # update user given sample kwargs
            for k, v in sample_kwargs.items():
                default_sample_kwargs[k] = v
            if "output_dir" in default_sample_kwargs:
                os.makedirs(default_sample_kwargs["output_dir"], exist_ok=True)
            start_time = datetime.now()
            if prior:
                fit = self._prior_model.sample(self.stan_data, **default_sample_kwargs)
            else:
                _logger.debug(f"exe info: {self._model.exe_info()}")
                try:
                    if pathfinder:
                        pf = self._model.pathfinder(
                            data=self.stan_data, show_console=True
                        )
                        inits = pf.create_inits()
                    else:
                        inits = None
                except (RuntimeError, ValueError) as e:
                    _logger.warning(
                        f"Stan pathfinder failed: normal initialisation will be used! Reason: {e}"
                    )
                    inits = None
                fit = self._model.sample(
                    self.stan_data, inits=inits, **default_sample_kwargs
                )
                self._write_input_data_yml(fit.runset.csv_files[0])
            _logger.info(f"Sampling completed in {datetime.now()-start_time}")
            _logger.info(f"Number of threads used: {os.environ['STAN_NUM_THREADS']}")
            if diagnose:
                self._sample_diagnosis = fit.diagnose()
                _logger.info(f"\n{fit.summary(sig_figs=4)}")
                _logger.info(f"\n{self.sample_diagnosis}")
            else:
                _logger.warning("No diagnosis will be done on fit!")
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
            self._model = cmdstanpy.CmdStanModel(
                stan_file=self._model_file,
            )

    @abstractmethod
    def sample_model(self, sample_kwargs={}, diagnose=True, pathfinder=True):
        """
        Wrapper function around _sampler() to sample a stan likelihood model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
             kwargs to be passed to CmdStanModel.sample(), by default {}
        diagnose : bool, optional
            diagnose the fit (should always be done), by default True
        pathfinder : bool, optional
            use pathfinder algorithm to optimise chain initialisation, by
            default True
        """
        if self._model is None and not self._loaded_from_file:
            self.build_model()
        self._fit = self._sampler(
            sample_kwargs=sample_kwargs, diagnose=diagnose, pathfinder=pathfinder
        )
        # TODO capture arviz warnings about NaN
        self._fit_for_az = az.from_cmdstanpy(posterior=self._fit)
        if diagnose:
            # prior sensitivity only done for posterior model
            self._fit_for_az.add_groups(
                {"log_prior": self._fit_for_az["posterior"]["lprior"]}
            )
            priorsens = az.psens(self._fit_for_az, var_names=self._latent_qtys)
            _logger.info(
                f"Maximum CJS distance for latent variables:\n {priorsens.max()}"
            )

    def sample_prior(self, sample_kwargs={}, diagnose=True):
        """
        Wrapper function around _sampler() to sample a stan prior model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
            kwargs to be passed to CmdStanModel.sample(), by default {}
        diagnose : bool, optional
            diagnose the fit (should always be done), by default True
        """
        if self._prior_model is None and not self._loaded_from_file:
            self.build_model(prior=True)
        self._prior_fit = self._sampler(
            sample_kwargs=sample_kwargs, prior=True, diagnose=diagnose
        )
        self._fit_for_az = az.from_cmdstanpy(prior=self._prior_fit)

    def _get_GQ_indices(self, state, collapsed=False):
        """
        Get the indices of a generated quantity block for either predictive
        inference or out of sample inference

        Parameters
        ----------
        state : str
            inference type, must be one of 'pred' or 'OOS'
        collapsed : bool, optional
            has the variable been collapsed (1-dimensional), by default False

        Returns
        -------
        np.ndarray
            array of indices
        """
        dividing_idx = self.num_obs_collapsed if collapsed else self.num_obs
        return (
            np.r_[0:dividing_idx]
            if state == "pred"
            else np.r_[dividing_idx : self.num_OOS + dividing_idx]
        )

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

        def _choose_model():
            # determine if we should use the prior or posterior model
            if self._model is None:
                _logger.debug("Generated quantities will be taken from the prior model")
                return self._prior_model, self._prior_fit
            else:
                _logger.debug(
                    "Generated quantities will be taken from the posterior model"
                )
                return self._model, self._fit

        try:
            if self.generated_quantities is None or force_resample:
                _model, _fit = _choose_model()
                self._generated_quantities = _model.generate_quantities(
                    data=self.stan_data, previous_fit=_fit
                )
            else:
                _logger.debug(
                    "Generated quantities already exist and will not be resampled"
                )
            self.generated_quantities.stan_variable(gq)
        except ValueError as e:
            _model, _fit = _choose_model()
            TMPDIRs.make_new_dir()
            _logger.error(
                f"{e}\n > Value error trying to read generated quantities data: creating temporary directory {TMPDIRs.register[-1]}"
            )
            self._generated_quantities = _model.generate_quantities(
                data=self.stan_data,
                previous_fit=_fit,
                gq_output_dir=TMPDIRs.register[-1],
            )
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
        """if self._fit is None:
            _fit = self._prior_fit
            _logger.debug("Generated quantities will be taken from the prior model")
        else:
            _fit = self._fit
            _logger.debug("Generated quantities will be taken from the posterior model")"""
        x, dens = az.kde(self.generated_quantities.stan_variables()[v])
        return x[np.nanargmax(dens)]

    def _parameter_corner_plot(
        self,
        var_names,
        figsize=None,
        labeller=None,
        levels=None,
        combine_dims=None,
        backend_kwargs=None,
        divergences=True,
    ):
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
        divergences : bool, optional
            plot divergences, by default True

        Returns
        -------
        ax : matplotlib.axes.Axes
            corner plot
        """
        if levels is None:
            levels = self._default_hdi_levels
        levels.sort(reverse=True)
        levels = [lev / 100 for lev in levels]
        # show divergences on plots where no dimension combination has
        # occurred: combining dimensions changes the length of boolean mask
        # "diverging_mask" in arviz --> index mismatch error
        divergences = divergences if combine_dims is None else False
        if "prior" in self._fit_for_az.groups():
            group = "prior"
            divergences = False
        else:
            group = "posterior"
        num_vars = len(var_names)
        with az.rc_context(
            {"plot.max_subplots": num_vars**2 - np.sum(np.arange(num_vars)) + 1}
        ):
            # first lay down the markers
            # ax = az.plot_pair(self._fit_for_az, group=group, var_names=var_names, kind="scatter", marginals=True, combine_dims=combine_dims, scatter_kwargs={"marker":".", "alpha":0.1, "s":10}, figsize=figsize, labeller=labeller, textsize=rcParams["font.size"], backend_kwargs=backend_kwargs)
            # then add the KDE
            try:
                ax = az.plot_pair(
                    self._fit_for_az,
                    group=group,
                    var_names=var_names,
                    kind="kde",
                    divergences=divergences,
                    combine_dims=combine_dims,
                    figsize=figsize,
                    marginals=True,
                    kde_kwargs={
                        "contour_kwargs": {"linewidths": 0.5},
                        "hdi_probs": levels,
                        "contourf_kwargs": {"cmap": "Blues"},
                    },
                    point_estimate_marker_kwargs={"marker": ""},
                    labeller=labeller,
                    textsize=rcParams["font.size"],
                    backend_kwargs=backend_kwargs,
                )
            except ValueError:
                _logger.error(
                    "HDI interval cannot be determined for corner plots! KDE levels will not correspond to a particular HDI, but follow matplotlib contour defaults"
                )
                ax = az.plot_pair(
                    self._fit_for_az,
                    group=group,
                    var_names=var_names,
                    kind="kde",
                    divergences=divergences,
                    combine_dims=combine_dims,
                    figsize=figsize,
                    marginals=True,
                    kde_kwargs={
                        "contour_kwargs": {"linewidths": 0.5},
                        "contourf_kwargs": {"cmap": "Blues"},
                    },
                    point_estimate_marker_kwargs={"marker": ""},
                    labeller=labeller,
                    textsize=rcParams["font.size"],
                    backend_kwargs=backend_kwargs,
                )
        return ax

    def parameter_corner_plot(
        self,
        var_names,
        figsize=None,
        labeller=None,
        levels=None,
        combine_dims=None,
        backend_kwargs=None,
    ):
        """
        See docs for _parameter_corner_plot()
        """
        ax = self._parameter_corner_plot(
            var_names,
            figsize=figsize,
            labeller=labeller,
            levels=levels,
            combine_dims=combine_dims,
            backend_kwargs=backend_kwargs,
        )
        self._parameter_corner_plot_counter += 1
        return ax

    def parameter_diagnostic_plots(
        self, var_names, figsize=None, labeller=None, levels=None
    ):
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
        # XXX: LaTeX labels will not render if the label is longer than the
        # plotting axis
        if figsize is None:
            max_dim = max(rcParams["figure.figsize"])
            figsize = (max_dim, max_dim)
        try:
            assert len(var_names) > 1
        except AssertionError:
            _logger.exception(
                "Pair plot requires at least two variables!", exc_info=True
            )
            raise

        # set trace colour scheme
        if self._parameter_diagnostic_plots_counter == 0:
            vmax = len(self._fit_for_az.posterior["chain"])
            cmapper, sm = create_normed_colours(-vmax / 2, vmax, cmap="Blues")
            self._trace_plot_cols = [
                cmapper(x) for x in self._fit_for_az.posterior["chain"]
            ]
        # limit to 4 variables per plot: figures will be saved with an
        # additional index in the name, e.g. 0-0.png
        num_var_per_plot = 4
        for i in range(0, len(var_names), num_var_per_plot):
            # plot trace
            ax = az.plot_trace(
                self._fit_for_az,
                var_names=var_names[i : i + num_var_per_plot],
                figsize=figsize,
                chain_prop={"color": self._trace_plot_cols},
                trace_kwargs={"alpha": 0.9},
                labeller=labeller,
            )
            fig = ax.flatten()[0].get_figure()
            savefig(
                self._make_fig_name(
                    self.figname_base,
                    f"trace_{self._parameter_diagnostic_plots_counter}-{i//num_var_per_plot}",
                ),
                fig=fig,
            )
            plt.close(fig)

            # plot rank
            ax = az.plot_rank(
                self._fit_for_az,
                var_names=var_names[i : i + num_var_per_plot],
                labeller=labeller,
            )
            fig = ax.flatten()[0].get_figure()
            savefig(
                self._make_fig_name(
                    self.figname_base,
                    f"rank_{self._parameter_diagnostic_plots_counter}-{i//num_var_per_plot}",
                ),
                fig=fig,
            )
            plt.close(fig)

        # plot pair
        ax = self._parameter_corner_plot(
            var_names=var_names, figsize=figsize, labeller=labeller, levels=levels
        )
        fig = ax.flatten()[0].get_figure()
        savefig(
            self._make_fig_name(
                self.figname_base, f"pair_{self._parameter_diagnostic_plots_counter}"
            ),
            fig=fig,
        )
        plt.close(fig)
        self._parameter_diagnostic_plots_counter += 1

    def group_parameter_plot(
        self, var_names, figsize=None, levels=None, xlabel="Factor", ylabels=None
    ):
        """
        Plot group parameter variation as a sequence of shaded regions.

        Parameters
        ----------
        var_names : list
            variable names to plot
        figsize : tuple, optional
            figure size, by default None
        levels : list, optional
            HDI intervals to plot, by default None
        xlabel : str, optional
            x axis label, by default "Factor"
        ylabels : list, optional
            y axis labels, by default None
        """
        num_vars = len(var_names)
        if figsize is None:
            figsize = (6, num_vars)
        if levels is None:
            levels = self._default_hdi_levels
        levels = [lev / 100 for lev in levels]
        levels.sort(reverse=True)
        az_group = "prior"
        cmapper, sm = create_normed_colours(
            max(0, 0.9 * min(levels)), 1.3, cmap="Blues_r", trunc=(None, max(levels))
        )
        fig, ax = plt.subplots(num_vars, 1, sharex="all", figsize=figsize)
        ax[-1].set_xlabel(xlabel)
        # loop through variables
        for i, v in enumerate(var_names):
            # loop through HDI levels
            for j, level in enumerate(levels):
                p = []
                hdi = az.hdi(
                    self._fit_for_az[az_group].get(v), hdi_prob=level, skipna=True
                )
                try:
                    lower = hdi[0]
                    upper = hdi[1]
                except KeyError:
                    lower = hdi[v][:, 0]
                    upper = hdi[v][:, 1]
                # loop through each group
                for k, (ll, uu) in enumerate(zip(lower, upper)):
                    r = patches.Rectangle((k - 0.5, ll), 1, uu - ll)
                    p.append(r)
                ax[i].add_collection(collections.PatchCollection(p, fc=cmapper(level)))
            ax[i].autoscale_view()
            for j in range(len(lower) - 1):
                ax[i].axvline(j + 0.5, c="k", alpha=0.4, lw=0.5)
            if ylabels is not None:
                ax[i].set_ylabel(ylabels[i])
        ax[-1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax[-1].set_xlim(-0.5, len(lower) - 0.5)
        for i, axi in enumerate(ax):
            if i == num_vars - 1:
                break
            axi.tick_params(axis="x", which="both", bottom=False)
        plt.colorbar(sm, label="HDI", location="top", ax=ax[0])
        savefig(
            self._make_fig_name(
                self.figname_base, f"group_par_{self._group_par_counter}"
            ),
            fig=fig,
        )
        plt.close(fig)
        self._group_par_counter += 1

    @abstractmethod
    def _plot_predictive(
        self,
        xmodel,
        ymodel,
        state,
        xobs=None,
        yobs=None,
        yobs_err=None,
        levels=None,
        ax=None,
        collapsed=True,
        show_legend=True,
    ):
        pass

    def plot_generated_quantity_dist(
        self,
        gq,
        state="pred",
        bounds=None,
        ax=None,
        xlabels=None,
        save=True,
        **kwargs,
    ):
        """
        Plot the 1-D distribution of an arbitrary variable in the generated quantities block of a stan model.

        Parameters
        ----------
        gq : list
            variables to plot
        state : str, optional
            return generated quantities for predictive checks or out-of-sample
            quantities, by default "pred"
        bounds : list
            list of tuples [(a,b), ..., (a,b)] giving the lower and upper bound for each variable in gq
        ax : matplotlib.axes.Axes or np.ndarray of, optional
            axes object to plot to, by default None
        xlabels : list, optional
            labels for the x-axis, by default None
        save : bool, optional
            save the plot, by default True
        **kwargs :
            other parameters parsed to arviz.plot_dist()

        Returns
        -------
        matplotlib.axes.Axes
            plotting axes
        """
        if ax is None:
            fig, ax = plt.subplots(len(gq), 1)
        elif isinstance(ax, (np.ndarray, list)):
            fig = np.atleast_2d(ax)[0, 0].get_figure()
        else:
            fig = ax.get_figure()
        if isinstance(ax, np.ndarray):
            ax_shape = ax.shape
        else:
            ax = np.array(ax)
            ax_shape = (1,)
        ax = ax.flatten()
        try:
            assert isinstance(gq, list)
        except AssertionError:
            _logger.exception(f"Input `gq` must be of type <list>, not {type(gq)}!")
            raise
        if bounds is not None:
            try:
                assert len(bounds) == len(gq)
                for b in bounds:
                    assert len(b) == 2
            except AssertionError:
                _logger.exception(
                    "Setting bounds requires the `bounds` argument to have the same length as `gq`, and each entry must be a 2-tuple",
                    exc_info=True,
                )
                raise
        if xlabels is None:
            xlabels = gq
        else:
            assert isinstance(xlabels, list)
            try:
                assert len(gq) == len(xlabels)
            except AssertionError:
                _logger.exception(
                    f"There are {len(gq)} generated quantity variables to plot, but only {len(xlabels)} labels!",
                    exc_info=True,
                )
                raise
        for i, (_gq, lab) in enumerate(zip(gq, xlabels)):
            ys = self.sample_generated_quantity(_gq, state=state)
            if bounds is not None:
                if bounds[i][0] is not None:
                    mask_lower = ys > bounds[i][0]
                else:
                    mask_lower = True
                if bounds[i][1] is not None:
                    mask_upper = ys <= bounds[i][1]
                else:
                    mask_upper = True
                ys = ys[np.logical_and(mask_lower, mask_upper)]
            try:
                assert len(ys.shape) < 3
            except AssertionError:
                _logger.exception(
                    f"Generated quantity {_gq} must have shape 2, has shape {len(ys.shape)}",
                    exc_info=True,
                )
                raise
            cumulative = kwargs.pop("cumulative", False)
            az.plot_dist(ys, ax=ax[i], cumulative=cumulative, **kwargs)
            ax[i].set_xlabel(lab)
            ax[i].set_ylabel(r"$\mathrm{CDF}$" if cumulative else r"$\mathrm{PDF}$")
        ax.reshape(ax_shape)
        if save:
            savefig(
                self._make_fig_name(
                    self.figname_base, f"gqs_{self._gq_distribution_plot_counter}"
                ),
                fig=fig,
            )
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
        group = "prior" if "prior" in self._fit_for_az.groups() else "posterior"
        qvals = self._fit_for_az[group][vars].quantile(quantiles).to_dataframe()
        vars = vars.copy()
        vars.insert(0, "Variable")
        max_str_len = max([len(v) for v in vars]) + 1
        head_str = (
            f"\n{vars[0]:>{max_str_len}}          5%        50%        95%        IQR  "
        )
        print(head_str)
        dashes = ["-" for _ in range(len(head_str))]
        print("".join(dashes))
        for v in vars[1:]:
            _iqr = qvals.loc[0.75, v] - qvals.loc[0.25, v]
            print(
                f"{v:>{max_str_len}}:  {qvals.loc[0.05,v]:>+.2e}  {qvals.loc[0.50,v]:>+.2e}  {qvals.loc[0.95,v]:>+.2e}  {_iqr:>+.2e}"
            )
        print()

    def determine_loo(self, stan_log_lik="log_lik"):
        """
        Determine Leave-One Out Statistics (wrapper for arviz method)

        Parameters
        ----------
        stan_log_lik : str, optional
            name of log-likelihood variable in stan code, by default "log_lik"

        Returns
        -------
        loo : arviz.ELPDData
            elpd data object from arviz
        """
        if self.generated_quantities is None:
            self.sample_generated_quantity(stan_log_lik, state="pred")
        if "log_likelihood" not in self._fit_for_az:
            self.sample_generated_quantity(stan_log_lik, state="pred")
            self._fit_for_az.add_groups(
                {"log_likelihood": self.generated_quantities.draws_xr(stan_log_lik)}
            )
        loo = az.loo(self._fit_for_az)
        print(loo)
        return loo

    def rename_dimensions(self, dim_map):
        """
        Rename dimensions of arviz InferenceData object
        TODO: is it worth keeping this method?

        Parameters
        ----------
        dim_map : dict
            mapping of old dimension names to new names
        """
        self._fit_for_az.rename_dims(dim_map, inplace=True)

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
            _logger.exception(
                f"Expanding variables requires first arugment to be a list or tuple, not {type(varnames)}",
                exc_info=True,
            )
            raise
        group = "prior" if "prior" in self._fit_for_az.groups() else "posterior"
        for k in varnames:
            self._fit_for_az[group][k] = self._fit_for_az[group][k].expand_dims(
                {dim: np.arange(self._fit_for_az[group].dims[dim])}, axis=-1
            )

    @classmethod
    def load_fit(cls, fit_files, figname_base, rng=None):
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
        C = cls(figname_base=figname_base, rng=rng)

        # set up the model, be aware of changes between sampling and loading
        C.build_model()
        C._fit = cmdstanpy.from_csv(fit_files)
        fit_time = datetime.strptime(
            C._fit.metadata.cmdstan_config["start_datetime"], "%Y-%m-%d %H:%M:%S %Z"
        ).replace(tzinfo=timezone.utc)
        model_build_time = datetime.fromtimestamp(
            get_mod_time(C._model.exe_file), tz=timezone.utc
        )
        if model_build_time.timestamp() > fit_time.timestamp():
            print("==========================================")
            _logger.error(
                f"Stan executable {C._model.exe_file} has been modified since sampling was performed! This could be due to `git checkout`. Check the file update time with `git log -- {C.model_file}`. Proceed with caution!\n  --> Compile time: {model_build_time} UTC\n  --> Sample time:  {fit_time}"
            )
            print("==========================================")

        # load path to observation data
        try:
            # assume a glob pattern was parsed
            tstamp = (
                os.path.basename(fit_files).split("-")[-1].split("_")[0].split("*")[0]
            )
            dir_name = os.path.dirname(fit_files)
        except TypeError:
            # actually a list of files was parsed
            tstamp = os.path.basename(fit_files[0]).split("-")[-1].split("_")[0]
            dir_name = os.path.dirname(fit_files[0])
        with open(os.path.join(dir_name, f"input_data-{tstamp}.yml"), "r") as f:
            C._input_data_files = yaml.safe_load(f)
        for v in C._input_data_files.values():
            if os.path.getmtime(v["path"]) > v["created"]:
                _logger.error(
                    f"HMQ file {v['path']} has been modified since the Stan model was run, proceed with caution!"
                )
        C._loaded_from_file = True
        return C


class HierarchicalModel_1D(_StanModel):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)

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
    def sample_model(self, sample_kwargs={}, diagnose=True):
        return super().sample_model(sample_kwargs, diagnose=diagnose)

    @abstractmethod
    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        return super().sample_generated_quantity(gq, force_resample, state)

    def _plot_predictive(
        self,
        xmodel,
        state,
        xobs=None,
        xobs_err=None,
        levels=None,
        ax=None,
        collapsed=True,
    ):
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
        quantiles = [0.5 - lev / 200 for lev in levels]
        quantiles.extend([0.5 + lev / 200 for lev in levels])
        quantiles.sort()
        if ax is None:
            fig, ax = plt.subplots(1, 1, squeeze=False)
        obs = self.obs_collapsed if collapsed else self.obs
        xs = self.sample_generated_quantity(xmodel, state=state)
        az.plot_dist(xs, quantiles=quantiles, ax=ax)
        # overlay data
        if xobs_err is None:
            ax.scatter(
                obs[xobs],
                np.zeros(len(obs[xobs])),
                c=obs["label"],
                **self._plot_obs_data_kwargs,
            )
        else:
            colvals = np.unique(obs["label"])
            ncols = len(colvals)
            cmapper, sm = create_normed_colours(
                np.min(colvals),
                np.max(colvals),
                cmap=self._plot_obs_data_kwargs["cmap"],
            )
            for i, c in enumerate(colvals):
                col = cmapper(c)
                mask = obs["label"] == c
                ys = np.zeros(len(obs[xobs][mask]))
                ax.scatter(
                    obs[xobs][mask],
                    np.zeros(len(obs[xobs][mask])),
                    color=col,
                    **self._plot_obs_data_kwargs,
                )
                ax.errorbar(
                    obs[xobs][mask],
                    ys,
                    xerr=obs[xobs_err][mask],
                    c=col,
                    zorder=20,
                    fmt=".",
                    label=("Sims." if i == ncols - 1 else ""),
                )

    def plot_predictive(
        self,
        xmodel,
        xobs,
        xobs_err=None,
        levels=None,
        ax=None,
        collapsed=True,
        save=True,
    ):
        """
        Predictive plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(
            xmodel=xmodel,
            state="pred",
            xobs=xobs,
            xobs_err=xobs_err,
            levels=levels,
            collapsed=collapsed,
        )
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"pred_{xobs}"), fig=fig)
        return ax

    def posterior_OOS_plot(
        self, xmodel, levels=None, ax=None, collapsed=True, save=True
    ):
        """
        Posterior out-of-sample plot, observed data is not added to plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(
            xmodel=xmodel, state="OOS", levels=levels, collapsed=collapsed
        )
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"OOS_{xmodel}"), fig=fig)
        return ax


class HierarchicalModel_2D(_StanModel):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)

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
    def sample_model(self, **kwargs):
        return super().sample_model(**kwargs)

    @abstractmethod
    def sample_generated_quantity(self, gq, force_resample=False, state="pred"):
        return super().sample_generated_quantity(gq, force_resample, state)

    def reduce_obs_between_groups(self, ivar, key, newkey, func):
        """
        Apply a reduction algorithm element-wise between groups. This only
        makes sense for regression data, where we might have N observations of
        a dependent variable y at the same independent variable x, and want for
        example the mean of those points.

        Parameters
        ----------
        ivar : str
            indepedent variable name
        key : str
            variable to reduce
        newkey : str
            new variable name
        func : callable
            reduction function
        """
        if newkey in self.obs.keys():
            _logger.warning(
                f"Requested key {newkey} already exists! Transformation will not be reapplied --> Skipping."
            )
        else:
            try:
                assert isinstance(key, str)
            except AssertionError:
                _logger.exception(
                    f"'key' must be a str, not {type(key)}", exc_info=True
                )
                raise
            try:
                for i in range(1, self._num_groups):
                    assert np.allclose(self.obs[ivar][0], self.obs[ivar][i])
            except AssertionError:
                _logger.exception(
                    "Independent variable arrays must all have the same shape and be element-wise equal",
                    exc_info=True,
                )
                raise
            self.obs[f"{ivar}_reduced"] = self.obs[ivar][0]
            self.obs[newkey] = np.full_like(self.obs[f"{ivar}_reduced"], np.nan)
            for i in range(len(self.obs[key][0])):
                self.obs[newkey][i] = func(
                    [self.obs[key][n][i] for n in range(self._num_groups)]
                )
            for k in (f"{ivar}_reduced", newkey):
                self.obs[k] = np.atleast_2d(self.obs[k])

    def _make_default_hdi_colours(self, levels):
        """
        Create the default colour scheme for HDI regression plots. Basically a wrapper around create_normed_colours()

        Parameters
        ----------
        levels : list
            HDI levels

        Returns
        -------
        : function
            takes an argument in the range [vmin, vmax] and returns the scaled
            colour
        : matplotlib.cm.ScalarMappable
            object that is required for creating a colour bar
        """
        return create_normed_colours(
            max(0, 0.9 * min(levels)), 1.2 * max(levels), cmap="Blues_r", norm="LogNorm"
        )

    def _plot_predictive(
        self,
        xmodel,
        ymodel,
        state,
        xobs=None,
        yobs=None,
        yobs_err=None,
        levels=None,
        ax=None,
        collapsed=True,
        show_legend=True,
        smooth=False,
    ):
        """
        Plot a predictive check for a regression stan model.

        Parameters
        ----------
        xmodel : str
            dictionary key for modelled independent variable
        ymodel : str
            dictionary key for modelled dependent variable
        state : str
            predictive or OOS samples, must be one of 'pred' or 'OOS'
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
            fig, ax = plt.subplots(1, 1)
        if isinstance(ymodel, str):
            ys = self.sample_generated_quantity(ymodel, state=state)
        else:
            _logger.warning("Plotting an array outside of generated_quantities")
            ys = ymodel
        cmapper, sm = self._make_default_hdi_colours(levels)
        for lev in levels:
            _logger.debug(f"Fitting level {lev}")
            az.plot_hdi(
                self.stan_data[xmodel],
                ys,
                hdi_prob=lev / 100,
                ax=ax,
                plot_kwargs={"c": cmapper(lev)},
                fill_kwargs={
                    "color": cmapper(lev),
                    "alpha": 0.8,
                    "label": f"{lev}% HDI",
                    "edgecolor": None,
                },
                smooth=smooth,
                hdi_kwargs={"skipna": True},
            )
        if xobs is not None and yobs is not None:
            # overlay data
            obs = self.obs_collapsed if collapsed else self.obs
            if self._num_groups < 2:
                self._plot_obs_data_kwargs["cmap"] = "Set1"
            if yobs_err is None:
                ax.scatter(
                    obs[xobs], obs[yobs], c=obs["label"], **self._plot_obs_data_kwargs
                )
            else:
                colvals = np.unique(obs["label"])
                ncols = len(colvals)
                cmapper, sm = create_normed_colours(
                    np.min(colvals),
                    np.max(colvals),
                    cmap=self._plot_obs_data_kwargs["cmap"],
                )
                for i, c in enumerate(colvals):
                    col = cmapper(c)
                    mask = obs["label"] == c
                    ax.errorbar(
                        obs[xobs][mask],
                        obs[yobs][mask],
                        yerr=obs[yobs_err][mask],
                        c=col,
                        zorder=20,
                        fmt=".",
                        label=("Sims." if i == ncols - 1 else ""),
                    )
        if show_legend:
            ax.legend()
        return ax

    def plot_predictive(
        self,
        xmodel,
        ymodel,
        xobs=None,
        yobs=None,
        yobs_err=None,
        levels=None,
        ax=None,
        collapsed=True,
        show_legend=True,
        smooth=False,
        save=True,
    ):
        """
        Predictive plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(
            xmodel=xmodel,
            ymodel=ymodel,
            state="pred",
            xobs=xobs,
            yobs=yobs,
            yobs_err=yobs_err,
            levels=levels,
            ax=ax,
            collapsed=collapsed,
            show_legend=show_legend,
        )
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"pred_{yobs}"), fig=fig)

    def posterior_OOS_plot(
        self,
        xmodel,
        ymodel,
        levels=None,
        ax=None,
        collapsed=True,
        save=True,
        show_legend=True,
        smooth=False,
    ):
        """
        Posterior out-of-sample plot, observed data is not added to plot.
        See docs for _plot_predictive()
        """
        ax = self._plot_predictive(
            xmodel=xmodel,
            ymodel=ymodel,
            state="OOS",
            levels=levels,
            ax=ax,
            collapsed=collapsed,
            show_legend=show_legend,
            smooth=smooth,
        )
        fig = ax.get_figure()
        if save:
            savefig(self._make_fig_name(self.figname_base, f"OOS_{ymodel}"), fig=fig)


class FactorModel_2D(_StanModel):
    def __init__(self, model_file, prior_file, figname_base, rng=None) -> None:
        super().__init__(model_file, prior_file, figname_base, rng)

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

    def _set_factor_context_idxs(self, k):
        """
        Determine the factor and context indexing based off the dependent
        quantity being modelled.

        Parameters
        ----------
        k : str
            dependent variable used in the factor modelling
        """
        factor_idx = np.full(self.stan_data["N_contexts"], -99)
        idxf0, idxf1 = 0, 0
        context_idx = np.full(self.num_obs_collapsed, -99)
        idxc0, idxc1, last_idxc = 0, 0, 0
        for i, v in enumerate(self.obs[k], start=1):
            try:
                assert v.ndim == 2
            except AssertionError:
                _logger.exception(
                    f"Dependent quantity to be modelled must be two dimensional! Currently has {v.ndim} dimensions.",
                    exc_info=True,
                )
                raise
            idxf1 += v.shape[0]
            factor_idx[idxf0:idxf1] = i
            idxc1 += np.dot(*v.shape)
            _idxc = np.arange(1, v.shape[0] + 1) + last_idxc
            last_idxc = _idxc[-1]
            context_idx[idxc0:idxc1] = np.repeat(_idxc, v.shape[1])
            idxc0 = idxc1
            idxf0 = idxf1
        self.stan_data.update(dict(factor_idx=factor_idx, context_idx=context_idx))

    def _plot_predictive(
        self,
        xmodel,
        ymodel,
        state,
        xobs=None,
        yobs=None,
        yobs_err=None,
        levels=None,
        ax=None,
        collapsed=True,
        show_legend=True,
    ):
        if levels is None:
            levels = self._default_hdi_levels
        levels.sort(reverse=True)
        if ax is None:
            fig, ax = plt.subplots(self.num_groups, 1, sharex="all", sharey="all")
        else:
            ax = ax.flatten()
        ys = self.sample_generated_quantity(ymodel, state=state)
        # idxs = self._get_GQ_indices(state, collapsed=collapsed)
        obs_to_factor = np.full(self.num_obs_collapsed, 0)
        for i in range(self.num_obs_collapsed):
            obs_to_factor[i] = (
                self.stan_data["factor_idx"][self.stan_data["context_idx"][i] - 1] - 1
            )
        for i in range(self.num_groups):
            _logger.info(f"Creating HDI predictive for factor {i}")
            mask = obs_to_factor == i
            _ys = ys[:, mask]
            cmapper, sm = create_normed_colours(
                max(0, 0.9 * min(levels)),
                1.2 * max(levels),
                cmap="Blues_r",
                norm="LogNorm",
            )
            for lev in levels:
                _logger.debug(f"Fitting level {lev}")
                label = f"{lev}% HDI" if i == 0 else ""
                # TODO will this indexing work for GQ values in posterior pred?
                az.plot_hdi(
                    self.stan_data[xmodel][..., mask],
                    _ys,
                    hdi_prob=lev / 100,
                    ax=ax[i],
                    plot_kwargs={"c": cmapper(lev)},
                    fill_kwargs={
                        "color": cmapper(lev),
                        "alpha": 0.8,
                        "label": label,
                        "edgecolor": None,
                    },
                    smooth=False,
                    hdi_kwargs={"skipna": True},
                )
        if xobs is not None and yobs is not None:
            # overlay data
            obs = self.obs_collapsed if collapsed else self.obs
            if self._num_groups < 2:
                self._plot_obs_data_kwargs["cmap"] = "Set1"
            if yobs_err is None:
                ax.scatter(
                    obs[xobs], obs[yobs], c=obs["label"], **self._plot_obs_data_kwargs
                )
            else:
                colvals = np.unique(obs["label"])
                ncols = len(colvals)
                cmapper, sm = create_normed_colours(
                    np.min(colvals),
                    np.max(colvals),
                    cmap=self._plot_obs_data_kwargs["cmap"],
                )
                for i, c in enumerate(colvals):
                    col = cmapper(c)
                    mask = obs["label"] == c
                    ax.errorbar(
                        obs[xobs][mask],
                        obs[yobs][mask],
                        yerr=obs[yobs_err][mask],
                        c=col,
                        zorder=20,
                        fmt=".",
                        label=("Sims." if i == ncols - 1 else ""),
                    )
        if show_legend:
            ax[0].legend()
        return ax

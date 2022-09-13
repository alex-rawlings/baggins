import os
import numpy as np
import matplotlib.pyplot as plt
import cmdstanpy
import arviz as av


from ...plotting import savefig, create_normed_colours, mplColours
from ...utils import load_data
from ...env_config import figure_dir, data_dir, _logger

__all__ = ["StanModel"]


class StanModel:
    def __init__(self, model_file, prior_file, figname_base, rng=None, random_select_obs=None) -> None:
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
        random_select_obs : dict, optional
            dictionary specifying how to randomly select observations for 
            fitting (keys: num, group), by default None (all observations used)
        """
        self._model_file = model_file
        self._prior_file = prior_file
        self.figname_base = figname_base
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng
        self._obs_len = None
        self._stan_data = None
        self._model = None
        self._fit = None
        self._fit_for_av = None
        self._prior_stan_data = None
        self._prior_model = None
        self._prior_fit = None
        if random_select_obs is not None:
            assert all(k in random_select_obs.keys() for k in ["num", "group"])
            self.random_obs_select_dict = random_select_obs
            self._random_obs_select(**random_select_obs)
        self._parameter_plot_counter = 0
        self._observation_mask = True
        self._plot_obs_data_kwargs = {"marker":".", "linewidth":0.5, "edgecolor":"k", "label":"Obs.", "cmap":"PuRd"}
        self._num_groups = 0
        self._loaded_from_file = False
        self._generated_quantities = None


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
        self._check_observation_validity(d)
        self._obs = d
    
    @property
    def obs_len(self):
        return self._obs_len
    
    @property
    def num_groups(self):
        return self._num_groups
    
    @property
    def categorical_label(self):
        return self._categorical_label
    
    @categorical_label.setter
    def categorical_label(self, group):
        self._categorical_label = np.full(self.obs_len, 0, dtype=int)
        for i, g in enumerate(np.unique(self.obs[group])):
            mask = self.obs[group] == g
            self._categorical_label[mask] = i
            self._num_groups += 1
    
    @property
    def figname_base(self):
        return self._figname_base
    
    @figname_base.setter
    def figname_base(self, f):
        self._figname_base = os.path.join(figure_dir, f)
        d = os.path.join(figure_dir, f[::-1].partition("/")[-1][::-1])
        os.makedirs(d, exist_ok=True)
    
    @property
    def generated_quantities(self):
        return self._generated_quantities
    

    def _check_observation_validity(self, d):
        """
        Ensure that an observation is numerically valid: it is a dict, no NaN 
        values, and each member of the dict is mutable to the same shape

        Parameters
        ----------
        d : any
            proposed value to set the observations to. Should be a dict, but an
            error is thrown if it is not.
        """
        try:
            assert isinstance(d, dict)
        except AssertionError:
            _logger.logger.exception(f"Observational data must be a dict! Current type is {type(d)}")
            raise
        for i, (k,v) in enumerate(d.items()):
            try:
                assert not isinstance(v, dict)
            except AssertionError:
                _logger.logger.exception("Single-layer dictionary only for observation data!", exc_info=True)
                raise
            if isinstance(v, list):
                d[k] = np.array(v)
            try:
                assert not np.any(np.isnan(v))
            except AssertionError:
                _logger.logger.error("NaN values detected in observed variable {k}! This can lead to undefined behaviour.")
                raise
            # data consistency checks
            if i==0:
                self._obs_len = d[k].shape[-1]
            else:
                try:
                    assert d[k].shape[-1] == self.obs_len
                except AssertionError:
                    _logger.logger.exception(f"Data must be of the same length! Length is {self.obs_len}, but variable {k} has length {d[k].shape[-1]}", exc_info=True)
                    raise
    

    def load_observations_from_pickle(self, obs_file):
        """
        Load observed data from a pickle file

        Raises
        ------
        NotImplementedError
            for files other than .pickle
        """
        if obs_file.endswith(".pickle"):
            self.obs = load_data(obs_file)
        else:
            raise NotImplementedError("Only .pickle files currently supported!")
    
    
    def transform_obs(self, key, newkey, func):
        """
        Apply a transformation to an observed quantity, saving the result to the
        data frame.

        Parameters
        ----------
        key : str
            observation dictionary key
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
            _logger.logger.info(f"Applying transformation designated by {newkey}")
            self.obs[newkey] = func(self.obs[key])
            self._check_observation_validity(self.obs)
    

    def _sampler(self, data, prior=False, sample_kwargs={}):
        """
        Sample a stan model

        Parameters
        ----------
        data : dict
            stan data values
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
            return self._fit
        else:
            default_sample_kwargs = {"chains":4, "iter_sampling":2000, "show_progress":True, "show_console":False, "max_treedepth":12}
            if not prior:
                default_sample_kwargs["output_dir"] = os.path.join(data_dir, "stan_files")
            # update user given sample kwargs
            for k, v in sample_kwargs.items():
                default_sample_kwargs[k] = v
            if "output_dir" in default_sample_kwargs:
                os.makedirs(default_sample_kwargs["output_dir"], exist_ok=True)
            if prior:
                fit = self._prior_model.sample(data=data, **default_sample_kwargs)
            else:
                fit = self._model.sample(data=data, **default_sample_kwargs)
            _logger.logger.info(f"\n{fit.summary(sig_figs=4)}")
            _logger.logger.info(f"\n{fit.diagnose()}")
            return fit
    

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
            self._model = cmdstanpy.CmdStanModel(stan_file=self._model_file)
    

    def _random_obs_select(self, num, group):
        """
        Randomly select a subset of observed quantity data. Data that is not
        selected will be dropped from the data frame of observed quantities. 
        To ensure fair representation between all data series, <num>
        observations are taken for each member specified by <group>. E.g., if 
        there are 1000 observations, with each observation belonging to 1 of 10 
        groups, choosing num=20 will result in 20 selected points from group G, 
        giving a total of 20 selections/group * 10 groups = 200 observations.

        Parameters
        ----------
        num : int
            number of quantities per group to include in subset 
        group : str
            dataframe dictionary key that specifies the group
        """
        i_s = []
        counter = 0
        for kk in np.unique(self.obs[group]):
            _mask = self.obs[group] == kk
            ids = np.arange(self._obs_len)[_mask]
            _i_s = self._rng.integers(int(ids.min()), int(ids.max()), num)
            i_s.extend(_i_s)
            counter += 1
        idxs = np.r_[i_s]
        # delete unselected rows
        del_mask = np.full(self._obs_len, 1, dtype=bool)
        del_mask[idxs] = False
        for k, v in self.obs.items():
            self.obs[k] = v[del_mask]
    

    def sample_model(self, data, sample_kwargs={}):
        """
        Wrapper function around _sampler() to sample a stan likelihood model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
             kwargs to be passed to CmdStanModel.sample(), by default {}
        """
        self._stan_data = data
        if self._model is None and not self._loaded_from_file:
            self.build_model()
        self._fit = self._sampler(data=self._stan_data, sample_kwargs=sample_kwargs)
        self._fit_for_av = av.from_cmdstanpy(self._fit)
    

    def sample_prior(self, data, sample_kwargs={}):
        """
        Wrapper function around _sampler() to sample a stan prior model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
            kwargs to be passed to CmdStanModel.sample(), by default {}
        """
        self._prior_stan_data = data
        if self._prior_model is None and not self._loaded_from_file:
            self.build_model(prior=True)
        self._prior_fit = self._sampler(data=self._prior_stan_data, sample_kwargs=sample_kwargs, prior=True)
    

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
        return f"{fname_parts[0]}_{tag}{fname_parts[1]}"
    

    def parameter_plot(self, var_names, figsize=(9.0, 6.75), labeller=None):
        """
        Plot key pair plots and diagnostics of a stan likelihood model.

        Parameters
        ----------
        var_names : list
            variable names to plot
        figsize : tuple, optional
            size of figures, by default (9.0, 6.75)
        """
        assert len(var_names) > 1, "Pair plot requires at least two variables!"
        if len(var_names) > 4:
            _logger.logger.warning("Corner plots with more than 4 variables may not correctly map the labels given by the labeller!")
        # plot trace
        ax = av.plot_trace(self._fit_for_av, var_names=var_names, figsize=figsize)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"trace_{self._parameter_plot_counter}"), fig=fig)
        plt.close(fig)

        # plot rank
        ax = av.plot_rank(self._fit_for_av, var_names=var_names)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"rank_{self._parameter_plot_counter}"), fig=fig)
        plt.close(fig)

        # plot pair
        ax = av.plot_pair(self._fit_for_av, var_names=var_names, kind="scatter", marginals=True, scatter_kwargs={"marker":".", "markeredgecolor":"k", "markeredgewidth":0.5, "alpha":0.2}, figsize=figsize, labeller=labeller)
        av.plot_pair(self._fit_for_av, var_names=var_names, kind="kde", divergences=True, ax=ax, point_estimate="mode", marginals=True, kde_kwargs={"contour_kwargs":{"linewidths":0.5}}, point_estimate_marker_kwargs={"marker":""}, labeller=labeller)
        fig = ax.flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"pair_{self._parameter_plot_counter}"), fig=fig)
        plt.close(fig)
        self._parameter_plot_counter += 1
    

    def prior_plot(self, xobs, yobs, xmodel, ymodel, yobs_err=None, levels=[50, 90, 95, 99], ax=None):
        """
        Plot a prior predictive check for a regression stan model.

        Parameters
        ----------
        xobs : str
            dictionary key for observed independent variable 
        yobs : str
            dictionary key for observed dependent variable
        xmodel : str
            dictionary key for modelled independent variable
        ymodel : str
            dictionary key for modelled dependent variable
        yobs_err : str, optional
             dictionary key for observed dependent variable scatter, by default 
             None
        levels : list, optional
            HDI intervals to plot, by default [50, 90, 95, 99]
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            axis to plot to, by default None (creates new instance)
        """
        levels.sort(reverse=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
        ys = self._prior_fit.stan_variable(ymodel)
        cmapper, sm = create_normed_colours(max(0, 0.8*min(levels)), max(levels), cmap="Blues", normalisation="LogNorm")
        for l in levels:
            _logger.logger.info(f"Fitting level {l}")
            av.plot_hdi(self._prior_stan_data[xmodel], ys, hdi_prob=l/100, ax=ax, plot_kwargs={"c":cmapper(l)}, fill_kwargs={"color":cmapper(l), "alpha":0.8, "label":f"{l}% CI", "edgecolor":None}, smooth=False)
        # overlay data
        if self._num_groups < 2:
            self._plot_obs_data_kwargs["cmap"] = "Set1"
        if yobs_err is None:
            ax.scatter(self.obs[xobs], self.obs[yobs], c=self.categorical_label, **self._plot_obs_data_kwargs)
        else:
            colvals = np.unique(self.categorical_label)
            ncols = len(colvals)
            cmapper, sm = create_normed_colours(np.min(colvals), np.max(colvals), cmap=self._plot_obs_data_kwargs["cmap"])
            for i, c in enumerate(colvals):
                col = cmapper(c)
                mask = self.categorical_label==c
                ax.errorbar(self.obs[xobs][mask], self.obs[yobs][mask], yerr=self.obs[yobs_err][mask], c=col, zorder=20, fmt=".", capsize=5, label=("Obs" if i==ncols-1 else ""))
        ax.legend()
        savefig(self._make_fig_name(self.figname_base, f"prior_pred_{yobs}"), fig=fig)
    

    def posterior_plot(self, xobs, yobs, ymodel, yobs_err=None, levels=[50, 90, 95, 99], ax=None):
        """
        Plot a posterior predictive check for a regression stan model.

        Parameters
        ----------
        xobs : str
            dictionary key for observed independent variable
        yobs : str
            dictionary key for observed dependent variable
        ymodel : str
            dictionary key for modelled dependent variable
        yobs_err : str, optional
             dictionary key for observed dependent variable scatter, by default 
             None
        levels : list, optional
            HDI intervals to plot, by default [50, 90, 95, 99]
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            axis to plot to, by default None (creates new instance)
        """
        levels.sort(reverse=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()
        ys = self._fit.stan_variable(ymodel)
        cmapper, sm = create_normed_colours(max(0, 0.8*min(levels)), max(levels), cmap="Blues", normalisation="LogNorm")
        for l in levels:
            _logger.logger.info(f"Fitting level {l}")
            av.plot_hdi(self.obs[xobs][self._observation_mask].flatten(), ys[self._observation_mask], hdi_prob=l/100, ax=ax, plot_kwargs={"c":cmapper(l)}, fill_kwargs={"color":cmapper(l), "alpha":0.9, "label":f"{l}%"}, smooth=False)
        # overlay data
        if self._num_groups < 2:
            self._plot_obs_data_kwargs["cmap"] = "Set1"
        if yobs_err is None:
            ax.scatter(self.obs[xobs][self._observation_mask], self.obs[yobs][self._observation_mask], c=self.categorical_label, zorder=20, **self._plot_obs_data_kwargs)
        else:
            colvals = np.unique(self.categorical_label)
            cmapper, sm = create_normed_colours(np.min(colvals), np.max(colvals), cmap=self._plot_obs_data_kwargs["cmap"])
            ncols = len(colvals)
            for i, c in enumerate(colvals):
                col = cmapper(c)
                mask = np.logical_and(self.categorical_label==c, self._observation_mask)
                ax.errorbar(self.obs[xobs][mask], self.obs[yobs][mask], yerr=self.obs[yobs_err][mask], c=col, zorder=20, fmt=".", capsize=5, label=("Obs." if i==ncols-1 else ""))
        ax.legend()
        savefig(self._make_fig_name(self.figname_base, f"posterior_pred_{yobs}"), fig=fig)

    
    def sample_generated_quantity(self, gq, force_resample=False):
        """
        Sample the 'generated quantities' block of a Stan model

        Parameters
        ----------
        gq : str
            stan variable to sample
        force_resample : bool, optional
            run the generate_quantities method() again even if already run, by 
            default False

        Returns
        -------
        np.ndarray
            set of draws for the variable gq
        """
        if self.generated_quantities is None or force_resample:
            self.generated_quantities = self._model.generate_quantities(data=self._stan_data, mcmc_sample=self._fit)
        return self.generated_quantities.stan_variable(gq)


    def plot_generated_quantity_dist(self, gq, ax=None, xlabels=None, save=True, plot_kwargs={}):
        """
        Plot the 1-D distribution of an arbitrary variable in the generated quantities block of a stan model.

        Parameters
        ----------
        gq : list
            variables to plot
        ax : matplotlib.axes._subplots.AxesSubplot or np.ndarray of, optional
            axes object to plot to, by default None
        xlabels : list, optional
            labels for the x-axis, by default None
        save : bool, optional
            save the plot, by default True

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
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
            assert len(gq) == len(xlabels)
        for i, (_gq, l) in enumerate(zip(gq, xlabels)):
            ys = self.sample_generated_quantity(gq)
            try:
                assert len(ys.shape) < 3
            except AssertionError:
                _logger.logger.exception(f"Generated quantity {_gq} must have shape 2, has shape {len(ys.shape)}", exc_info=True)
                raise
            av.plot_dist(ys, ax=ax[i], plot_kwargs=plot_kwargs)
            ax[i].set_xlabel(l)
        ax.reshape(ax_shape)
        if save:
            savefig(self._make_fig_name(self.figname_base, "gqs"), fig=fig)
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
        df = self._fit.summary(sig_figs=4)
        vars.insert(0, "Variable")
        max_str_len = max([len(v) for v in vars]) + 1
        head_str = f"\n{vars[0]:>{max_str_len}}       5%      50%     95%"
        print(head_str)
        dashes = ["-" for _ in range(len(head_str))]
        print("".join(dashes))
        for v in vars[1:]:
            print(f"{v:>{max_str_len}}:  {df.loc[v,'5%']:>6}  {df.loc[v,'50%']:>6}  {df.loc[v,'95%']:>6}")
        print()


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
        C = cls(model_file=None, prior_file=None, figname_base=figname_base, rng=rng, random_select_obs=None)
        C._fit = cmdstanpy.from_csv(fit_files)
        C._fit_for_av = av.from_cmdstanpy(C._fit)
        C._loaded_from_file = True
        return C



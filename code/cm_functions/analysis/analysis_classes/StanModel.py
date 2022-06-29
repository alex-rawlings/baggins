import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmdstanpy
import arviz as av


from ...plotting import savefig, create_normed_colours
from ...env_config import figure_dir, data_dir

__all__ = ["StanModel"]


class StanModel:
    def __init__(self, model_file, prior_file, obs_file, figname_base, rng=None, random_select_obs=None, autoregress=False) -> None:
        """
        Class to set up, run, and plot key plots of a stan model.

        Parameters
        ----------
        model_file : str
            path to .stan file specifying the likelihood model
        prior_file : str
            path to .stan file specifying the prior model
        obs_file : str
            path to .pickle or .hdf5 file of observed quantities
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
        self._obs_file = obs_file
        self.figname_base = figname_base
        if rng is None:
            self._rng = np.random.default_rng()
        self.obs = None
        self._stan_data = None
        self._model = None
        self._fit = None
        self._fit_gqs = None
        self._prior_stan_data = None
        self._prior_model = None
        self._prior_fit = None
        self.load_obs()
        if random_select_obs is not None:
            assert all(k in random_select_obs.keys() for k in ["num", "group"])
            self.random_obs_select_dict = random_select_obs
            self._random_obs_select(**random_select_obs)
        self._autoregress = autoregress
        self._parameter_plot_counter = 0
    
    @property
    def model_file(self):
        return self._model_file

    @property
    def prior_file(self):
        return self._prior_file

    @property
    def obs_file(self):
        return self._obs_file
    
    @property
    def fit_gqs(self):
        return self._fit_gqs

    def load_obs(self):
        """
        Load observed data

        Raises
        ------
        NotImplementedError
            for files other than .pickle and .hdf5
        """
        if self._obs_file.endswith(".pickle"):
            self.obs = pd.read_pickle(self._obs_file)
        elif self._obs_file.endswith(".hdf5"):
            self.obs = pd.read_hdf(self._obs_file)
        else:
            raise NotImplementedError("Only .pickle and .hdf5 files supported!")
    

    def transform_obs(self, key, newkey, func):
        """
        Apply a transformation to an observed quantity, saving the result to the
        data frame.

        Parameters
        ----------
        key : str
            observation column name
        newkey : str
            column name for transformed quantity
        func : function
            transformation

        Raises
        ------
        ValueError
            when proposed column name is a reserved keyword
        """
        if newkey == "mask":
            raise ValueError("Key 'mask' is a reserved key!")
        self.obs[newkey] = func(self.obs.loc[:, key])
    

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
        default_sample_kwargs = {"chains":4, "iter_sampling":2000, "show_progress":True, "show_console":False, "adapt_delta":1-1e-1, "max_treedepth":12}
        # update user given sample kwargs
        for k, v in sample_kwargs.items():
            default_sample_kwargs[k] = v
        if prior:
            fit = self._prior_model.sample(data=data, **default_sample_kwargs)
        else:
            fit = self._model.sample(data=data, **default_sample_kwargs)
        print(fit.summary(sig_figs=4))
        print(fit.diagnose())
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
            dataframe column name that specifies the group
        """
        i_s = []
        counter = 0
        for kk in np.unique(self.obs.loc[:, group]):
            _mask = self.obs.loc[:, group] == kk
            ids = self.obs.loc[_mask, group].index
            _i_s = self._rng.integers(int(ids.min()), int(ids.max()), num)
            i_s.extend(_i_s)
            counter += 1
        idxs = np.r_[i_s]
        # delete unselected rows, and rename the row indices to default
        self.obs.drop(self.obs.index.difference(idxs), inplace=True)
        self.obs.reset_index(drop=True, inplace=True)
    

    def sample_model(self, data, sample_kwargs={}, save=False):
        """
        Wrapper function around _sampler() to sample a stan likelihood model.

        Parameters
        ----------
        data : dict
            stan data values
        sample_kwargs : dict, optional
             kwargs to be passed to CmdStanModel.sample(), by default {}
        save : bool, optional
            save stan sampling to .csv files, by default False
        """
        self._stan_data = data
        if self._model is None:
            self.build_model()
        self._fit = self._sampler(data=self._stan_data, sample_kwargs=sample_kwargs)
        if save:
            print("Saving")
            self._fit.save_csvfiles(os.path.join(data_dir, "stan_files"))
    

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
        if self._prior_model is None:
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
        return os.path.join(figure_dir, f"{fname_parts[0]}_{tag}{fname_parts[1]}")
    

    def parameter_plot(self, var_names, figsize=(9.0, 6.75)):
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
        # plot trace
        ax = av.plot_trace(self._fit, var_names=var_names, figsize=figsize)
        fig = np.concatenate(ax).flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, "trace"), fig=fig)
        plt.close(fig)

        # plot rank
        ax = av.plot_rank(self._fit, var_names=var_names)
        fig = np.concatenate(ax).flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, "rank"), fig=fig)
        plt.close(fig)

        # plot pair
        ax = av.plot_pair(self._fit, var_names=var_names, kind="scatter", divergences=True, marginals=True, scatter_kwargs={"marker":".", "markeredgecolor":"k", "markeredgewidth":0.5, "alpha":0.2}, figsize=figsize)
        av.plot_pair(self._fit, var_names=var_names, kind="kde", ax=ax, point_estimate="mode", marginals=True, kde_kwargs={"contour_kwargs":{"linewidths":0.5}}, point_estimate_marker_kwargs={"marker":""})
        fig = np.concatenate(ax).flatten()[0].get_figure()
        savefig(self._make_fig_name(self.figname_base, f"pair_{self._parameter_plot_counter}"), fig=fig)
        #plt.close(fig)
        self._parameter_plot_counter += 1
    

    def prior_plot(self, xobs, yobs, xmodel, ymodel, levels=[50, 90, 95, 99], ax=None):
        """
        Plot a prior predictive check for a regression stan model.

        Parameters
        ----------
        xobs : str
            column name for observed independent variable 
        yobs : str
            column name for observed dependent variable
        xmodel : str
            column name for modelled independent variable
        ymodel : str
            column name for modelled dependent variable
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
            print(f"Fitting level {l}")
            av.plot_hdi(self._prior_stan_data[xmodel], ys, hdi_prob=l/100, ax=ax, plot_kwargs={"c":cmapper(l)}, fill_kwargs={"color":cmapper(l), "alpha":0.8, "label":f"{l}%"})
        # overlay data
        data_kwargs = {"c":"tab:red", "marker":".", "label":"Obs.", "linewidth":0.5, "edgecolor":"k"}
        ax.scatter(self.obs.loc[:, xobs], self.obs.loc[:, yobs], **data_kwargs)
        ax.legend(loc="upper left")
        savefig(self._make_fig_name(self.figname_base, f"prior_pred_{yobs}"), fig=fig)
    

    def posterior_plot(self, xobs, yobs, ymodel, levels=[50, 90, 95, 99], ax=None):
        """
        Plot a posterior predictive check for a regression stan model.

        Parameters
        ----------
        xobs : str
            column name for observed independent variable
        yobs : str
            column name for observed dependent variable
        ymodel : str
             column name for modelled dependent variable
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
            print(f"Fitting level {l}")
            if self._autoregress:
                 av.plot_hdi(self.obs.loc[1:,xobs], ys, hdi_prob=l/100, ax=ax, plot_kwargs={"c":cmapper(l)}, fill_kwargs={"color":cmapper(l), "alpha":0.9, "label":f"{l}%"})
            else:
                av.plot_hdi(self.obs.loc[:,xobs], ys, hdi_prob=l/100, ax=ax, plot_kwargs={"c":cmapper(l)}, fill_kwargs={"color":cmapper(l), "alpha":0.9, "label":f"{l}%"})
        #ordered_idxs = np.argsort(self.obs.loc[:, xobs])
        #ax.plot(self.obs.loc[ordered_idxs, xobs], np.nanmean(ys, axis=0)[ordered_idxs], label="Mean", lw=0.7, c="tab:orange")
        # overlay data
        data_kwargs = {"c":"tab:red", "marker":".", "label":"Obs.", "linewidth":0.5, "edgecolor":"k"}
        if self._autoregress:
            ax.scatter(self.obs.loc[:-1, xobs], self.obs.loc[1:, yobs], zorder=100, **data_kwargs)
        else:
            ax.scatter(self.obs.loc[:, xobs], self.obs.loc[:, yobs], zorder=100, **data_kwargs)
        ax.legend(loc="upper left")
        savefig(self._make_fig_name(self.figname_base, f"posterior_pred_{yobs}"), fig=fig)


    @classmethod
    def load_fit(cls, fit_files, obs_file, figname_base, rng=None):
        """
        Restore a stan model from a previously-saved set of csv files

        Parameters
        ----------
        fit_files : str, path-like
            path to previously saved csv files
        obs_file : str
            path to .pickle or .hdf5 file of observed quantities
        figname_base : str
            path-like base name that all plots will share
        rng : np.random._generator.Generator, optional
            random number generator, by default None (creates a new instance)
        """
        # initiate a class instance
        C = cls(model_file=None, prior_file=None, obs_file=obs_file, figname_base=figname_base, rng=rng, random_select_obs=None)
        C._fit = cmdstanpy.from_csv(fit_files)



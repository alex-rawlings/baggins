# helper functions for scripts in this directory

__all__ = ["stan_model_selector"]

def stan_model_selector(args, model_class, model_file, prior_file, fig_base, L):
    """
    Select the desired Stan model to run

    Parameters
    ----------
    args : ArgumentParser
        command line arguments
    model_class : StanModel-like
        the particular StanModel derived class to use
    model_file : path-like
        .stan model file (for posterior sampling)
    prior_file : path-like
        .stan prior file (for prior sampling)
    fig_base : path-like
        base of figure names
    L : logger
        script logger

    Returns
    -------
    : StanModel-like
        the particular instance of model_class
    """
    if args.type == "loaded":
        # load a previous sample for improved performance: no need to resample 
        # the likelihood function
        try:
            assert args.model in args.dir
        except AssertionError:
            L.exception(f"Using model '{args.model}', but Stan files do not contain this keyword: you may have loaded the incorrect files for this model!", exc_info=True)
            raise
        return model_class.load_fit(model_file=model_file, fit_files=args.dir, figname_base=fig_base)
    else:
        # sample
        return model_class(model_file=model_file, prior_file=prior_file, figname_base=fig_base, num_OOS=args.NOOS)
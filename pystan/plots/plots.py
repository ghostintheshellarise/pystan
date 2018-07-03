import logging

logger = logging.getLogger('pystan')

def _import_arviz():
    """
    Utility function to trying to import ArviZ library. Raises exception if not found

    Returns
    -------
    handle
        handle for library
    """
    try:
        import arviz
    except ImportError:
        msg = "ArviZ library required"
        logger.critical(msg)
        raise
    return arviz


def _normalize_parameters(posterior, var_names, kwargs):
    """

    Utility function to test whether kwargs contains duplicate information
      - posterior == fit
      - var_names == pars

    Parameters
    ----------
    posterior : StanFit4Model
        Posterior samples
    var_names : list
    kwargs : dict

    Returns
    -------
    posterior : posterior object
    var_names : list
    kwargs : dict
    """
    if posterior is None or (posterior and 'posterior' in kwargs):
        raise TypeError("Use either `fit` or `posterior´ parameter")
    if var_names is None or (var_names and 'var_names' in kwargs):
        raise TypeError("Use either `pars` or `var_names` parameter") 
    if posterior is None:
        posterior = kwargs.pop('posterior')
    if var_names is None:
        var_names = kwargs.pop('var_names')
    return posterior, var_names, kwargs


def autocorrplot(fit=None, pars=None, **kwargs):
    """autocorrplot(posterior, var_names=None, max_lag=100, symmetric_plot=False, combined=False, figsize=None, textsize=None, skip_first=0)
   
   # ArviZ autocorrplot - `arviz.autocorrplot`
    
    Parameters
    ----------
    posterior : xarray, or object that can be converted (pystan or pymc3 draws)
        Posterior samples
    var_names : list of variable names, optional
        Variables to be plotted, if None all variable are plotted.
        Vector-value stochastics are handled automatically.
    max_lag : int, optional
        Maximum lag to calculate autocorrelation. Defaults to 100.
    symmetric_plot : boolean, optional
        Plot from either [0, +lag] or [-lag, lag]. Defaults to False, [-, +lag].
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inches.
        Note this is not used if ax is supplied.
    textsize: int
        Text size for labels, titles and lines. If None it will be autoscaled based on figsize.
    skip_first : int, optional
        Number of first samples not shown in plots (burn-in).
    
    Returns
    -------
    ax : matplotlib axes
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.autocorrplot(posterior, var_names, **kwargs)

def densityplot(fit=None, pars=None, **kwargs):
    """densityplot(data, data_labels=None, var_names=None, alpha=0.05, point_estimate='mean', colors='cycle', outline=True, hpd_markers='', shade=0., bw=4.5, figsize=None, textsize=None, skip_first=0)  

    # ArviZ densityplot - `arviz.densityplot`
    
    Generates KDE plots for continuous variables and histograms for discretes ones.
    Plots are truncated at their 100*(1-alpha)% credible intervals. Plots are grouped per variable
    and colors assigned to models.
    
    Parameters
    ----------
    data : xarray.Dataset, object that can be converted, or list of these
           Posterior samples
    data_labels : list[str]
        List with names for the samples in the list of datasets. Useful when
        plotting more than one trace.
    varnames: list
        List of variables to plot (defaults to None, which results in all
        variables plotted).
    alpha : float
        Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).
    point_estimate : str or None
        Plot point estimate per variable. Values should be 'mean', 'median' or None.
        Defaults to 'mean'.
    colors : list or string, optional
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically choose a color per model from matplolib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to `cycle`.
    outline : boolean
        Use a line to draw KDEs and histograms. Default to True
    hpd_markers : str
        A valid `matplotlib.markers` like 'v', used to indicate the limits of the hpd interval.
        Defaults to empty string (no marker).
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    figsize : tuple
        Figure size. If None, size is (6, number of variables * 2)
    textsize: int
        Text size for labels and legend. If None it will be autoscaled based on figsize.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    
    Returns
    -------
    ax : Matplotlib axes
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.densityplot(posterior, var_names, **kwargs)


def energyplot(fit, pars, **kwargs):
    """energyplot(trace, kind='kde', bfmi=True, figsize=None, legend=True, fill_alpha=(1, .75), fill_color=('C0', 'C5'), bw=4.5, skip_first=0, kwargs_shade=None, ax=None, **kwargs)
    
    # ArviZ energyplot - `arviz.energyplot`

    Plot energy transition distribution and marginal energy distribution in
    order to diagnose poor exploration by HMC algorithms.
    
    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    kind : str
        Type of plot to display (kde or histogram)
    bfmi : bool
        If True add to the plot the value of the estimated Bayesian fraction of missing information
    figsize : figure size tuple
        If None, size is (8 x 6)
    legend : bool
        Flag for plotting legend (defaults to True)
    fill_alpha : tuple of floats
        Alpha blending value for the shaded area under the curve, between 0
        (no shade) and 1 (opaque). Defaults to (1, .75)
    fill_color : tuple of valid matplotlib color
        Color for Marginal energy distribution and Energy transition distribution.
        Defaults to ('C0', 'C5')
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind='kde'`
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    kwargs_shade : dicts, optional
        Additional keywords passed to `fill_between` (to control the shade)
    ax : axes
        Matplotlib axes.
    
    Returns
    -------
    ax : matplotlib axes
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.energyplot(posterior, var_names, **kwargs)


def forestplot(fit, pars, **kwargs):
    """forestplot(trace, models=None, varnames=None, combined=False, alpha=0.05, quartiles=True, rhat=True, neff=True, main=None, xtitle=None, xlim=None, ylabels=None, colors='C0', chain_spacing=0.1, vline=None, vcolors=None, figsize=None, textsize=None, skip_first=0, plot_kwargs=None, gridspec=None)
    
    # ArviZ Forest plot - `arviz.forestplot`

    Forest plot
    
    Generates a forest plot of 100*(1-alpha)% credible intervals from a trace or list of traces.
    
    Parameters
    ----------
    trace : trace or list of traces
        Trace(s) from an MCMC sample
    models : list of strings (optional)
        List with names for the models in the list of traces. Useful when plotting more that one
        trace
    varnames: list, optional
        List of variables to plot (defaults to None, which results in all variables plotted)
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    alpha : float, optional
        Alpha value for (1-alpha)*100% credible intervals. Defaults to 0.05.
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the (1-alpha)*100% intervals.
        Defaults to True
    rhat : bool, optional
        Flag for plotting Gelman-Rubin statistics. Requires 2 or more chains. Defaults to True
    neff : bool, optional
        Flag for plotting the effective sample size. Requires 2 or more chains. Defaults to True
    main : string, optional
        Title for main plot. Passing False results in titles being suppressed. Defaults to None
    xtitle : string, optional
        Label for x-axis. Defaults to None, i.e. no label
    xlim : list or tuple, optional
        Range for x-axis. Defaults to None, i.e. matplotlib's best guess.
    ylabels : list or array, optional
        User-defined labels for each variable. If not provided, the node
        __name__ attributes are used
    colors : list or string, optional
        list with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle`, it will automatically chose a color per model from the
        matyplolibs cycle. If a single color is passed, eg 'k', 'C2', 'red' this color will be used
        for all models. Defauls to 'C0' (blueish in most matplotlib styles)
    chain_spacing : float, optional
        Plot spacing between chains. Defaults to 0.1
    vline : list, optional
        Location of vertical references lines. Defaults to None}
    vcolors : list or string, optional
        list with valid matplotlib colors, one color per value in vline. If None (Defaults)
        `vcolors` is the same as `colors`.
    figsize : tuple, optional
        Figure size. Defaults to None
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    plot_kwargs : dict, optional
        Optional arguments for plot elements. Currently accepts `fontsize`, `linewidth`, `marker`
        and `markersize`.
    gridspec : GridSpec
        Matplotlib GridSpec object. Defaults to None.
    
    Returns
    -------
    gridspec : matplotlib GridSpec
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.forestplot(posterior, var_names, **kwargs)


def jointplot(fit, pars, **kwargs):
    """jointplot(trace, varnames=None, figsize=None, textsize=None, kind='scatter', gridsize='auto', skip_first=0, joint_kwargs=None, marginal_kwargs=None)
    
    # ArviZ jointplot - `arviz.jointplot`

    Plot a scatter or hexbin of two variables with their respective marginals distributions.
    
    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, two variables are required.
    figsize : figure size tuple
        If None, size is (8, 8)
    textsize: int
        Text size for labels
    kind : str
        Type of plot to display (scatter of hexbin)
    hexbin : Boolean
        If True draws an hexbin plot
    gridsize : int or (int, int), optional.
        Only works when hexbin is True.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    skip_first : int
        Number of first samples not shown in plots (burn-in)
    joint_shade : dicts, optional
        Additional keywords modifying the join distribution (central subplot)
    marginal_shade : dicts, optional
        Additional keywords modifying the marginals distributions (top and right subplot)
        (to control the shade)
    
    Returns
    -------
    axjoin : matplotlib axes, join (central) distribution
    ax_hist_x : matplotlib axes, x (top) distribution
    ax_hist_y : matplotlib axes, y (right) distribution
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.jointplot(posterior, var_names, **kwargs)


def pairplot(fit, pars, **kwargs):
    """pairplot(trace, varnames=None, figsize=None, textsize=None, kind='scatter', gridsize='auto', colorbar=False, divergences=False, skip_first=0, gs=None, ax=None, kwargs_divergences=None, **kwargs)

    # ArviZ pairplot - `arviz.pairplot`

    Plot a scatter or hexbin matrix of the sampled parameters.
    
    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    kind : str
        Type of plot to display (kde or hexbin)
    gridsize : int or (int, int), optional
        Only works for kind=hexbin.
        The number of hexagons in the x-direction. The corresponding number of hexagons in the
        y-direction is chosen such that the hexagons are approximately regular.
        Alternatively, gridsize can be a tuple with two elements specifying the number of hexagons
        in the x-direction and the y-direction.
    colorbar : bool
        If True a colorbar will be included as part of the plot (Defaults to False).
        Only works when kind=hexbin
    divergences : Boolean
        If True divergences will be plotted in a diferent color
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    gs : Grid spec
        Matplotlib Grid spec.
    kwargs_divergences : dicts, optional
        Aditional keywords passed to ax.scatter for divergences
    ax: axes
        Matplotlib axes
    
    Returns
    -------
    ax : matplotlib axes
    gs : matplotlib gridspec
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.pairplot(posterior, var_names, **kwargs)


def parallelplot(fit, pars, **kwargs):
    """parallelplot(trace, varnames=None, figsize=None, textsize=None, legend=True, colornd='k', colord='C1', shadend=.025, skip_first=0, ax=None)

    # ArviZ parallel coordinates plot - `arviz.parallelplot`

    A parallel coordinates plot showing posterior points with and without divergences
    
    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted. Can be used to change the order
        of the plotted variables
    figsize : figure size tuple
        If None, size is (12 x 6)
    textsize: int
        Text size for labels. If None it will be autoscaled based on figsize.
    legend : bool
        Flag for plotting legend (defaults to True)
    colornd : valid matplotlib color
        color for non-divergent points. Defaults to 'k'
    colord : valid matplotlib color
        color for divergent points. Defaults to 'C1'
    shadend : float
        Alpha blending value for non-divergent points, between 0 (invisible) and 1 (opaque).
        Defaults to .025
    skip_first : int, optional
        Number of first samples not shown in plots (burn-in).
    ax : axes
        Matplotlib axes.
    Returns
    -------
    ax : matplotlib axes
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.parallelplot(posterior, var_names, **kwargs)


def posteriorplot(fit, pars, **kwargs):
    """posteriorplot(trace, varnames=None, figsize=None, textsize=None, alpha=0.05, round_to=1, point_estimate='mean', rope=None, ref_val=None, kind='kde', bw=4.5, bins=None, skip_first=0, ax=None, **kwargs)
    
    # ArviZ posteriorplot - `arviz.posteriorplot`

    Plot Posterior densities in the style of John K. Kruschke's book.
    
    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : tuple
        Figure size. If None, size is (12, num of variables * 2)
    textsize: int
        Text size of the point_estimates, axis ticks, and HPD. If None it will be autoscaled
        based on figsize.
    alpha : float, optional
        Alpha value for (1-alpha)*100% credible intervals. Defaults to 0.05.
    round_to : int
        Controls formatting for floating point numbers
    point_estimate: str
        Must be in ('mode', 'mean', 'median')
    rope: tuple of list of tuples
        Lower and upper values of the Region Of Practical Equivalence. If a list is provided, its
        length should match the number of variables.
    ref_val: float or list-like
        display the percentage below and above the values in ref_val. If a list is provided, its
        length should match the number of variables.
    kind: str
        Type of plot to display (kde or hist) For discrete variables this argument is ignored and
        a histogram is always used.
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kind == kde`.
    bins : integer or sequence or 'auto', optional
        Controls the number of bins, accepts the same keywords `matplotlib.hist()` does. Only works
        if `kind == hist`. If None (default) it will use `auto` for continuous variables and
        `range(xmin, xmax + 1)` for discrete variables.
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    ax : axes
        Matplotlib axes. Defaults to None.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function depending on the value of `kind`.
    Returns
    -------
    ax : matplotlib axes
    """
    arviz = _import_arviz()
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return arviz.posteriorplot(posterior, var_names, **kwargs)


def traceplot(fit, pars, **kwargs):
    """traceplot(trace, varnames=None, figsize=None, textsize=None, lines=None, combined=False, grid=True, shade=0.35, priors=None, prior_shade=1, prior_style='--', bw=4.5, skip_first=0, ax=None, altair=False)

    Use ArviZ's traceplot to display parameters.
    If ArviZ not installed, use a customized ArviZ traceplot (Arviz version X.X)
    

    # ArviZ traceplot -- `arviz.traceplot`

    Plot samples histograms and values.
    
    Parameters
    ----------
    trace : Pandas DataFrame or PyMC3 trace
        Posterior samples
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    textsize: int
        Text size for labels, titles and lines. If None it will be autoscaled based on figsize.
    lines : dict
        Dictionary of variable name / value  to be overplotted as vertical lines to the posteriors
        and horizontal lines on sample values e.g. mean of posteriors, true values of a simulation.
        If an array of values, line colors are matched to posterior colors. Otherwise, a default
        `C3` line.
    combined : bool
        Flag for combining multiple chains into a single chain. If False (default), chains will be
        plotted separately.
    grid : bool
        Flag for adding gridlines to histogram. Defaults to True.
    shade : float
        Alpha blending value for plot line. Defaults to 0.35.
    priors : iterable of scipy distributions
        Prior distribution(s) to be plotted alongside posterior. Defaults to None (no prior plots).
    prior_Shade : float
        Alpha blending value for prior plot. Defaults to 1.
    prior_style : str
        Line style for prior plot. Defaults to '--' (dashed line).
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).
    skip_first : int
        Number of first samples not shown in plots (burn-in).
    ax : axes
        Matplotlib axes. Accepts an array of axes, e.g.:
        
        >>> fig, axs = plt.subplots(3, 2) # 3 RVs
        >>> pymc3.traceplot(trace, ax=axs)
        
        Creates own axes by default.
    altair : bool
        Should returned plot be an altair chart.
    
    Returns
    -------
    ax : matplotlib axes
    """
    if not kwargs.get('altair', False):
        try:
            import matplotlib
        except ImportError:
            logger.critical("matplotlib required for plotting.")
            raise
    else:
        try:
            import altair
        except ImportError:
            logger.critical("altair required for plotting with `altair=True`")
            raise
    try:
        from arviz import traceplot
    except ImportError:
        msg = "arviz not found, using customized arviz.traceplot"
        logger.warning(msg)
        from pystan.plots.custom import traceplot
   
    posterior, var_names, kwargs = _normalize_parameters(fit, pars, kwargs)
    return traceplot(posterior, pars, **kwargs)




import xarray as xr
import numpy as np
import scipy
import scipy.stats as stats
from scipy.stats import theilslopes
import itertools
import geopandas
import regionmask
import pandas as pd
from statsmodels.tsa.stattools import acf

import dask
import dask.bag as db
import dask.array as darray

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import cmasher as cmr

# ==================================================================== Data preparation functions

def switch_lons(ds, lon_name='lon'):
    """ Switches lons from -180-180 to 0-360 or vice versa"""
    ds = ds.copy()
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        if np.any(ds.coords[lon_name] < 0): # if current coords are -180 to 180
            ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360
        else:
            ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
        return ds.sortby(ds[lon_name])

def remove_data(ds, nh_lim, sh_lim, time_max, lat_name='lat', time_name='time'):
    """ Set data between two latitudes to NaN before a chosen time"""
    return xr.where((ds[lat_name] < nh_lim) &
                    (ds[lat_name] > sh_lim) &
                    (ds[time_name] < pd.to_datetime([time_max]).values),
                    np.nan,
                    ds)

def get_quantile_thresholds(ds, quantile, dim, lat_name='lat', lon_name='lon', lat_chunk=1, lon_chunk=1):
    """ Get quantile of ds"""
    ds = ds.chunk({**{lat_name: 1, lon_name: 1},
                   **{d: -1 for d in dim}})
    return ds.quantile(quantile, dim)

def get_max_month(da, time_name='time'):
    """Get the month of maximum average of da"""
    da = da.resample({time_name: 'MS'}) \
           .mean()
    return da.groupby(time_name+'.month').mean() \
             .argmax('month') + 1

def get_year_end(da, shift=5):
    """ Returns array of end years."""
    da = (da + shift) % 12
    return da.where(da > 0, 12)

def accumulate_to_year_end(da, year_ends, mask=None, shift=5, accumulate=12, time_name='time'):
    """ Returns accumulated monthly values up to year end"""
    da = da.shift({time_name: shift}) \
           .rolling({time_name: accumulate}).sum()
    da = da.where(da[time_name].dt.month == year_ends) \
           .resample({time_name: '1YS'}).sum(skipna=True)
    if mask is None:
        return da
    else:
        return da.where(mask == 1)

def get_ar6_region_mask(da, lon_name='lon', lat_name='lat'):
    """ Return masks on input grid """
    ar6_regions = geopandas.read_file("/scratch1/ric368/data/ar6_regions/IPCC-WGI-reference-regions-v4_shapefile/IPCC-WGI-reference-regions-v4.shp")
    #ar6_regions_dict = {number: name for number, name in zip(list(ar6_regions.index), list(ar6_regions.Acronym))}
    ar6_regions_mask = regionmask.Regions_cls(name='AR6_regions', 
                                              numbers=list(ar6_regions.index), 
                                              names=list(ar6_regions.Name), 
                                              abbrevs=list(ar6_regions.Acronym), 
                                              outlines=list(ar6_regions.geometry))
    return ar6_regions_mask.mask(da, lon_name=lon_name, lat_name=lat_name)

def calculate_extreme_days_per_fire_year(da, year_end, compute_monthly=False):
    """
    Calculates number of extreme days per year, where a year ends at the month indicated by year_end.
    
    Parameters
    ----------
    da : xarray DataArray
        Array to compute on.
    year_end : xarray DataArray
        Array of values indicating which month is the end year
    compute_monthly : bool, optional
        Whether to compute after calculating monthly sums.
    """
    
    da_m = da.resample(time='1MS').sum()
    if compute_monthly:
        da_m = da_m.compute()
    da_12m = da_m.rolling(time=12).sum()
    return da_12m.where(da_12m.time.dt.month == year_end) \
                 .resample(time='1YS').sum(skipna=True)

def normalise(da):
    """Normalise values to lie between 0 and 1"""
    return (da - da.min()) / (da.max() - da.min())

# ============================================================= Plotting tools
def adjust_lightness(color, amount=0.5):
    """
        Adjusts lightness of the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> adjust_lightness('g', 0.3)
        >> adjust_lightness('#F034A3', 0.6)
        >> adjust_lightness((.3,.55,.1), 0.5)
        
        From: https://stackoverflow.com/a/49601444
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def get_magma_waterlily_cmap():
    """ Generate custom colormap"""
    raw_colors = np.concatenate((plt.get_cmap('cmr.waterlily')(np.linspace(0.07, 1, 256))[:116:9],
                                np.array([[1,1,1,1]]),
                                cm.get_cmap('magma_r')(np.linspace(0, 0.78, 128))[7::10]))
    f = scipy.interpolate.interp1d(np.linspace(0,1,len(raw_colors)), raw_colors.T)
    colors_ = f(np.linspace(0,1,256)).T
    return mpl.colors.ListedColormap(colors_, name='custom_cmap')

# ============================================================= Trend and change-point testing and tools

def fdr(p_values_da, alpha=0.1):
    """
        Calculates significance on a DataArray of gridded p-values (p_values_da) by controlling the false discovery rate.
        Returns a DataArray of ones (significant) and zeros (not significant).
    """
    
    p_1d = p_values_da.values.reshape(-1) # 1-D array of p-values
    p_1d = p_1d[~np.isnan(p_1d)] # Remove NaNs
    sorted_pvals = np.sort(p_1d) # sort p-values
    N = len(sorted_pvals) # sample size
    
    fdr_criteria = alpha * (np.arange(1, N+1) / N) # the diagonal line of criteria
    pvals_less_than_fdr_criteria = np.where(sorted_pvals < fdr_criteria)[0]
    
    if len(pvals_less_than_fdr_criteria) > 0: #if any p-values satisfy the FDR criteria
        largest_p_less_than_criteria = pvals_less_than_fdr_criteria[-1] # index of the largest p-value still under the fdr_criteria line.
        p_fdr = sorted_pvals[largest_p_less_than_criteria] # the p-value for controlling the FDR
    else:
        p_fdr = -1 # abritrary number < 0. Ensures no significant results.
    
    # massage data into binary indicators of FDR significance
    keep_signif = p_values_da.where(p_values_da <= p_fdr, -999)
    signif_da = keep_signif.where(keep_signif == -999, 1)
    
    return signif_da.where(signif_da == 1, 0)

def get_signif_locs(da, lat, lon):
    """
        Gets lat/lon locations of significant and insignificant grid boxes.
        da should be a binary DataArray, with zeros indicating insignificant results.
    """        
    y = lat[np.where(da.values > 0)[0]].values
    x = lon[np.where(da.values > 0)[1]].values
    return [(x[i], y[i]) for i in range(len(x))]

def estimate_L(da):
    """
        Estimates block length L for each grid box of da.
    """
    from statsmodels.tsa.stattools import acf
    
    def acf_lag1(x):
        if np.sum(~np.isnan(x)) == 0: # if all NaNs
            return np.nan
        else:
            x = x[~np.isnan(x)]
            return acf(x, nlags=1)[-1]
    
    n = len(da.time.values)
    
    # DataArray of lag1 ACF coefficients
    rho_da = xr.apply_ufunc(acf_lag1, da, input_core_dims=[['time']], output_core_dims=[[]], vectorize=True, dask='allowed')
    
    # DataArray of effective sample size
    n_eff_da = n * ((1 - rho_da) / (1 + rho_da))
    
    # Initialise guess for block length
    Ls_da = xr.full_like(rho_da, 1)
    for i in range(10): # iterate to get estimate of L
        L_da = (n - Ls_da + 1) ** ( (2/3) * (1 - n_eff_da / n) )
        Ls_da = L_da
    
    return np.ceil(L_da) # round up to get block length

def random_resample(*args, samples,
                    function=None, function_kwargs=None, bundle_args=True,
                    replace=True):
    """
        Randomly resample from provided xarray args and return the results of the subsampled dataset passed through \
        a provided function
                
        Parameters
        ----------
        *args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}. The first \
            object in args must contain all dimensions listed in samples, but subsequent objects need not.
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        replace : boolean, optional
            Whether the sample is with or without replacement
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
            
        Author: Dougie Squire https://github.com/dougiesquire/Squire_2021_fire_susceptibility
    """
    samples_spec = samples.copy() # copy because use pop below
    args_sub = [obj.copy() for obj in args]
    dim_block_1 = [d for d, s in samples_spec.items() if s[1] == 1]

    # Do all dimensions with block_size = 1 together
    samples_block_1 = { dim: samples_spec.pop(dim) for dim in dim_block_1 }
    random_samples = {dim: 
                      np.random.choice(
                          len(args_sub[0][dim]),
                          size=n,
                          replace=replace)
                      for dim, (n, _) in samples_block_1.items()}
    args_sub = [obj.isel(
        {dim: random_samples[dim] 
         for dim in (set(random_samples.keys()) & set(obj.dims))}) for obj in args_sub]

    # Do any remaining dimensions
    for dim, (n, block_size) in samples_spec.items():
        n_blocks = int(n / block_size)
        random_samples = [slice(x,x+block_size) 
                          for x in np.random.choice(
                              len(args_sub[0][dim])-block_size+1, 
                              size=n_blocks,
                              replace=replace)]
        args_sub = [xr.concat([obj.isel({dim: random_sample}) 
                               for random_sample in random_samples],
                              dim=dim) 
                       if dim in obj.dims else obj 
                       for obj in args_sub]

    if function:
        if bundle_args:
            if function_kwargs is not None:
                res = function(*args_sub, **function_kwargs)
            else:
                res = function(*args_sub)
        else:
            if function_kwargs is not None:
                res = tuple([function(obj, **function_kwargs) for obj in args_sub])
            else:
                res = tuple([function(obj) for obj in args_sub])
    else:
        res = tuple(args_sub,)

    if isinstance(res, tuple):
        if len(res) == 1:
            return res[0]
    else:
        return res
    
    
def n_random_resamples(*args, samples, n_repeats, 
                       function=None, function_kwargs=None, bundle_args=True, 
                       replace=True, with_dask=True):
    """
        Repeatedly randomly resample from provided xarray objects and return the results of the subsampled dataset passed \
        through a provided function
                
        Parameters
        ----------
        args : xarray DataArray or Dataset
            Objects containing data to be resampled. The coordinates of the first object are used for resampling and the \
            same resampling is applied to all objects
        samples : dictionary
            Dictionary containing the dimensions to subsample, the number of samples and the continuous block size \
            within the sample. Of the form {'dim1': (n_samples, block_size), 'dim2': (n_samples, block_size)}
        n_repeats : int
            Number of times to repeat the resampling process
        function : function object, optional
            Function to reduced the subsampled data
        function_kwargs : dictionary, optional
            Keyword arguments to provide to function
        replace : boolean, optional
            Whether the sample is with or without replacement
        bundle_args : boolean, optional
            If True, pass all resampled objects to function together, otherwise pass each object through function \
            separately
        with_dask : boolean, optional
            If True, use dask to parallelize across n_repeats using dask.delayed
                
        Returns
        -------
        sample : xarray DataArray or Dataset
            Array containing the results of passing the subsampled data through function
            
        Author: Dougie Squire https://github.com/dougiesquire/Squire_2021_fire_susceptibility
    """

    if with_dask & (n_repeats > 1000):
        n_args = itertools.repeat(args[0], times=n_repeats)
        b = db.from_sequence(n_args, npartitions=100)
        rs_list = b.map(random_resample, *(args[1:]), 
                        **{'samples':samples, 'function':function, 
                           'function_kwargs':function_kwargs, 'replace':replace}).compute()
    else:              
        resample_ = dask.delayed(random_resample) if with_dask else random_resample
        rs_list = [resample_(*args,
                             samples=samples,
                             function=function,
                             function_kwargs=function_kwargs,
                             bundle_args=bundle_args,
                             replace=replace) for _ in range(n_repeats)] 
        if with_dask:
            rs_list = dask.compute(rs_list)[0]
            
    if all(isinstance(r, tuple) for r in rs_list):
        return tuple([xr.concat([r.unify_chunks() for r in rs], dim='k') for rs in zip(*rs_list)])
    else:
        return xr.concat([r.unify_chunks() for r in rs_list], dim='k')
    
def pettitt_test(X):
    """
        Pettitt test calculated following Pettitt (1979): https://www.jstor.org/stable/2346729?seq=4#metadata_info_tab_contents
    """
    X = X[~np.isnan(X)]
    T = X.shape[-1]
    U = np.empty_like(X)
    for t in range(T): # t is used to split X into two subseries
        X_stack = np.zeros((*X.shape[:-1], t, X[...,t:].shape[-1] + 1), dtype=int)
        X_stack[...,:,0] = X[...,:t] # first column is each element of the first subseries
        X_stack[...,:,1:] = np.expand_dims(X[...,t:],-2) # all rows after the first element are the second subseries
        U[...,t] = np.sign(np.expand_dims(X_stack[...,:,0],-1) - X_stack[...,:,1:]).sum(axis=(-1,-2)) # sign test between each element of the first subseries and all elements of the second subseries, summed.

    tau = np.argmax(np.abs(U), axis=-1) # location of change (last data point of first sub-series, so new regime starts at tau+1)
    K = np.max(np.abs(U), axis=-1)
    p = 2 * np.exp(-6 * K**2 / (T**3 + T**2))
    
    return (tau, K, p)


def pettitt_nan_wrap(X):
    """
        Only run Pettitt test on data that isn't just NaNs.
    """
    if np.count_nonzero(~np.isnan(X)) > 0:
        return pettitt_test(X)
    else:
        return (np.nan, np.nan, np.nan)


def apply_pettitt(da):
    """
        Applies Pettitt test to every grid box in da.
        
        Parameters
        ----------
        da : xarray DataArray
            Array of values.
            
        Returns
        -------
        ds : xarray DataSet
            DataSet with the variables tau, K and p.
    """
    pettitt = xr.apply_ufunc(pettitt_nan_wrap, da, input_core_dims=[['time']], output_core_dims=[[], [], []], dask='allowed', vectorize=True) # compute pettitt test and output to tuple
    ds = pettitt[0].to_dataset(name='tau').merge(pettitt[1].to_dataset(name='K')).merge(pettitt[2].to_dataset(name='p')) # convert each element to a dataset and merge together
    
    return ds

def get_percentile(obs, bootstrap):
    """Returns the percentile of obs in bootstrap"""
    if np.isnan(obs):
        return np.nan
    else:
        return np.searchsorted(np.sort(bootstrap), obs) / len(bootstrap)


def apply_percentile(obs_da, bs_da, absolute=False):
    """
        Gets the percentiles of obs_da in bs_da.
        This function is slow and I'm sure could be improved.
    """
    obs_da = obs_da.sortby('lat', ascending=True) # need lats in ascending order: https://stackoverflow.com/a/53175095
    bs_da = bs_da.sortby('lat', ascending=True)
    
    if absolute: # for e.g. magnitude of change point, we want the percentile of the absolute change
        obs_da = np.abs(obs_da)
        bs_da = np.abs(bs_da)
    
    obs_stack = obs_da.stack(point=('lat', 'lon')).groupby('point') # stack lat and lon onto new dimension 'point', then group by point
    bs_stack = bs_da.stack(point=('lat', 'lon')).groupby('point')

    percentiles = xr.apply_ufunc(get_percentile, obs_stack, bs_stack, input_core_dims=[[], ['bs_sample']], output_core_dims=[[]], dask='allowed')
    ds = percentiles.unstack('point').sortby('lat', ascending=False) # unstack and sort lats to original ordering
    
    return ds

def cp_map_pettitt_ufunc(da, reps, autocorrelation_length, fy, ly):
    """
        Applies Pettitt test and generates relevant statistics.
        
        Parameters
        ----------
        da : xarray DataArray
            Array for which to compute Pettitt test
        reps : int
            Number of bootstrap samples per batch
        rep_batches : int
            Number of batches of bootstrap samples.
        autocorrelation_length : int
            Block length used in generating bootstrap samples
        fy : int
            First year to be used in generating yearly (Jul-Jun) event totals
        ly : int
            Last year to be used in generating event totals. Note this year corresponds to the end of the Jul-Jun year.

        Returns
        -------
        obs_cp_ds : xarray DataSet
            DataSet containing variables of tau (change point location), K (Pettitt test statistic),
            p (p-value of test statistic, according to the Pettitt formula), p_value (p_value from bootstrapping),
            cp_year (the year of the change point) and cp_magnitude (median of the subseries
            from tau minus that before tau).
    """
        
    # Pettitt tests
    obs_cp_ds = apply_pettitt(da)
    # For bootstrap samples
    bs_cp_ds = n_random_resamples(da, samples={'time': (len(da.time), autocorrelation_length)}, n_repeats=reps, function=apply_pettitt, with_dask=True)
    print('Finished bootstrap')
    bs_cp_ds = bs_cp_ds.rename({'k': 'bs_sample'}) # rename boostrap samples
    
    # p-values of observed test statistic in bootstrap samples
    test_stat_percentiles = apply_percentile(obs_cp_ds.K, bs_cp_ds.K, absolute=True)
    obs_cp_ds['p_value'] = 1 - test_stat_percentiles
    
    # rename p dimension to make it clear its the Pettitt approximation
    obs_cp_ds = obs_cp_ds.rename({'p': 'pettitt_p_value'})
    
    # Get year and magnitude of CP
    obs_cp_ds['cp_year'] = fy + obs_cp_ds['tau'] # last year of 'old regime'
    obs_cp_ds['cp_magnitude'] = da.where(da.time.dt.year > obs_cp_ds['cp_year']).quantile(0.5, 'time') - \
                                da.where(da.time.dt.year <= obs_cp_ds['cp_year']).quantile(0.5, 'time')
    
    return obs_cp_ds

def mk_theil_sen_1d(X):
    """Performs the Mann-Kendall trend test and calculates the Theil-Sen slope"""
    
    from scipy.stats import theilslopes
    
    X = X[~np.isnan(X)]
    n = X.shape[-1] # length of X (usually time)
    stack = np.full((n - 1, n), np.nan)
    
    np.fill_diagonal(stack[:,1:], X[0]) # fill the above-diagonal with first element of X
    for i in range(n-1): # fill diagonal and below-diagonals with remaining elements
        np.fill_diagonal(stack[i:,:], X[i+1])
    
    diff = np.expand_dims(stack[:,0], -1) - stack[:,1:] # difference between each element and all previous elements of X
    sign = np.sign(diff) # sign of these differences
    S = sign[~np.isnan(sign)].astype(int).sum() # sum of signed differences
    
    ts_slope, ts_intercept = theilslopes(X)[:2] # First element is estimate of slope, second is intercept
    
    return (S, ts_slope, ts_intercept)

def mk_nan_wrap(X):
    """
        Only run Mann-Kendall test on data that isn't just NaNs.
    """
    if np.count_nonzero(~np.isnan(X)) > 0:
        return mk_theil_sen_1d(X)
    else:
        return (np.nan, np.nan, np.nan)


def apply_mk(da):
    """
        Applies Mann-Kendall test to every grid box in da.
        Also calculates the Theil-Sen slope and intercept.
        
        Parameters
        ----------
        da : xarray DataArray
            Array of values.
            
        Returns
        -------
        ds : xarray DataSet
            DataSet with the variables S, ts_slope and ts_intercept.
    """
    mk = xr.apply_ufunc(mk_nan_wrap, da, input_core_dims=[['time']], output_core_dims=[[], [], []], dask='allowed', vectorize=True)
    ds = mk[0].to_dataset(name='S').merge(mk[1].to_dataset(name='ts_slope')).merge(mk[2].to_dataset(name='ts_intercept'))
    
    return ds

def cp_map_mk_ufunc(da, reps, autocorrelation_length, fy, ly):
    """
        Applies MK test and generates relevant statistics.
        
        Parameters
        ----------
        da : xarray DataArray
            Array for which to compite MK test
        reps : int
            Number of bootstrap samples per batch
        rep_batches : int
            Number of batches of bootstrap samples.
        autocorrelation_length : int
            Block length used in generating bootstrap samples
        fy : int
            First year to be used in generating yearly (Jul-Jun) event totals
        ly : int
            Last year to be used in generating event totals. Note this year corresponds to the end of the Jul-Jun year.

        Returns
        -------
        obs_cp_ds : xarray DataSet
            DataSet containing variables of S (MK test statistic), p_value (p-value of test statistic from bootstrapping S),
            ts_slope (the Theil-Sen slope estimate) and ts_intercept (the Theil-Sen intercept estimate).
    """
        
    # MK tests
    obs_ds = apply_mk(da)
    # For bootstrap samples
    bs_ds = n_random_resamples(da, samples={'time': (len(da.time), autocorrelation_length)}, n_repeats=reps, function=apply_mk, with_dask=True)
    bs_ds = bs_ds.rename({'k': 'bs_sample'}) # rename boostrap samples
    
    # p-values of observed test statistic in bootstrap samples
    test_stat_percentiles = apply_percentile(obs_ds.S, bs_ds.S, absolute=True)
    obs_ds['p_value'] = 1 - test_stat_percentiles
    
    return obs_ds


# ============================================ Climate diagnostics tools

def climatology_monthly(da, climatology_slice=None, time_dim='time'):
    """Returns the monthly climatology"""
    if climatology_slice is None:
        clim = da.groupby(time_dim+'.month').mean()
    else:
        clim = da.sel({time_dim: climatology_slice}).groupby(time_dim+'.month').mean()
    return clim

def anomalise_monthly(da, climatology_slice=None, standardise=False, time_dim='time'):
    """Returns the monthly anomalies of da"""
    clim = climatology_monthly(da, climatology_slice=climatology_slice, time_dim=time_dim)
    anoms = da.groupby(time_dim+'.month') - clim
    
    if standardise:
        return (anoms.groupby(time_dim+'.month') / anoms.groupby(time_dim+'.month').std()).drop('month')
    else:
        return anoms.drop('month')
    
def detrend_dim(da, dim, deg=1):
    """
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """    
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    """
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended
    
# ========================================= Climate modes
    
def calc_dmi(da):
    """ Calculates Dipole Mode Index"""
    boxW = [-10.0,10.0,50.0,70.0]
    boxE = [-10.0,0.0,90.0,110.0]
    
    da_W = da.sel(lat=slice(10, -10), lon=slice(50, 70)).mean(['lat', 'lon'])
    da_E = da.sel(lat=slice(0, -10), lon=slice(90, 110)).mean(['lat', 'lon'])
    
    return (da_W - da_E)

def calc_pna(h500_anom, time_name='time', lat_name='lat', lon_name='lon'):
    """
        Returns the Wallace and Gutzler Pacific north American mode index (see, for example, 
            http://research.jisao.washington.edu/data_sets/pna/)
        
        | Author: Dougie Squire
        | Date: 10/04/2018
        
        Parameters
        ----------
        h500_anom : xarray DataArray
            Array containing anomalies of 500hPa geopotential height
        lat_name : str, optional
            Name of the latitude dimension. If None, doppyo will attempt to determine lat_name \
                    automatically
        lon_name : str, optional
            Name of the longitude dimension. If None, doppyo will attempt to determine lon_name \
                    automatically
        time_name : str, optional
            Name of the time dimension. If None, doppyo will attempt to determine time_name \
                    automatically
            
        Returns
        -------
        pna : xarray DataArray
            Array containing the Pacific north American mode index
    """
    
    lat_p1 = 20
    lon_p1 = -160+360
    lat_p2 = 45
    lon_p2 = -165+360
    lat_p3 = 55
    lon_p3 = -115+360
    lat_p4 = 30
    lon_p4 = -85+360

    h500_anom_p1 = h500_anom.interp({lat_name : lat_p1, lon_name : lon_p1})
    h500_anom_p2 = h500_anom.interp({lat_name : lat_p2, lon_name : lon_p2})
    h500_anom_p3 = h500_anom.interp({lat_name : lat_p3, lon_name : lon_p3})
    h500_anom_p4 = h500_anom.interp({lat_name : lat_p4, lon_name : lon_p4})

    h500_anom_p1_group = h500_anom_p1.groupby(time_name+'.month')
    h500_anom_p2_group = h500_anom_p2.groupby(time_name+'.month')
    h500_anom_p3_group = h500_anom_p3.groupby(time_name+'.month')
    h500_anom_p4_group = h500_anom_p4.groupby(time_name+'.month')
    
    return  0.25 * ((h500_anom_p1_group / h500_anom_p1_group.std(time_name)).drop('month') - \
                        (h500_anom_p2_group / h500_anom_p2_group.std(time_name)).drop('month') + \
                        (h500_anom_p3_group / h500_anom_p3_group.std(time_name)).drop('month') - \
                        (h500_anom_p4_group / h500_anom_p4_group.std(time_name)).drop('month'))


def calc_sam(slp, clim_period, lat_name='lat', lon_name='lon', time_name='time'):
    """
        Author: Dougie Squire
        
        Returns southern annular mode index as defined by Gong, D. and Wang, S., 1999. Definition \
            of Antarctic oscillation index. Geophysical research letters, 26(4), pp.459-462.
        
        Parameters
        ----------
        slp : xarray DataArray
            Array containing monthly sea level pressure
        clim_period : size (1,2) array-like containing start and end dates of period used to \
            anomalize and normalise
        lat_name : str, optional
            Name of the latitude dimension.
        lon_name : str, optional
            Name of the longitude dimension.
        time_name : str, optional
            Name of the time dimension.
            
        Returns
        -------
        sam : xarray DataArray
            Array containing the Gong and Wang (1999) southern annular mode index
    """
    def _normalise(group, clim_group):
        """ Return the anomalies normalize by their standard deviation """
        month = group[time_name].dt.month.values[0]
        months, _ = zip(*list(clim_group))
        clim_group_month = list(clim_group)[months.index(month)][1]
        return (group - clim_group_month.mean(time_name)) / clim_group_month.std(time_name)

    slp_40 = slp.interp({lat_name: -40}).mean(lon_name)
    slp_65 = slp.interp({lat_name: -65}).mean(lon_name)

    slp_40_group = slp_40.groupby(time_name+'.month')
    slp_40_group_clim = slp_40.sel(
        {time_name: slice(clim_period[0],
                          clim_period[-1])}).groupby(time_name+'.month')
    slp_65_group = slp_65.groupby(time_name+'.month')
    slp_65_group_clim = slp_65.sel(
        {time_name: slice(clim_period[0],
                          clim_period[-1])}).groupby(time_name+'.month')

    norm_40 = slp_40_group.map(_normalise, clim_group=slp_40_group_clim)
    norm_65 = slp_65_group.map(_normalise, clim_group=slp_65_group_clim)
    
    return (norm_40 - norm_65).rename('SAM')
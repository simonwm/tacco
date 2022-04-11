import numpy as np
import pandas as pd
import anndata as ad
import gc

from .. import get
from .. import preprocessing
from .. import utils
from . import _helper as helper
from ._annotate import annotate
from scipy.sparse import issparse, csr_matrix
from numba import njit, prange

@njit(parallel=True,cache=True)
def _get_minimal_transitions(vis, vos):
    transitions = np.zeros((vis.shape[0],vis.shape[1],vos.shape[1]))
    for i in prange(vis.shape[0]):
        vi = vis[i]
        vo = vos[i]
        delta = vo - vi
        delta_p = delta * (delta > 0)
        delta_m = delta * (delta < 0)
        sum_p = delta_p.sum()
        sum_m = delta_m.sum()
        if sum_m != 0 and sum_p != 0:
            delta_m *= sum_p/sum_m
            transitions[i] += np.outer(delta_m, delta_p * (1 / sum_p))
        transitions[i] += np.diag(vi-delta_m)
    return transitions

def get_minimal_transitions(
    aa,
    bb,
):
    assert(aa.shape==bb.shape)
    if issparse(aa):
        aa = aa.A
    if issparse(bb):
        bb = bb.A
    
    res = _get_minimal_transitions(aa,bb)
    
    return res

def _annotate_OT(
    adata,
    reference,
    annotation_key=None,
    annotation_prior=None,
    epsilon=5e-3,
    lamb=0.1,
    decomposition=False,
    deconvolution=False,
    **kw_args,
    ):

    """\
    Implements the functionality of :func:`~annotate_OT` without data
    integrity checks.
    """

    cell_prior = helper.prep_cell_priors(adata, reads=True)
    
    type_cell_dist = helper.prep_distance(adata, reference, annotation_key, decomposition=decomposition, deconvolution=deconvolution, **kw_args)
    if decomposition:
        type_cell_dist, mixtures = type_cell_dist
    types = type_cell_dist.index
    
    if decomposition: # include the annotation profiles themselves to obtain a measurement of confusion
        test_weight = 1e-6*cell_prior.sum()/(len(type_cell_dist.columns)-len(cell_prior.index)) # keep a low weight to not influence the actual typing (much)
        cell_prior = cell_prior.reindex(index=type_cell_dist.columns,fill_value=test_weight)

    cell_type = utils.run_OT(type_cell_dist, annotation_prior, cell_prior=cell_prior, epsilon=epsilon, lamb=lamb)
    
    if decomposition:
        cell_type /= cell_type.sum(axis=1).to_numpy()[:,None]
        # need to run measurement on mixtures instead of pure profiles to get the confusion as pure profiles should never be confused at all.
        # miXture Measurements
        xm = cell_type.loc[~cell_type.index.isin(adata.obs.index)].to_numpy()
        
        # Observation Measurements
        om = cell_type.loc[adata.obs.index].to_numpy()
        # miXture Annotation joint probability distribution
        xa = mixtures.copy()
        # get probability of miXture given Annotation
        utils.col_scale(xa,1/xa.sum(axis=0).A.flatten())
        # get measurement given annotation

        # assume independent m and a
        # ma = utils.gemmT(xm.T, xa.T)

        # get probability of Annotation given mixture
        ax = mixtures.T
        utils.col_scale(ax,1/ax.sum(axis=0).A.flatten())
        # assume minimal error between m and a
        xam = get_minimal_transitions(ax.T, xm)
        ax.eliminate_zeros()
        ax.data = 1 / ax.data
        ma = np.einsum('xam,xa,ax->ma', xam, xa.A, ax.A)
        
        cell_type = utils.parallel_nnls(ma, om)
        
        cell_type = pd.DataFrame(cell_type, columns=types, index=adata.obs.index)
    
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_OT(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    annotation_prior=None,
    epsilon=5e-3,
    lamb=0.1,
    decomposition=False,
    deconvolution=False,
    **kw_args,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by
    semi-balanced optimal transport.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tacco.preprocessing.create_reference` for options to create it.
    annotation_key
        The `.obs`, `.obsm`, and/or `.varm` key where the annotation and
        profiles are stored in the `reference`. If `None`, it is inferred from
        `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    annotation_prior
        A :class:`~pandas.Series` containing the annotation prior distribution.
        This argument is required and will throw a :class:`~ValueError` if it
        is unset.
    epsilon
        The prefactor for entropy regularization in Optimal Transport. Small
        values like `5e-3` tend to give only a few or a single annotation
        category per observation, while large values like `5e-1` give many
        annotation categories per observation.
    lamb
        The prefactor for prior constraint relaxation by KL divergence in
        unbalanced Optimal Transport. Smaller values like `1e-2` relax the
        constraint more than larger ones like `1e0`. If `None`, do not relax
        the prior constraint and fix the annotation fraction at
        `annotation_prior`.
    decomposition
        Whether to decompose the annotation using information from injected
        in-silico type mixtures.
    deconvolution
        Which method to use for deconvolution of the cost based on similarity of
        different annotation profiles. If `False`, no deconvolution is done.
        Available methods are:
        
        - 'nnls': solves nnls to get only non-negative deconvolved projections
        - 'linear': solves a linear system to disentangle contributions; can
          result in negative values which makes sense for general vectors and
          amplitudes, i.e.
          
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tacco.tools._helper.prep_distance`.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """

    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location)

    # call typing without data integrity checks
    cell_type = _annotate_OT(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        annotation_prior=annotation_prior,
        epsilon=epsilon,
        lamb=lamb,
        decomposition=decomposition,
        deconvolution=deconvolution,
        **kw_args,
    )
    
    return cell_type

def _select_best_prior_epsilon(epsilons, type_priors, ax=None):
    log_epsilons = np.log(epsilons)
    changes = np.exp(np.abs(np.gradient(np.log(type_priors),log_epsilons,axis=0))) # relative absolute changes of type prior with log(epsilon)
    # define "good" epsilons as these which have a low std of the changes over the types, wrt. some stable measure of the mean the mean: the log(epsilon) integrated mean up to the previous epsilon
    stds = changes.std(axis=1)
    means = changes.mean(axis=1)
    eps_mean_avg = utils.math.integrate_mean(means, log_epsilons)
    # take the smallest epsilon which is good, but not separated by bad epsilons from the biggest epsilon; take the first epsilon, if all are bad...
    best_epsilon = np.concatenate([[epsilons[0]],epsilons[1:][np.cumprod(stds[1:] < eps_mean_avg[:-1]).astype(bool)]])[-1]
    
    #print(np.array(type_priors).sum(axis=0))
    #print(np.array(type_priors).sum(axis=1))
    kl = np.array([np.sum(type_priors[i] * np.log(type_priors[i]/type_priors[0])) for i in range(len(epsilons))]) / np.log(2)
    kls = np.array([np.sum(type_priors[i] * np.log(type_priors[i]/type_priors[max(0,i-1)])) for i in range(len(epsilons))]) / np.log(2)
    ent = np.array([-np.sum(type_priors[i] * np.log(type_priors[i])) for i in range(len(epsilons))]) / np.log(2)
    
    #if ax is None: #this forces the plot also as part of the actual run which is a bit disruptive
    #    import matplotlib.pyplot as plt
    #    fig, ax = plt.subplots()
    if ax is not None:
        ax.plot(epsilons,changes)
        ax.plot(epsilons,means,c='green')
        ax.plot(epsilons,stds,c='black')
        ax.plot(epsilons,eps_mean_avg,c='blue')
        ax.plot(epsilons,kl,c='black',linestyle='dashed')
        ax.plot(epsilons,kls,c='blue',linestyle='dashed')
        ax.plot(epsilons,ent,c='green',linestyle='dashed')
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    #print(pd.DataFrame([(epsilons[i], eps_mean_avg[i], stds[i], kl[i], kls[i], ent[i]) for i in range(len(epsilons))], columns=['epsilon','mean_avg','std','kl','kls','ent']).set_index('epsilon'))
    print('best prior epsilon: %s' % best_epsilon)
    return best_epsilon

def _prep_priors_reservoir(tdata, reference, type_key, epsilon='auto', gamma=1e6, out_epsilon=None, **prep_distance_kw):
    out_epsilon = [] if out_epsilon is None else out_epsilon
    # join buffer and test data for evaluation
    joining_batch = 'joining_batch'
    # csr-ing sparse data makes the concatenate much faster
    if issparse(tdata.X) and issparse(reference.X):
        def _csrify(adata,type_key):
            if isinstance(adata.X, csr_matrix):
                return adata
            else:
                var = adata.var[[]]
                obs = adata.obs[[type_key]] if type_key in adata.obs else adata.obs[[]]
                obsm = {type_key:adata.obsm[type_key]} if type_key in adata.obsm else None
                varm = {type_key:adata.varm[type_key]} if type_key in adata.varm else None
                return ad.AnnData(X=adata.X.tocsr(), obs=obs, obsm=obsm, varm=varm, var=var)
        tdata = _csrify(tdata,type_key)
        reference = _csrify(reference,type_key)
    joined = reference.concatenate(tdata, batch_key=joining_batch)
    
    del tdata # clean up the copies.
    gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    type_cell_dist = helper.prep_distance(joined, reference, type_key, **prep_distance_kw)

    # find filtered indices corresponding to either part of the merged data
    ot_ref_index = joined.obs.index[joined.obs[joining_batch] == '0']
    ot_test_index = joined.obs.index[joined.obs[joining_batch] == '1']

    # get the buffer cell type fractions to calibrate the cell type prior with
    ref_anno_prior, joined_cell_prior = helper.prep_priors_reference(joined, reference, type_key, reads=True)
    
    del reference, joined # clean up the copies.
    gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    ref_anno_prior /= ref_anno_prior.sum()
    # increase weights of buffer cells to simulate a much larger reservoir of buffer cells, meaning that the small fraction of test cells can have their share of type weight
    joined_cell_prior.loc[ot_ref_index] *= gamma
    joined_cell_prior /= joined_cell_prior.sum()

    #print(type_cell_dist, buf, joined_cell_prior, epsilon)
    
    if isinstance(epsilon, str) and epsilon == 'auto':
        print('optimizing prior epsilon... [if that takes too long, consider specifying a fixed "epsilon" in the prep priors key word arguments]')
        epsilons = np.array([0.5,0.2,0.1,0.05,0.02,0.01,0.005])
    else:
        epsilons = np.array(epsilon)
    
    # support lists of epsilons
    if len(epsilons.shape) == 0:
        epsilons = np.array([epsilon])
    annotation_prior = []
    
    # solve common OT of test and reference cells
    for _epsilon in epsilons:
        cell_type = utils.run_OT(type_cell_dist, type_prior=ref_anno_prior, cell_prior=joined_cell_prior, epsilon=_epsilon)

        annotation_prior.append(cell_type.loc[ot_test_index].sum())
        annotation_prior[-1] /= annotation_prior[-1].sum()
    #print('referrers:',len(gc.get_referrers(type_cell_dist)))
    del type_cell_dist # clean up
    gc.collect()
    
    if isinstance(epsilon, str) and epsilon == 'auto':
        epsilon = _select_best_prior_epsilon(epsilons, annotation_prior)
        annotation_prior = [annotation_prior[np.argwhere(epsilons == epsilon).flatten()[0]]]
        out_epsilon.append(epsilon)
    
    if len(np.array(epsilon).shape) == 0:
        annotation_prior = annotation_prior[0]

    return annotation_prior

def priors_OT(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    epsilon='auto',
    gamma=1e6,
    **kw_args,
    ):

    """\
    Determines annotation prior distribution using the reference as an
    annotation reservoir and solving OT on the joint system.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tacco.preprocessing.create_reference` for options to create it.
    annotation_key
        The `.obs`, `.obsm`, and/or `.varm` key where the annotation and
        profiles are stored in the `reference`. If `None`, it is inferred from
        `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    epsilon
        The prefactor for entropy regularization in Optimal Transport. Small
        values like `5e-2` give priors closer to the real distribution, while
        large values like `2e-1` give distributions closer to the reference
        annotation. (Very) small values can lead to instable prior
        determinations if `reference` and `adata` have (very) different
        properties, e.g. coming from (very) different experimental methods.
    gamma
        The upweighting factor for the `reference` data in the joint `adata`-
        `reference` OT system to solve. Should make weighted `reference`
        dataset much larger than `adata` in terms of cells/beads/counts.
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tacco.tools._helper.prep_distance`.
        
    Returns
    -------
    Returns the annotation priors in a :class:`~pandas.Series`.
    
    """
    
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location)

    annotation_prior = _prep_priors_reservoir(adata, reference, annotation_key, epsilon=epsilon, gamma=gamma, **kw_args)
    
    return annotation_prior

def _prep_priors_unbalanced(tdata, reference, annotation_key, epsilon=0.1, lamb=0.1, annotation_prior=None, **prep_distance_kw):

    type_cell_dist = helper.prep_distance(tdata, reference, annotation_key, **prep_distance_kw)

    ref_anno_prior, cell_prior = helper.prep_priors_reference(tdata, reference, annotation_key, reads=True)
    if annotation_prior is None:
        annotation_prior = ref_anno_prior
    
    annotation_prior /= annotation_prior.sum()
    
    cell_type = utils.run_OT(type_cell_dist, type_prior=annotation_prior, cell_prior=cell_prior, epsilon=epsilon, lamb=lamb)

    annotation_prior = cell_type.sum()
    annotation_prior /= annotation_prior.sum()
    
    return annotation_prior

def priors_uOT(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    epsilon=0.1,
    lamb=0.1,
    annotation_prior=None,
    **kw_args,
    ):

    """\
    Determines annotation prior distribution using unbalanced OT with some
    input annotation prior as anchor.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tacco.preprocessing.create_reference` for options to create it.
    annotation_key
        The `.obs`, `.obsm`, and/or `.varm` key where the annotation and
        profiles are stored in the `reference`. If `None`, it is inferred from
        `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    epsilon
        The prefactor for entropy regularization in Optimal Transport.
    lamb
        The prefactor for prior constraint relaxation by KL divergence in
        unbalanced Optimal Transport.
    annotation_prior
        The annotation prior distribution. If `None`, use the one in the
        `reference` data.
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`~tacco.tools._helper.prep_distance`.
        
    Returns
    -------
    Returns the annotation priors in a :class:`~pandas.Series`.
    
    """
    
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location)

    annotation_prior = _prep_priors_unbalanced(adata, reference, annotation_key, epsilon=epsilon, lamb=lamb, annotation_prior=annotation_prior, **kw_args)
    
    return annotation_prior

def drive_OT(
    adata,
    reference,
    annotation_key=None,
    epsilon=5e-3,
    lamb=0.1,
    decomposition=False,
    deconvolution=False,
    prior_epsilon=None, # 'auto'
    prior_lambda=None,
    prior_gamma=1e6,
    prep_distance_kw_args=None,
    **kw_args,
    ):
    
    """\
    Runs :func:`~tc.tl.annotate` specialized for the method 'OT'.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tc.pp.create_reference` for options to create it.
    epsilon
        The prefactor for entropy regularization in Optimal Transport. Small
        values like `5e-3` tend to give only a few or a single annotation
        category per observation, while large values like `5e-1` give many
        annotation categories per observation.
    lamb
        If not `None`, the types are calculated with unbalanced instead of
        balanced OT and the value gives the the prefactor of the KL divergence
        term for the marginal relaxation.
    decomposition
        Whether to decompose the typing using a measured confusion matrix.
    deconvolution
        Which method to use for deconvolution of the cost based on similarity
        of different annotation profiles. If `False`, no deconvolution is done.
        Available methods are:
        
        - 'nnls': solves nnls to get only non-negative deconvolved projections
        - 'linear': solves a linear system to disentangle contributions; can
          result in negative values which makes sense for general vectors and
          amplitudes, i.e.
        
    prior_epsilon
        The prefactor for entropy regularization in Optimal Transport. Small
        values like `5e-2` give priors closer to the real distribution, while
        large values like `2e-1` give distributions closer to the reference
        annotation. (Very) small values can lead to instable prior
        determinations if `reference` and `adata` have (very) different
        properties, e.g. coming from (very) different experimental methods. If
        `None`, use the reference type fractions.
    prior_lambda
        If not `None` (and `prior_epsilon` not `None`), the priors are
        calculated with unbalanced OT instead of the reservoir method. The
        value gives the prefactor of the KL divergence term for the marginal
        relaxation.
    prior_gamma
        The upweighting factor for the `reference` data in the joint `adata`-
        `reference` OT system to solve. Should make weighted `reference`
        dataset much larger than `adata` in terms of cells/beads/counts.
    prep_distance_kw_args
        Dictionary of keyword arguments forwarded to
        :func:`~tacco.tools._helper.prep_distance`.
    **kw_args
        Additional keyword arguments are forwarded to :func:`~tc.tl.annotate`.
        
    Returns
    -------
    Depending on `result_key`, either returns the original `adata` with\
    annotation written in `.obsm` keys, or as a new :class:`~pandas.DataFrame`.\
    If `return_reference`, then the platform normalized `reference` is returned\
    as a second object.
    
    """
    prep_distance_kw_args = {} if prep_distance_kw_args is None else prep_distance_kw_args
    def priors(adata, reference, annotation_key):
        nonlocal prior_epsilon, prior_lambda, prior_gamma, decomposition, deconvolution, prep_distance_kw_args
        if prior_epsilon is None:
            annotation_priors, _ = helper.prep_priors_reference(adata, reference, annotation_key, reads=True)
        elif prior_lambda is None:
            annotation_priors = priors_OT(adata, reference, annotation_key=annotation_key, epsilon=prior_epsilon, gamma=prior_gamma, decomposition=decomposition, deconvolution=deconvolution, **prep_distance_kw_args)
        else:
            annotation_priors = priors_uOT(adata, reference, annotation_key=annotation_key, epsilon=prior_epsilon, lamb=prior_lambda, decomposition=decomposition, deconvolution=deconvolution, **prep_distance_kw_args)
        return annotation_priors
    
    return annotate(
        adata,
        reference,
        annotation_key=annotation_key,
        method='OT',
        annotation_prior=priors(adata, reference, annotation_key),
        **kw_args,
        epsilon=epsilon,
        lamb=lamb,
        decomposition=decomposition,
        deconvolution=deconvolution,
        # prior_epsilon=prior_epsilon,
        # prior_lambda=prior_lambda,
        # prior_gamma=prior_gamma,
        **prep_distance_kw_args,
    )

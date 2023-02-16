import warnings
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.optimize import nnls
from difflib import SequenceMatcher
import scanpy as sc
import time
import gc
from .. import get
from .. import utils
from ..utils._utils import _transfer_pca, _infer_annotation_key
from .. import preprocessing

def normalize_result_format(type_fractions, types=None):
    if types is not None and isinstance(type_fractions.columns[0], str): # Rename columns and types using some "best" match. Necessary due to some weird R conventions...
        res_map = pd.Series(type_fractions.columns,index=type_fractions.columns).map(lambda x: x.replace('.',' '))
        types_map = pd.Series(types,index=types).map(lambda x: f'X{x}' if isinstance(x,int) else str(x).replace('.',' '))
        for r_i in res_map.index:
            r = res_map[r_i]
            best_match = None
            best_score = None
            for t_i in types_map.index:
                t = types_map[t_i]
                if best_match is None:
                    best_match = t_i
                    best_score = SequenceMatcher(None, r, t).ratio()
                else:
                    new_score = SequenceMatcher(None, r, t).ratio()
                    if new_score > best_score:
                        best_score = new_score
                        best_match = t_i
            res_map[r_i] = best_match
        type_fractions.rename(columns=res_map,inplace=True)
    
    if types is not None:
        type_fractions = type_fractions.reindex(columns=types).fillna(0)
    
    type_fractions = type_fractions.astype(float)
    type_fractions *= (type_fractions > 0) # sometimes there appear negative values (at least at some point in RCTD...)
    type_fractions /= type_fractions.sum(axis=1).to_numpy()[:,None] # normalize to 1 for all beads
    
    return type_fractions

def log_normalize(adata, target_sum=1e4, scale=False, type_key=None):
    sc.pp.normalize_total(adata, target_sum=target_sum)
    utils.log1p(adata)
    if scale:
        sc.pp.scale(adata)
    if type_key is not None and type_key in adata.varm:
        adata.varm[type_key] *= (target_sum / adata.varm[type_key].sum(axis=0).to_numpy())
        adata.varm[type_key] = np.log1p(adata.varm[type_key])
    return adata
    
def prep_cell_priors(tdata, reads=False):
    if reads:
        cell_prior = pd.Series(np.array(tdata.X.sum(axis=1)).flatten(), index=tdata.obs.index)
    else:
        cell_prior = None
    return cell_prior

def prep_distance(tdata, reference, type_key, n_pca=None, zero_center=False, log_norm=False, scale=False, metric='h2', min_distance=False, decomposition=False, deconvolution=False):
    if type_key in reference.varm and min_distance:
        raise ValueError('Supplying the reference as average profiles does not work with min_distance!')
    if decomposition and min_distance:
        raise ValueError('The decomposition workflow does not work with min_distance!')
    if deconvolution and min_distance:
        raise ValueError('The deconvolution workflow does not work with min_distance!')
    if deconvolution and decomposition:
        raise ValueError('The deconvolution and decomposition workflow dont work together!')
    
    if (np.array(tdata.X.sum(axis=1)).flatten()==0).any():
        raise ValueError('There are observations without non-zero variables!')
    
    tdata, reference = tdata.copy(), reference.copy() # dont touch the originals

    if log_norm:
        tdata = log_normalize(tdata, scale=scale, type_key=type_key)
        reference = log_normalize(reference, scale=scale, type_key=type_key)
    
    tX, rX, pca_offset, pca_trafo = _transfer_pca(tdata, reference, n_pca, zero_center=zero_center)
    
    if min_distance:
        # find min distances from one cell to every distinct other cell
        cref_cell_dist = utils.cdist(rX, tX, metric)
        cref_cell_dist[cref_cell_dist < 1e-10] = np.inf # make equal cell comparisons irrelevant for the min operation
        types = reference.obs[type_key].unique()
        type_cell_dist = pd.DataFrame({ l: np.min(cref_cell_dist[reference.obs[type_key] == l],axis=0) for l in types }, index=tdata.obs_names.copy())[types].T
    else:
        average_profiles = utils.get_average_profiles(type_key, reference, rX, pca_offset, pca_trafo)
        if deconvolution or metric in ['weighted','nwsp',]:
            if metric == 'h2':
                metric = 'bc'
            proj = utils.projection(tX, average_profiles.T, metric=metric, deconvolution=deconvolution)
            _dist = np.maximum(1-proj,0).T
        else:
            _dist = utils.cdist(average_profiles.T, tX, metric=metric)
        type_cell_dist = pd.DataFrame(_dist, index=average_profiles.columns.copy(), columns=tdata.obs_names.copy()) # have to use copies of the indices as otherwise there is still a reference to temporary objects from within the function, which cannot be cleaned up by the garbage collector...
        if decomposition:
            mixture_profiles, mixtures = utils.generate_mixture_profiles(average_profiles.to_numpy(), include_pure_profiles=(decomposition<2))
            type_mix_dist = pd.DataFrame(utils.cdist(average_profiles.T, mixture_profiles, metric=metric), index=average_profiles.columns.copy())
            type_cell_dist = pd.concat([type_cell_dist,type_mix_dist],axis=1)
    
    del tdata, reference # clean up the copies.
    gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    if decomposition:
        return type_cell_dist, mixtures
    else:
        return type_cell_dist
    
def prep_priors_nnls(tdata, reference, type_key, n_pca=None, zero_center=True, reads=True):
    
    cell_prior = prep_cell_priors(tdata, reads=reads)
    
    tdata, reference = tdata.copy(), reference.copy() # dont touch the originals
    
    tX, rX, pca_offset, pca_trafo = _transfer_pca(tdata, reference, n_pca, zero_center=zero_center)

    pseudo_bulk = np.array(tX.sum(axis=0)).flatten()
    
    # create average type profiles and use them to get the marginal distributions of celltypes via nnls
    average_profiles = utils.get_average_profiles(type_key, reference, rX, pca_offset, pca_trafo)

    type_prior = pd.Series(nnls(average_profiles, pseudo_bulk)[0],index=average_profiles.columns)
    
    if type_key is not None:
        # fix missing types
        type_prior = pd.Series({t: type_prior[t] if t in type_prior else 0.0 for t in reference.obs[type_key].unique() })
    
    del tdata, reference # clean up the copies.
    gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    return type_prior, cell_prior
    
def prep_priors_reference(tdata, reference, type_key, reads=True, mixture_contribution=0):
        
    # get identity mapping
    if type_key in reference.obs:
        cell_profile_mapping = pd.get_dummies(reference.obs[type_key])
    elif type_key in reference.obsm:
        cell_profile_mapping = reference.obsm[type_key].fillna(0)
    elif type_key in reference.varm:
        print('Reference type priors can only be automatically determined from `.obs` and `.obsm` annotation! Using flat priors...')
        type_names = reference.varm[type_key].columns
        cell_profile_mapping = pd.DataFrame(np.ones((reference.shape[0],len(type_names))),columns=type_names,index=reference.obs.index)
    else:
        raise ValueError('Without any annotation in `.obs`, `.obsm`, or `.varm` reference type priors cannot be determined!')
    cell_profile_mapping /= cell_profile_mapping.sum(axis=1).to_numpy()[:,None] # every cell sums up to 1

    if reads and np.prod(reference.shape) > 0 and reference.X is not None:
        type_prior = np.array(reference.X.sum(axis=1)).flatten() @ cell_profile_mapping
    else:
        type_prior = cell_profile_mapping.sum(axis=0)
    
    return type_prior, prep_cell_priors(tdata, reads=reads)

def validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=False):
    
    if adata is None:
        raise ValueError('"adata" cannot be None!')
    if reference is None:
        raise ValueError('"reference" cannot be None!')
        
    annotation_key = _infer_annotation_key(reference, annotation_key)
    
    tdata = get.counts(adata, counts_location=counts_location, annotation=True, copy=False)
    reference = get.counts(reference, counts_location=counts_location, annotation=annotation_key, copy=False)
    
    if np.prod(tdata.shape) == 0:
        raise ValueError('The data provided via `adata` (either directly or via the `counts_location` parameter) must have at least one observation/cell and variable/gene!')
    if full_reference:
        if np.prod(reference.shape) == 0:
            raise ValueError('The data provided via `reference` (either directly or via the `counts_location` parameter) must have at least one observation/cell and variable/gene!')
    
#    if annotation_key in reference.varm:
#        reference = preprocessing.filter_profiles(adata=reference, annotation_key=annotation_key, fill_na=None, fill_negative=None) # filter out zero-only genes in the profiles
    if annotation_key in reference.obsm:
        reference = preprocessing.filter_annotation(adata=reference, annotation_key=annotation_key, fill_na=None, fill_negative=None) # filter out zero-only cells in the annotation

    tdata,reference = preprocessing.filter(adata=(tdata,reference)) # ensure consistent gene selection
    
    return tdata, reference, annotation_key

def guess_obs_index_key(obs_index_key, adata):
    if obs_index_key is None:
        obs_index_key = adata.obs.index.name
        if obs_index_key is None:
            obs_index_key = 'index'
        if obs_index_key in adata.obs:
            raise ValueError('The obs_index_key "%s" determined by the obs_index_key-finding-heuristic is already used in .obs! Please specify a obs_index_key explicitly.')
    return obs_index_key

def guess_var_index_key(var_index_key, adata):
    if var_index_key is None:
        var_index_key = adata.var.index.name
        if var_index_key is None:
            var_index_key = 'gene'
        if var_index_key in adata.var:
            raise ValueError('The var_index_key "%s" determined by the var_index_key-finding-heuristic is already used in .var! Please specify a var_index_key explicitly.')
    return var_index_key

def guess_merge_key(merge_key, adata, annotation, profiles):
    if merge_key is None:
        merge_key = annotation.columns.name
        if merge_key is None:
            merge_key = profiles.columns.name
        if merge_key in adata.obs:
            raise ValueError('The merge_key "%s" determined by the merge_key-finding-heuristic is already used in .obs! Please specify a merge_key explicitly.')
    return merge_key

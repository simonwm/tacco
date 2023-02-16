import numpy as np
import pandas as pd
import scanpy as sc
from numba import njit
from .. import utils
from .. import get

def _find_neighbours(adata, annotation_key, key_added, amplitudes=False, **kw_args):
    """ Runs sc.pp.neighbours and handles pd.DataFrames correctly... TODO: Should probably fix that in scanpy. """
    _annotation_key = annotation_key
    if annotation_key is not None and isinstance(adata.obsm[annotation_key], pd.DataFrame):
        _annotation_key = utils.find_unused_key(adata.obsm)
        adata.obsm[_annotation_key] = adata.obsm[annotation_key].to_numpy()
    if amplitudes:
        if annotation_key is None or _annotation_key == annotation_key:
            _annotation_key = utils.find_unused_key(adata.obsm)
            if annotation_key is None:
                adata.obsm[_annotation_key] = adata.X
            else:
                adata.obsm[_annotation_key] = adata.obsm[annotation_key]
        adata.obsm[_annotation_key] = np.sqrt(adata.obsm[_annotation_key] / utils.get_sum(adata.obsm[_annotation_key], axis=1)[:,None])
    sc.pp.neighbors(adata, random_state=42, use_rep=_annotation_key, key_added=key_added, **kw_args)
    if _annotation_key != annotation_key:
        del adata.obsm[_annotation_key]

def find_regions(
    adata,
    key_added='region',
    batch_key=None,
    position_weight=3.0,
    resolution=1.0,
    annotation_connectivity=None,
    annotation_key=None,
    position_connectivity=None,
    position_key=('x','y'),
    batch_position_distance=None,
    amplitudes=False,
    cross_batch_overweight_factor=1,
    **kw_args,
    ):

    """\
    Find regions in position space defined by proximity in position space and
    in expression (or annotation) space.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    key_added
        The `.obs` key where the resulting regions are written to. It is also
        used to determine where new neighbour and connectivity results should
        be written to, see parameters `connectivity` and
        `position_connectivity`.
    batch_key
        The `.obs` key with categorical batch information. The expression
        connectivity is then balanced within and between batches. If `None`, a
        single batch is assumed.
    position_weight
        The weight of the position connectivity compared to the expression
        connectivity.
    resolution
        The clustering resolution.
    annotation_connectivity
        The `.obsp` key where the precomputed expression connectivity can be
        found. If `None`, a new connectivity is computed with
        :func:`~scanpy.pp.neighbors` and default parameters and then saved
        under "nn_<key_added>_annotation".
    annotation_key
        The `.obsm` key containing data to use instead of expression. If
        `None`, use expression data.
    position_connectivity
        The `.obsp` key where the precomputed position connectivity can be
        found. If `None`, a new connectivity is computed with
        :func:`scanpy.pp.neighbors` and default parameters and then saved
        under "nn_<key_added>_position".
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates
    batch_position_distance
        The additional position distance to apply to observations between
        batches. If `None`, it is taken much larger than any distance within a
        batch. 
    amplitudes
        Whether to treat the annotation as probabilities and calculate
        amplitudes from them
    cross_batch_overweight_factor
        The factor to increase cross batch connection weight over within batch
        connections.
    **kw_args
        Additional key word arguments are forwarded to
        :func:`~scanpy.pp.neighbors`. Particularly interesting is probably
        `n_neighbors`.
        
    Returns
    -------
    The updated input `adata` containing e.g. the `obs[key_added]` containing\
    the region information.
    
    """

    key_added_annotation = 'nn_' + key_added + '_annotation'
    if annotation_connectivity is not None:
        if annotation_connectivity not in adata.obsp:
            raise ValueError('The key "%s" for annotation_connectivity is not available in .obsp!' % annotation_connectivity)
        annotation_connectivity = adata.obsp[annotation_connectivity]
    else:
        annotation_connectivity = key_added_annotation + '_connectivities'
        if annotation_connectivity in adata.obsp:
            annotation_connectivity = adata.obsp[annotation_connectivity]
        else:
            _find_neighbours(adata, annotation_key, key_added_annotation, amplitudes=amplitudes, **kw_args)
            annotation_connectivity = adata.obsp[annotation_connectivity]

    key_added_position = 'nn_' + key_added + '_position'
    if position_connectivity is not None:
        if position_connectivity not in adata.obsp:
            raise ValueError('The key "%s" for position_connectivity is not available in .obsp!' % position_connectivity)
        if batch_key is not None:
            print('Care must be taken to compute connectivities with a batch key included! Make sure this was properly done for the connectivities under .obsp["%s"]!' % position_connectivity)
        position_connectivity = adata.obsp[position_connectivity]
    else:
        position_connectivity = key_added_position + '_connectivities'
        if position_connectivity in adata.obsp:
            if batch_key is not None:
                print('Care must be taken to compute connectivities with a batch key included! Make sure this was properly done for the connectivities under .obsp["%s"]!' % position_connectivity)
            position_connectivity = adata.obsp[position_connectivity]
        else:
            positions = get.positions(adata, position_key)
            if not isinstance(position_key, str):
                position_key = '_'.join(position_key)

            if batch_key is not None: # let data from different batches have some fixed (large) spatial distance offset
                if batch_position_distance is None:
                    positions_range = positions.max().max() - positions.min().min()
                    batch_position_distance = 2.0 * positions_range
                if batch_position_distance != 0:
                    positions = np.hstack([positions, batch_position_distance * pd.get_dummies(adata.obs[batch_key])])

            new_position_key = 'X_' + position_key
            while new_position_key in adata.obsm:
                new_position_key += '1'
            adata.obsm[new_position_key] = positions

            _find_neighbours(adata, new_position_key, key_added_position, **kw_args)
            position_connectivity = adata.obsp[position_connectivity]

            del adata.obsm[new_position_key]

    if batch_key is not None: # batch balance the connectivities across batches
        batch_dummy = pd.get_dummies(adata.obs[batch_key])
        batch_dummy /= batch_dummy.sum(axis=0).to_numpy()
        batch_connectivity = batch_dummy.T @ (annotation_connectivity @ batch_dummy)
        cross_batch_rescaling_factor = np.diag(batch_connectivity).sum() / batch_connectivity.to_numpy()[np.triu_indices(batch_connectivity.shape[0],1)].sum()
        cross_batch_rescaling_factor *= cross_batch_overweight_factor
        @njit
        def upscale_cross_batch(data,row,col,batch):
            for i in range(len(data)):
                if batch[row[i]] != batch[col[i]]:
                    data[i] *= cross_batch_rescaling_factor
        _connectivity = annotation_connectivity.tocoo()
        if _connectivity is annotation_connectivity:
            _connectivity = annotation_connectivity.copy()
        annotation_connectivity = _connectivity
        upscale_cross_batch(annotation_connectivity.data,annotation_connectivity.row,annotation_connectivity.col,adata.obs[batch_key].cat.codes.to_numpy())
        annotation_connectivity = annotation_connectivity.tocsr()

    sum_weights = 1 + position_weight
    position_weight /= sum_weights
    annotation_weight = 1 - position_weight

    adjacency = annotation_weight * annotation_connectivity + position_weight * position_connectivity

    adata.obsp[key_added + '_connectivities'] = adjacency
    
    sc.tl.leiden(adata, resolution=resolution, random_state=42, key_added=key_added, adjacency=adjacency)
    
    return adata

def fill_regions(
    adata,
    region_key='region',
    batch_key=None,
    position_key=('x','y'),
    ):

    """\
    Fills the region annotation of not annotated observation (i.e. na values)
    with the annotation of the spatially closest annotated observation.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including region and position annotation.
    region_key
        The `.obs` key where the region annotation is stored.
    batch_key
        The `.obs` key with categorical batch information. If not `None`, the
        filling is performed per batch.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates.
        
    Returns
    -------
    The updated input `adata` containing the filled `obs[region_key]`.
    
    """
    
    if region_key not in adata.obs.columns:
        raise ValueError(f'The `region_key` "{region_key}" is not a column name in `adata.obs`')

    positions = get.positions(adata, position_key)
    
    if batch_key is None:
        batches = pd.Series(1,index=adata.obs.index)
    else:
        if batch_key not in adata.obs.columns:
            raise ValueError(f'The `batch_key` "{batch_key}" is not a column name in `adata.obs`')
        batches = adata.obs[batch_key]
    
    def get_closest_annotation(batch):
        all_pos = positions.loc[batch.index]
        all_anno = adata.obs.loc[batch.index,region_key].copy()
        anno = all_anno.dropna()
        pos = all_pos.loc[anno.index]
        new_pos = all_pos.loc[all_anno.index.difference(anno.index)]
        dists = utils.cdist(new_pos, pos, metric='euclidean')
        new_anno = pd.Series(anno.iloc[np.argmin(dists, axis=1)].to_numpy(), index=new_pos.index)
        return pd.concat([anno, new_anno]).reindex_like(all_anno)
    
    adata.obs[region_key] = batches.groupby(batches).transform(get_closest_annotation)
    
    return adata

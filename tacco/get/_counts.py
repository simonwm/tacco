import pandas as pd
import anndata as ad

def _get_counts_location(adata, counts_location=None):
    if counts_location is None:
        if 'counts_location' in adata.uns:
            counts_location = adata.uns['counts_location']
        else:
            counts_location = 'X'
    
    if isinstance(counts_location, str):
        counts_location = (counts_location,)
    
    if len(counts_location) < 1:
        counts_location = ('X',)
    
    def _check_excess_spec(counts_location, num):
        if len(counts_location) > num:
            raise Exception('There are excess elements %s in the counts location specification!' % str(counts_location[num:]))
    
    if counts_location[0] == 'X':
        _check_excess_spec(counts_location, num=1)
    elif counts_location[0] == 'layer':
        if len(counts_location) < 2:
            raise Exception('If specifying the counts location in a layer, you have to specify a layer name in the next element of the tuple, e.g. counts_location=("layer","my_bare_counts")')
        _check_excess_spec(counts_location, num=2)
    elif counts_location[0] == 'obsm':
        if len(counts_location) < 2:
            raise Exception('If specifying the counts location in obsm, you have to specify a obsm key in the next element of the tuple, e.g. counts_location=("obsm","my_bare_counts")')
        _check_excess_spec(counts_location, num=2)
    elif counts_location[0] == 'varm':
        if len(counts_location) < 2:
            raise Exception('If specifying the counts location in varm, you have to specify a varm key in the next element of the tuple, e.g. counts_location=("varm","my_bare_counts")')
        _check_excess_spec(counts_location, num=2)
    elif counts_location[0] == 'uns':
        if len(counts_location) < 2:
            raise Exception('If specifying the counts location in uns, you have to specify a uns key in the next element of the tuple, e.g. counts_location=("uns","my_bare_counts")')
        _check_excess_spec(counts_location, num=2)
    elif counts_location[0] == 'raw':
        counts_location = (counts_location[0], *_get_counts_location(adata.raw.to_adata(), counts_location=counts_location[1:]))
    else:
        raise Exception('The counts location "%s" cannot be interpreted!' % str(counts_location))
    
    return counts_location

def _get_var_names(X, adata, key):
    var = None
    try:
        var = pd.DataFrame(index=X.columns,columns=[])
    except:
        for ext in ('var_names', 'var', 'col', 'cols', 'columns', 'varnames', 'colnames'):
            key_ext = key + '_' + ext
            if key_ext in adata.uns:
                var = pd.DataFrame(index=adata.uns[key_ext])
                break
    return var

def _get_obs_names(X, adata, key):
    obs = None
    try:
        obs = pd.DataFrame(index=X.index,columns=[])
    except:
        for ext in ('obs_names', 'obs', 'row', 'rows', 'index', 'obsnames', 'rownames'):
            key_ext = key + '_' + ext
            if key_ext in adata.uns:
                obs = pd.DataFrame(index=adata.uns[key_ext])
                break
    return obs

def _get_obsm_counts(adata, key):
    X = adata.obsm[key]
    return ad.AnnData(X=X, obs=adata.obs[[]], var=_get_var_names(X, adata, key))

def _get_varm_counts(adata, key):
    X = adata.varm[key].T
    return ad.AnnData(X=X, obs=_get_obs_names(X,adata, key), var=adata.var[[]])

def _get_uns_counts(adata, key):
    X = adata.uns[key]
    return ad.AnnData(X=X, obs=_get_obs_names(X, adata, key), var=_get_var_names(X, adata, key))

def get_counts(
    adata,
    counts_location=None,
    annotation=False,
    copy=False
):

    """\
    Get an :class:`~anndata.AnnData` with bare counts in `.X` from an
    :class:`~anndata.AnnData` where the counts are buried somewhere else.

    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`.
    counts_location
        The "path" in `adata` where the counts are stored as a string (e.g.
        "X") or as a list-like of strings, e.g. `("raw","obsm","counts")`.
        Possible parts of paths are:
        
        - "X"
        - "layer"
        - "obsm"
        - "varm"
        - "uns"
        - "raw"
        
        If `None`, looks in `adata.uns['counts_location']` for the path and if
        unsuccessful uses "X".
    annotation
        What if any annotation should be included in the returned
        :class:`~anndata.AnnData`. Possible are:
        
        - `False`: dont take any annotation
        - `True`: take all `.obs`, `.var`, `.uns`, and those `.obsm` and
          `.varm` annotations, which are a :class:`~pandas.DataFrame`.
        - string or list-like of strings: take all `.obs`, `.var`, `.obsm`,
          `.varm`, and `.uns` annotations with these names.
         
    copy
        Whether to return a copy of the original `adata`, even if the counts
        are already in `.X`.
        
    Returns
    -------
    An :class:`~anndata.AnnData` with counts in `.X`.
    
    """

    counts_location = _get_counts_location(adata, counts_location=counts_location)
    
    # return a fresh clean anndata only with counts and row/col index, mostly backed by data from the original adata
    if counts_location[0] == 'X':
        if copy == False:
            return adata
        counts = ad.AnnData(X=adata.X, obs=adata.obs[[]], var=adata.var[[]])
    elif counts_location[0] == 'layer':
        counts = ad.AnnData(X=adata.layers[counts_location[1]], obs=adata.obs[[]], var=adata.var[[]])
    elif counts_location[0] == 'obsm':
        counts = _get_obsm_counts(adata, counts_location[1])
    elif counts_location[0] == 'varm':
        counts = _get_varm_counts(adata, counts_location[1])
    elif counts_location[0] == 'uns':
        counts = _get_uns_counts(adata, counts_location[1])
    elif counts_location[0] == 'raw':
        counts = get_counts(adata.raw.to_adata(), counts_location=counts_location[1:])
    else:
        raise Exception('The counts location "%s" cannot be interpreted!' % str(counts_location))
    
    if isinstance(annotation, str):
        annotation = [annotation]
    if isinstance(annotation, bool):
        if annotation:
            counts.obs = adata.obs.reindex(index=counts.obs.index)
            counts.var = adata.var.reindex(index=counts.var.index)
            for key in adata.obsm:
                try:
                    counts.obsm[key] = adata.obsm[key].reindex(index=counts.obs.index) # obsm can hold other things than df (e.g. sparse matrix)
                except:
                    pass
            for key in adata.varm:
                try:
                    counts.varm[key] = adata.varm[key].reindex(index=counts.var.index) # varm can hold other things than df (e.g. sparse matrix)
                except:
                    pass
            for key in adata.uns:
                counts.uns[key] = adata.uns[key]
    else:
        for anno in annotation:
            if anno in adata.obs:
                counts.obs[anno] = adata.obs[anno].reindex(index=counts.obs.index)
            if anno in adata.var:
                counts.var[anno] = adata.var[anno].reindex(index=counts.var.index)
            if anno in adata.obsm:
                counts.obsm[anno] = adata.obsm[anno].reindex(index=counts.obs.index)
            if anno in adata.varm:
                counts.varm[anno] = adata.varm[anno].reindex(index=counts.var.index)
            if anno in adata.uns:
                counts.uns[anno] = adata.uns[anno]
    
    return counts


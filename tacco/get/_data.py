import pandas as pd
import anndata as ad
from scipy.sparse import issparse as _issparse

def _as_series(series_like, index, name):
    """\
    If the series_like is already a :class:`~pandas.Series`, pass through,
    otherwise wrap in a :class:`~pandas.Series`.
    """
    if not isinstance(series_like, pd.Series):
        if _issparse(series_like):
            series_like = series_like.toarray()
        if len(series_like.shape) > 1:
            series_like = series_like.flatten()
        series_like = pd.Series(series_like,index=index,name=name)
    return series_like

def _as_frame(frame_like, index, columns):
    """\
    If the frame_like is already a :class:`~pandas.DataFrame`, pass through,
    otherwise wrap in a :class:`~pandas.DataFrame`.
    """
    if not isinstance(frame_like, pd.DataFrame):
        if _issparse(frame_like):
            frame_like = frame_like.toarray()
        col_args = {} if columns == ... else {'columns':columns}
        frame_like = pd.DataFrame(frame_like,index=index,**col_args)
    return frame_like

def _as_list(list_like_or_element):
    """\
    Wrap the list_like_or_element in a list.
    """
    _list = list_like_or_element
    if isinstance(_list,str): # wrap strings
        _list = [_list]
    _list = list(_list) # convert tuples etc.
    return _list

def get_data_from_key(
    adata,
    key,
    default_key=None,
    key_name='key',
    check_uns=False,
    search_order=('X','layer','obs','var','obsm','varm'),
    check_uniqueness=True,
    result_type='obs',
    verbose=1,
    raise_on_error=True,
):

    """\
    Given a short key for data stored somewhere in an :class:`~anndata.AnnData`
    find the data associated with it following a deterministic scheme.

    Parameters
    ----------
    adata
        A :class:`~anndata.AnnData`; if it is a :class:`~pandas.DataFrame`, use
        it in place of a `adata.obs` or `adata.var` data frame.
    key
        The short key or "path" in `adata` where the data is stored as a string
        (e.g. "X") or as a list-like of strings, e.g. `("raw","obsm","counts")`,
        where the last element can be a list-like of stings, e.g.
        `("obsm","types",["B","T"])`.
        Possible parts of paths are:
        
        - "X"
        - "layer"
        - "obs"
        - "var"
        - "obsm"
        - "varm"
        - "raw"
        
        If `None` and `check_uns` is not `False`, looks in
        `adata.uns[key_name]` for the key and if unsuccessful uses
        `default_key`.
    default_key
        The default location to use; see `key`.
    key_name
        The name of the key to use for lookup in `.uns`; see `key`. Also used
        for meaningful error messages.
    check_uns
        Whether to check `.uns` for the key; if a string, overrides the value
        of `key_name` for the lookup; see `key` and `key_name`.
    search_order
        In which order to check the properties of :class:`~anndata.AnnData` for
        the `key`, if it is a string. The first hit will be returned. If
        `check_uniqueness`, the remaining properties will be checked for
        uniqueness of the hit, generating a warning for non-unique hits. In
        addition to the properties 'X','layer','obs','var','obsm', and 'varm',
        the two pseudo properties 'multi-obs' and 'multi-var' can be specified
        to allow for the selection of multiple obs/var columns.
    check_uniqueness
        Whether to check for uniqueness of the hit; see `search_order`.
    result_type
        What type of data to look for. Options are:
        
        - "obs": return a :class:`~pandas.Series` with index like a `.obs`
          column
        - "var": return a :class:`~pandas.Series` with index like a `.var`
          column
        - "obsm": return a :class:`~pandas.DataFrame` with index like a `.obs`
          column
        - "varm": return a :class:`~pandas.DataFrame` with index like a `.var`
          column
        - "X": return an array-like of shape compatible with `adata`

        If "obs" or "obsm", then "var" and "varm" will be excluded from the
        `search_order`. If "var" or "varm", then "obs" and "obsm" will be
        excluded from the `search_order`. If "X", then "obs" and "var" will be
        excluded from the `search_order`.
    verbose
        Level of verbosity, with `0` (no output), `1` (some output), ...
    raise_on_error
        Whether to raise on errors. Alternatively just return `None`.
        
    Returns
    -------
    Depending on result_type, a :class:`~pandas.Series`,\
    :class:`~pandas.DataFrame`, or array-like containing the data (or `None`\
    on failure it not `raise_on_error`).
    
    """

    # generic exception handling
    def raise_Exception(msg):
        if raise_on_error:
            raise ValueError(msg)
        elif verbose > 0:
            print(msg)
        return None
    def print_Message(msg, verbose):
        if verbose > 0:
            print(msg)
    
    # extra layer of indirection for adata access
    if isinstance(adata, ad.AnnData):
        adata_obs = adata.obs
        adata_var = adata.var
        adata_obsm = adata.obsm
        adata_varm = adata.varm
        adata_uns = adata.uns
        adata_layers = adata.layers
        real_adata = True
    else: # support dataframes as replacement for adata.obs and adata.var
        # result_type indicates whether obs or var should be used, so we can populate both here
        adata_obs = adata
        adata_var = adata
        # put empty dictionaries here to enable the 'is in' evaluations below
        adata_obsm = {}
        adata_varm = {}
        adata_uns = {}
        adata_layers = {}
        # here we need explicit code changes below for .X and .raw access...
        real_adata = False
        def no_real_adata_message(requested_property):
            return f'The key path for the key named {key_name!r} is {key!r} and contains "{requested_property}", but the supplied adata is of type "{type(adata)!r}"! Specifying "{requested_property}" paths in the key is only possible if adata is indeed an AnnData object!'

    # ensure sanity of `key`
    if key is None and check_uns != False: # check_uns can also be a string, so `if check_uns` is not sufficient
        _key_name = key_name if isinstance(check_uns, bool) else check_uns
        if _key_name in adata_uns:
            key = adata_uns[_key_name]
    if key is None and not default_key is None:
        key = default_key
    if key is None:
        return raise_Exception(f'The key named {key_name!r} is `None` and did not get valid options for `default_key` and `check_uns`!')

    # ensure sanity of `result_type`
    valid_result_types = ['obs','obsm','var','varm','X']
    if result_type not in valid_result_types:
        return raise_Exception(f'The result_type {result_type!r} is invalid! Only {valid_result_types!r} can be used.')
    

    def _get_hit():
        # look for hits
        hits = []
        for element in search_order:
            if element == 'X': # look in the counts
                if not isinstance(key, str):
                    continue
                if result_type in ['obs','obsm']: # look for a gene in the counts
                    if key in adata_var.index:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
                elif result_type in ['var','varm']: # look for a cell in the counts
                    if key in adata_obs.index:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'layer': # look in the layers
                if not isinstance(key, str):
                    continue
                if result_type in ['X']: # only full count matrices can be specified like this
                    if key in adata_layers:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'obs': # look in the obs annotation
                if not isinstance(key, str):
                    continue
                if result_type in ['obs','obsm']: # look for a single annotation column
                    if key in adata_obs:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'var': # look in the var annotation
                if not isinstance(key, str):
                    continue
                if result_type in ['var','varm']: # look for a single annotation row
                    if key in adata_var:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'obsm': # look in the obsm annotation
                if not isinstance(key, str):
                    continue
                if result_type in ['obsm']: # obsms only fit to obsms
                    if key in adata_obsm:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'varm': # look in the varm annotation
                if not isinstance(key, str):
                    continue
                if result_type in ['varm']: # varms only fit to varms
                    if key in adata_varm:
                        hits.append((element,key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'multi-obs': # look for a set of obs annotations
                if isinstance(key, str):
                    continue
                if result_type in ['obsm']: # a set of obs annotations only fit to obsms
                    if all(_key in adata_obs for _key in key):
                        hits.append(('obs',key))
                        if not check_uniqueness:
                            return hits[0]
            elif element == 'multi-var': # look for a set of var annotations
                if isinstance(key, str):
                    continue
                if result_type in ['varm']: # a set of var annotations only fit to varms
                    if all(_key in adata_var for _key in key):
                        hits.append(('var',key))
                        if not check_uniqueness:
                            return hits[0]
            else:
                raise_Exception(f'The element {element!r} of search_order {search_order!r} is not valid!')
        if len(hits) == 0:
            if isinstance(key, str):
                raise_Exception(f'The key {key!r} was not found anywhere in {search_order!r} for a result_type of {result_type!r}!')
            else:
                hits.append(key)
        elif len(hits) > 1:
           print_Message(f'The key {key!r} was not found in more than one location: {hits!r}! Continue using the first hit {hits[0]!r}', verbose=verbose)
        return hits[0]
    key = _get_hit()

    # look at the specified position
    if len(key) < 1:
        return raise_Exception(f'The key named {key_name!r} is {key!r} which is not a valid path! Only list-likes of length > 0 could be valid.')

    if key[0] == 'raw':
        if not isinstance(adata, ad.AnnData):
            return raise_Exception(no_real_adata_message('raw'))
        return get_data_from_key(
            adata=adata.raw.to_adata(),
            key=key[1:], 
            default_key=default_key,
            key_name=key_name,
            check_uns=check_uns,
            search_order=search_order,
            check_uniqueness=check_uniqueness,
            result_type=result_type,
            verbose=verbose,
            raise_on_error=raise_on_error,
        )


    if result_type == 'obs':
        if key[0] == 'X':
            if not isinstance(adata, ad.AnnData):
                return raise_Exception(no_real_adata_message('X'))
            if len(key) != 2:
                return raise_Exception(f'An obs result_type can only be retrieved from "X" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_var.index:
                return raise_Exception(f'An obs result_type can only be retrieved from "X" if the second path element matches a gene name, but {key[1]!r} is not an available gene name!')
            return _as_series(adata[:,[key[1]]].X,index=adata_obs.index,name=key[1])
        elif key[0] == 'layer':
            if len(key) != 3:
                return raise_Exception(f'An obs result_type can only be retrieved from "layer" with a key with length of exactly 3, but {key!r} was given!')
            if key[1] not in adata_layers:
                return raise_Exception(f'An obs result_type can only be retrieved from "layer" if the second path element matches a layer name, but {key[1]!r} is not an available layer!')
            if key[2] not in adata_var.index:
                return raise_Exception(f'An obs result_type can only be retrieved from "layer" if the third path element matches a gene name, but {key[2]!r} is not an available gene name!')
            return _as_series(adata[:,[key[2]]].layers[key[1]],index=adata_obs.index,name=key[2])
        elif key[0] == 'obs':
            if len(key) != 2:
                return raise_Exception(f'An obs result_type can only be retrieved from "obs" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_obs.columns:
                return raise_Exception(f'An obs result_type can only be retrieved from "obs" if the second path element matches a obs key, but {key[1]!r} is not an available obs column!')
            return _as_series(adata_obs[key[1]],index=adata_obs.index,name=key[1])
        elif key[0] == 'obsm':
            if len(key) != 3:
                return raise_Exception(f'An obs result_type can only be retrieved from "obsm" with a key with length of exactly 3, but {key!r} was given!')
            if key[1] not in adata_obsm:
                return raise_Exception(f'An obs result_type can only be retrieved from "obsm" if the second path element matches an obsm name, but {key[1]!r} is not an available obsm!')
            if key[2] not in adata_obsm[key[1]].columns:
                return raise_Exception(f'An obs result_type can only be retrieved from "obsm" if the third path element matches a column name of the selected obsm, but {key[2]!r} is not a available there!')
            return _as_series(adata_obsm[key[1]][key[2]],index=adata_obs.index,name=key[2])
        else:
            return raise_Exception(f'An obs result_type can only be retrieved from "X","layer","obs", and "obsm" paths, but {key!r} starts with {key[0]!r}!')

    elif result_type == 'var':
        if key[0] == 'X':
            if not isinstance(adata, ad.AnnData):
                return raise_Exception(no_real_adata_message('X'))
            if len(key) != 2:
                return raise_Exception(f'An var result_type can only be retrieved from "X" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_obs.index:
                return raise_Exception(f'An var result_type can only be retrieved from "X" if the second path element matches a cell name, but {key[1]!r} is not an available cell name!')
            return _as_series(adata[[key[1]],:].X,index=adata_var.index,name=key[1])
        elif key[0] == 'layer':
            if len(key) != 3:
                return raise_Exception(f'An var result_type can only be retrieved from "layer" with a key with length of exactly 3, but {key!r} was given!')
            if key[1] not in adata_layers:
                return raise_Exception(f'An var result_type can only be retrieved from "layer" if the second path element matches a layer name, but {key[1]!r} is not an available layer!')
            if key[2] not in adata_obs.index:
                return raise_Exception(f'An var result_type can only be retrieved from "layer" if the third path element matches a cell name, but {key[2]!r} is not an available cell name!')
            return _as_series(adata[[key[2]],:].layers[key[1]],index=adata_var.index,name=key[2])
        elif key[0] == 'var':
            if len(key) != 2:
                return raise_Exception(f'An var result_type can only be retrieved from "var" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_var.columns:
                return raise_Exception(f'An var result_type can only be retrieved from "var" if the second path element matches a var key, but {key[1]!r} is not an available var column!')
            return _as_series(adata_var[key[1]],index=adata_var.index,name=key[1])
        elif key[0] == 'varm':
            if len(key) != 3:
                return raise_Exception(f'An var result_type can only be retrieved from "varm" with a key with length of exactly 3, but {key!r} was given!')
            if key[1] not in adata_varm:
                return raise_Exception(f'An var result_type can only be retrieved from "varm" if the second path element matches an varm name, but {key[1]!r} is not an available varm!')
            if key[2] not in adata_varm[key[1]].columns:
                return raise_Exception(f'An var result_type can only be retrieved from "varm" if the third path element matches a column name of the selected varm, but {key[2]!r} is not a available there!')
            return _as_series(adata_varm[key[1]][key[2]],index=adata_var.index,name=key[2])
        else:
            return raise_Exception(f'An var result_type can only be retrieved from "X","layer","var", and "varm" paths, but {key!r} starts with {key[0]!r}!')

    elif result_type == 'obsm':
        if key[0] == 'X':
            if not isinstance(adata, ad.AnnData):
                return raise_Exception(no_real_adata_message('X'))
            if len(key) != 2:
                return raise_Exception(f'An obsm result_type can only be retrieved from "X" with a key with length of exactly 2, but {key!r} was given!')
            columns = _as_list(key[1])
            for column in columns:
                if column not in adata_var.index:
                    return raise_Exception(f'An obsm result_type can only be retrieved from "X" if every element of the second path element matches a gene name, but {column!r} is not an available gene name!')
            return _as_frame(adata[:,columns].X,index=adata_obs.index,columns=columns)
        elif key[0] == 'layer':
            if len(key) != 3:
                return raise_Exception(f'An obsm result_type can only be retrieved from "layer" with a key with length of exactly 3, but {key!r} was given!')
            if key[1] not in adata_layers:
                return raise_Exception(f'An obsm result_type can only be retrieved from "layer" if the second path element matches a layer name, but {key[1]!r} is not an available layer!')
            columns = _as_list(key[2])
            for column in columns:
                if column not in adata_var.index:
                    return raise_Exception(f'An obsm result_type can only be retrieved from "layer" if every element of the third path element matches a gene name, but {column!r} is not an available gene name!')
            return _as_frame(adata[:,columns].layers[key[1]],index=adata_obs.index,columns=columns)
        elif key[0] == 'obs':
            if len(key) != 2:
                return raise_Exception(f'An obsm result_type can only be retrieved from "obs" with a key with length of exactly 2, but {key!r} was given!')
            columns = _as_list(key[1])
            for column in columns:
                if column not in adata_obs.columns:
                    return raise_Exception(f'An obsm result_type can only be retrieved from "obs" if every element of the second path element matches a obs key, but {column!r} is not an available obs column!')
            return _as_frame(adata_obs[columns],index=adata_obs.index,columns=columns)
        elif key[0] == 'obsm':
            if len(key) != 2:
                return raise_Exception(f'An obsm result_type can only be retrieved from "obsm" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_obsm:
                return raise_Exception(f'An obsm result_type can only be retrieved from "obsm" if the second path element matches an obsm name, but {key[1]!r} is not an available obsm!')
            return _as_frame(adata_obsm[key[1]],index=adata_obs.index,columns=...)
        else:
            return raise_Exception(f'An obsm result_type can only be retrieved from "X","layer","obs", and "obsm" paths, but {key!r} starts with {key[0]!r}!')

    elif result_type == 'varm':
        if key[0] == 'X':
            if not isinstance(adata, ad.AnnData):
                return raise_Exception(no_real_adata_message('X'))
            if len(key) != 2:
                return raise_Exception(f'An varm result_type can only be retrieved from "X" with a key with length of exactly 2, but {key!r} was given!')
            columns = _as_list(key[1])
            for column in columns:
                if column not in adata_obs.index:
                    return raise_Exception(f'An varm result_type can only be retrieved from "X" if every element of the second path element matches a cell name, but {column!r} is not an available cell name!')
            return _as_frame(adata[columns,:].X.T,index=adata_var.index,columns=columns)
        elif key[0] == 'layer':
            if len(key) != 3:
                return raise_Exception(f'An varm result_type can only be retrieved from "layer" with a key with length of exactly 3, but {key!r} was given!')
            if key[1] not in adata_layers:
                return raise_Exception(f'An varm result_type can only be retrieved from "layer" if the second path element matches a layer name, but {key[1]!r} is not an available layer!')
            columns = _as_list(key[2])
            for column in columns:
                if column not in adata_obs.index:
                    return raise_Exception(f'An varm result_type can only be retrieved from "layer" if every element of the third path element matches a cell name, but {column!r} is not an available cell name!')
            return _as_frame(adata[columns,:].layers[key[1]].T,index=adata_var.index,columns=columns)
        elif key[0] == 'var':
            if len(key) != 2:
                return raise_Exception(f'An varm result_type can only be retrieved from "var" with a key with length of exactly 2, but {key!r} was given!')
            columns = _as_list(key[1])
            for column in columns:
                if column not in adata_var.columns:
                    return raise_Exception(f'An varm result_type can only be retrieved from "var" if every element of the second path element matches a var key, but {column!r} is not an available var column!')
            return _as_frame(adata_var[columns],index=adata_var.index,columns=columns)
        elif key[0] == 'varm':
            if len(key) != 2:
                return raise_Exception(f'An varm result_type can only be retrieved from "varm" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_varm:
                return raise_Exception(f'An varm result_type can only be retrieved from "varm" if the second path element matches an varm name, but {key[1]!r} is not an available varm!')
            return _as_frame(adata_varm[key[1]],index=adata_var.index,columns=...)
        else:
            return raise_Exception(f'An varm result_type can only be retrieved from "X","layer","var", and "varm" paths, but {key!r} starts with {key[0]!r}!')

    elif result_type == 'X':
        if key[0] == 'X':
            if not isinstance(adata, ad.AnnData):
                return raise_Exception(no_real_adata_message('X'))
            if len(key) != 1:
                return raise_Exception(f'An X result_type can only be retrieved from "X" with a key with length of exactly 1, but {key!r} was given!')
            return adata.X
        elif key[0] == 'layer':
            if len(key) != 2:
                return raise_Exception(f'An X result_type can only be retrieved from "layer" with a key with length of exactly 2, but {key!r} was given!')
            if key[1] not in adata_layers:
                return raise_Exception(f'An X result_type can only be retrieved from "layer" if the second path element matches a layer name, but {key[1]!r} is not an available layer!')
            return adata_layers[key[1]]
        else:
            return raise_Exception(f'An X result_type can only be retrieved from "X" and "layer", but {key!r} starts with {key[0]!r}!')

    else:
        return raise_Exception(f'The result_type {result_type!r} cannot be interpreted.')


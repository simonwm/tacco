import pandas as pd
import anndata as ad
from ._data import get_data_from_key

def get_positions(
    adata,
    position_key,
):
    """\
    Get an :class:`~pandas.DataFrame` with the positions of the observations.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` with positions annotation in `.obs` or
        `.obsm`. Can also be a :class:`~pandas.DataFrame`, which is then
        treated like the `.obs` of an :class:`~anndata.AnnData`.
    position_key
        The `.obsm` key or array-like of `.obs` keys with the position space
        coordinates. Also supports data paths as specified in
        :func:`~tacco.tl.get_data_from_key`.
    
    Returns
    -------
    A :class:`~pandas.DataFrame` with the positions of the observations.
    
    """
    
    return get_data_from_key(
        adata=adata,
        key=position_key,
        key_name='position_key',
        result_type='obsm',
        search_order=('obsm','multi-obs','obs'),
    )

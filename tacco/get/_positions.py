import pandas as pd
import anndata as ad

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
        coordinates.
        
    Returns
    -------
    A :class:`~pandas.DataFrame` with the positions of the observations.
    
    """
    
    coords = None
    if isinstance(position_key, str):
        if isinstance(adata, ad.AnnData) and position_key in adata.obsm:
            coords = adata.obsm[position_key]
        else:
            position_key = [position_key]
    if coords is None:
        if isinstance(adata, ad.AnnData):
            coords = adata.obs[list(position_key)]
        else:
            coords = adata[list(position_key)]

    if not hasattr(coords, 'columns'):
        coords = pd.DataFrame(coords, index=adata.obs.index)
    
    return coords
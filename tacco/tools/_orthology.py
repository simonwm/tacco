import pandas as pd
import numpy as np
import scipy
import anndata as ad
from ._goa import _get_downloaded_file
from ..utils import row_scale, col_scale, get_sum

_homology_file = None
_gene2DB_matrices = None
_synonym2gene_matrices = None

def setup_orthology_converter(
    homology_file='http://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt',
    working_directory='.',
):
    """\
    Setup for converting gene names between species using MGI homology
    information (Mouse Genome Informatics Web Site,
    `http://www.informatics.jax.org/homology.shtml`).
    
    Parameters
    ----------
    homology_file
        File containing a mapping from species specific gene symbols to common
        MGI "DB class key"/"HomoloGene ID", e.g. downloaded from `http://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt`.
        If this is not available as a local file, it is treated as an URL and
        downloaded to the `working_directory` if necessary, see below.
    working_directory
        Directory where to buffer downloaded files. If a file of the same name
        already exists in this directory, it is not downloaded again.
        
    Returns
    -------
    A string containing the local path to the homology file.
    
    """

    global _homology_file
    
    # download files if necessary
    _homology_file = _get_downloaded_file(homology_file, working_directory)
    
    return _homology_file

def _get_homology_key(hom_df):
    homology_key = 'DB Class Key'
    if homology_key not in hom_df:
        homology_key = 'HomoloGene ID'
    if homology_key not in hom_df:
        raise ValueError(f'Neither "DB Class Key" nor "HomoloGene ID" are available in the file "{_homology_file}"!')
    return homology_key

def _construct_gene2DB_matrix(tax_id):
    
    global _homology_file
    if _homology_file is None:
        print('"run_orthology_converter" was run before "setup_orthology_converter"! Running "setup_orthology_converter" now with default arguments...')
        setup_orthology_converter()
    
    full_hom_df = pd.read_csv(_homology_file, sep='\t')
    
    homology_key = _get_homology_key(full_hom_df)
                         
    full_hom_df[homology_key] = full_hom_df[homology_key].astype('category') # consistent categories across all species
    
    sub_cols = ['Symbol',homology_key,]
    if 'Synonyms' in full_hom_df.columns:
        sub_cols.append('Synonyms')
    
    hom_df = full_hom_df.loc[full_hom_df['NCBI Taxon ID'] == tax_id, sub_cols].copy()
    
    if len(hom_df) == 0:
        raise ValueError(f'The tax_id {tax_id!r} is not available in the file {_homology_file!r}!')
    
    hom_df['Symbol'] = hom_df['Symbol'].astype('category') # consistent categories only within the selected species
    
    # create an anndata containing the mapping from gene symbol synonyms to gene symbols
    if 'Synonyms' not in hom_df.columns:
        hom_df['Synonyms'] = hom_df['Symbol']
    
    hom_df['Synonyms'] = np.where(hom_df['Synonyms'].isna(), hom_df['Symbol'], hom_df['Symbol'].astype(str) + '|' + hom_df['Synonyms'].astype(str))
    synonym_map = hom_df.set_index('Symbol')['Synonyms'].str.split('|').explode().reset_index().drop_duplicates()
    synonym_map['Symbol'] = synonym_map['Symbol'].astype(hom_df['Symbol'].dtype)
    synonym_map['Synonyms'] = synonym_map['Synonyms'].astype('category') # consistent categories only within the selected species
    
    synonym_matrix = scipy.sparse.coo_matrix((np.ones(len(synonym_map),dtype=np.float32),(synonym_map['Synonyms'].cat.codes.to_numpy(),synonym_map['Symbol'].cat.codes.to_numpy())),shape=(len(synonym_map['Synonyms'].cat.categories),len(synonym_map['Symbol'].cat.categories)))
    synonym_matrix = synonym_matrix.tocsr()
    
    synonym_adata = ad.AnnData(synonym_matrix, obs=pd.DataFrame(index=synonym_map['Synonyms'].cat.categories), var=pd.DataFrame(index=synonym_map['Symbol'].cat.categories))
    
    # create an anndata containing the mapping from gene symbols to DB Class Keys/HomoloGene ID
    hom_df = hom_df[hom_df.columns[:2]].drop_duplicates()
    merge_matrix = scipy.sparse.coo_matrix((np.ones(len(hom_df),dtype=np.float32),(hom_df['Symbol'].cat.codes.to_numpy(),hom_df[homology_key].cat.codes.to_numpy())),shape=(len(hom_df['Symbol'].cat.categories),len(hom_df[homology_key].cat.categories)))
    merge_matrix = merge_matrix.tocsr()
    
    merge_adata = ad.AnnData(merge_matrix, obs=pd.DataFrame(index=hom_df['Symbol'].cat.categories), var=pd.DataFrame(index=hom_df[homology_key].cat.categories))
    
    hom_df[homology_key] = hom_df[homology_key].astype(str)
    merge_adata.var[f'{tax_id} orthologs'] = hom_df.groupby(homology_key)['Symbol'].apply(lambda x: list(x))
    for dbck in merge_adata.var[merge_adata.var[f'{tax_id} orthologs'].isna()].index:
        merge_adata.var.loc[dbck,f'{tax_id} orthologs'] = []
    
    return ( merge_adata, synonym_adata )

def _get_gene2DB_matrix(tax_id):
    global _gene2DB_matrices
    if _gene2DB_matrices is None:
        _gene2DB_matrices = {}
    if tax_id not in _gene2DB_matrices:
        _gene2DB_matrices[tax_id] = _construct_gene2DB_matrix(tax_id)
    return _gene2DB_matrices[tax_id]

_name_to_tax_id_map = {
    'human': 9606,
    'mouse': 10090,
}
def _convert_identifier_to_tax_id(identifier):
    if isinstance(identifier, str):
        if identifier in _name_to_tax_id_map:
            identifier = _name_to_tax_id_map[identifier]
        else:
            raise ValueError(f'The identifier {identifier!r} is not supported as an alias of a taxonomy id. The currently supported aliases are:\n{_name_to_tax_id_map}')
    return identifier

def run_orthology_converter(
    adata,
    source_tax_id,
    target_tax_id=None,
    target_gene_symbols=None,
    use_synonyms=True,
):
    """\
    Run orthology conversion between species using MGI homology information
    information (Mouse Genome Informatics Web Site,
    `http://www.informatics.jax.org/homology.shtml`). The function works on the
    var.index of the adata and assumes to find gene symbols there.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` containing the data to convert. If `None`,
        return an :class:`~anndata.AnnData` containing the conversion matrix.
    source_tax_id
        The NCBI taxonomy ID of the source data. Some strings can be used as
        aliases of their taxonomy IDs (e.g. 'human'=9606 and 'mouse'=10090).
    target_tax_id
        The target NCBI taxonomy ID. Some strings can be used as aliases of
        their taxonomy IDs (e.g. 'human'=9606 and 'mouse'=10090). If `None`,
        convert to the common MGI "DB class key"/"HomoloGene ID" (recommended, as this avoids
        extra ambuigities in multi-to-multi mappings).
    target_gene_symbols
        The gene symbols appearing in the target data. Supplying this can make
        a difference if the symbols appearing there are contained in the
        "Synonyms" column of the homology information. Synonyms are
        unambiguously taken care of on the source side.
    use_synonyms
        Whether to use the "Synonyms" column from the homology information.
        
    Returns
    -------
    Returns a :class:`~anndata.AnnData` containing the result of the\
    orthology conversion.
    
    """
    
    source_tax_id = _convert_identifier_to_tax_id(source_tax_id)
    
    merge_matrix, synonym_matrix = _get_gene2DB_matrix(source_tax_id)
    
    # normalize the merge matrix to account for many-to-many mappings
    merge_matrix = merge_matrix.copy() # dont change the original
    row_scale(merge_matrix.X, 1/get_sum(merge_matrix.X, axis=1))
    
    if use_synonyms:
        # normalize the synonym matrix separately as the interpretation differs: uncertain many-to-many mapping of different entities VS many-to-many mapping of identical entities...
        synonym_matrix = synonym_matrix.copy() # dont change the original
        row_scale(synonym_matrix.X, 1/get_sum(synonym_matrix.X, axis=1))
    
        # combine the synonym and the merge matrices on the source side
        merge_matrix = ad.AnnData(synonym_matrix.X @ merge_matrix.X, obs=synonym_matrix.obs, var=merge_matrix.var)
    
    if target_tax_id is not None:
        target_tax_id = _convert_identifier_to_tax_id(target_tax_id)
        target_merge_matrix, target_synonym_matrix = _get_gene2DB_matrix(target_tax_id)
        target_merge_matrix = target_merge_matrix.copy() # dont change the original
        # transposed normalization on the target side, also separate: account for the distribution of counts to genes which are just not appearing in the measurement
        col_scale(target_merge_matrix.X, 1/get_sum(target_merge_matrix.X, axis=0))
        
        if use_synonyms:
            if target_gene_symbols is not None:
                # subset to genes actually appearing
                mapped_genes = target_synonym_matrix.var.index.intersection(target_gene_symbols)
                target_synonym_matrix = target_synonym_matrix[mapped_genes].copy() # dont change the original
            # normalize the synonym matrix after potential subsetting to the synonyms which actually appear in the dataset
            col_scale(target_synonym_matrix.X, 1/get_sum(target_synonym_matrix.X, axis=0))
            target_merge_matrix = ad.AnnData(target_synonym_matrix.X @ target_merge_matrix.X, obs=target_synonym_matrix.obs, var=target_merge_matrix.var)
        else:
            if target_gene_symbols is not None:
                # subset to genes actually appearing
                mapped_genes = target_merge_matrix.obs.index.intersection(target_gene_symbols)
                target_merge_matrix = target_merge_matrix[mapped_genes]
            
        merge_matrix = ad.AnnData(merge_matrix.X @ target_merge_matrix.X.T, obs=merge_matrix.obs, var=target_merge_matrix.obs)
    
    if adata is None:
        return merge_matrix
    
    # subset to genes actually appearing
    mapped_genes = adata.var.index.intersection(merge_matrix.obs.index)
    adata, merge_matrix = adata[:,mapped_genes], merge_matrix[mapped_genes]
    
    return ad.AnnData(adata.X @ merge_matrix.X, obs=adata.obs, obsm=adata.obsm, var=merge_matrix.var)

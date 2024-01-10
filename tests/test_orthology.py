import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def adata_to_convert():
    human_adata = ad.AnnData(scipy.sparse.csr_matrix(np.eye(4)),
        var=pd.DataFrame(index=['ISG15','TP53','GSTM3','GSTM2']),
        dtype=np.float32,
    )
    mouse_adata = ad.AnnData(scipy.sparse.csr_matrix(np.eye(4)),
        var=pd.DataFrame(index=['Isg15','Trp53','Gstm5','Gstm7']),
        dtype=np.float32,
    )
    return {'human':human_adata,'mouse':mouse_adata}

def test_orthology_converter(adata_to_convert):
    human_adata = adata_to_convert['human']
    mouse_adata = adata_to_convert['mouse']
    
    result = tc.tools.run_orthology_converter(adata=human_adata, source_tax_id='human',target_tax_id='mouse', target_gene_symbols=mouse_adata.var.index)
    
    result = result[:,mouse_adata.var.index].copy() # avoid assertion error for comparing differently ordered .var.index
    
    result.X = scipy.sparse.csr_matrix(result.X) # avoid assertion error for comparing a csr matrix view with an actual csr matrix
    
    tc.testing.assert_adata_equal(result, mouse_adata)

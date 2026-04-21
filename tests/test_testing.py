import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="function")
def adata_variants():
    _adatas = []
    adata = ad.AnnData(X=np.array([
        [1,0,0],
        [2,0,0],
        [3,0,0],
        [0,1,0],
        [0,2,0],
        [0,3,0],
        [0,0,1],
        [0,0,2],
        [0,0,3],
    ], dtype=np.float32))
    adata.obs=pd.DataFrame({
        'type': pd.Series([0,0,0,1,1,1,2,2,2,],dtype='category',index=pd.Index(range(adata.shape[0])).astype(str)),
    })
    adata.obsm['X_umap'] = np.array([
            [0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,],
            [0.0,0.0,0.0,0.0,0.0,0.0,2.0,2.0,2.0,],
        ]).T
    _adatas.append(adata)
    return _adatas

#@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.parametrize('dataset', [0,])
def test_assert_adata_equal(adata_variants, dataset):
    adata = adata_variants[dataset]
    copy = adata.copy()

    tc.testing.assert_adata_equal(adata, copy)

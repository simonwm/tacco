import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def adata_with_regions():
    adata = ad.AnnData(scipy.sparse.csr_matrix([
            [1,0],
            [0,1],
            [0,1],
            [0,1],
            [1,0],
            [1,0],
        ], dtype=np.int8),
        obs=pd.DataFrame({
            'x': pd.Series([1,2,3,101,102,103],dtype=float),
            'y': pd.Series([1,2,11,12,101,102],dtype=float),
            'partial': pd.Series([np.nan,np.nan,'0','1','1',np.nan]).astype('category'),
            'ground_truth': pd.Series(['0','0','0','1','1','1']).astype('category'),
        }),
    )
    return adata

def test_find_regions(adata_with_regions):
    adata = adata_with_regions.copy() # dont change the input

    ground_truth = adata.obs['ground_truth'].copy()
    ground_truth.name = 'new_regions'

    result = tc.tl.find_regions(adata, key_added='new_regions')

    tc.testing.assert_series_equal(result.obs['new_regions'], ground_truth, rtol=1e-14, atol=1e-50)

@pytest.mark.parametrize('input_type', ['adata','df'])
@pytest.mark.parametrize('result_key', ['new','partial',None,...])
@pytest.mark.parametrize('k', [1,2]) # for k=2 the example relies on the ordering of input data to get the correct majority vote result
def test_fill_regions(adata_with_regions, input_type, result_key, k):
    adata = adata_with_regions.copy() # dont change the input

    ground_truth = adata.obs['ground_truth'].copy()

    input_data = adata.obs if input_type == 'df' else adata
    
    result_data = tc.tl.fill_regions(input_data, region_key='partial', result_key=result_key, k=k)
    
    if result_key is None:
        result = result_data
        ground_truth.name = 'partial'
    else:
        result_df = result_data if input_type == 'df' else result_data.obs
        if result_key is ...:
            result_key = 'partial'
        ground_truth.name = result_key
        result = result_df[result_key]

    tc.testing.assert_series_equal(result, ground_truth, rtol=1e-14, atol=1e-50)

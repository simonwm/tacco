import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc

@pytest.fixture(scope="session")
def adata_and_subsampling():
    adata = ad.AnnData(X=np.array([
        [1,0,0],
        [2,0,0],
        [0,3,0],
        [0,4,0],
        [0,0,5],
        [0,0,6],
    ]))
    adata.obs['type'] = pd.Series(['a','a','ab','ab','bb','bb'],dtype='category',index=adata.obs.index)
    subsampling = {'a':2.0,'ab':1.0,'b':0.5}
    sub_adata = ad.AnnData(X=np.array([
        [2,0,0],
        [1,0,0],
        [2,0,0],
        [1,0,0],
        [0,4,0],
        [0,3,0],
        [0,0,6],
    ]))
    sub_adata.obs=pd.DataFrame({
        'type': pd.Series(['a','a','a','a','ab','ab','bb'],dtype='category',index=sub_adata.obs.index)
    })
    sub_adata.obs.index = pd.Index([1,0,1,0,3,2,5]).astype(str) # fix the order: random, but deterministic for given seed
    return ( adata, subsampling, sub_adata )

def test_subsample_annotation(adata_and_subsampling):
    adata, subsampling, sub_adata = adata_and_subsampling
    
    result = tc.pp.subsample_annotation(adata, modification=subsampling)
    
    tc.testing.assert_adata_equal(sub_adata,result)

@pytest.fixture(scope="session")
def adata_and_platform_effect():
    adata = ad.AnnData(X=np.array([
        [1,0,0], # need one gene overlap between pairs of types to fix relative factors
        [2,0,1],
        [0,3,0],
        [0,4,0],
        [0,0,5],
        [0,2,6],
        [1,1,1],
        [1,1,1],
        [1,1,1],
    ]))
    adata.obs['type'] = pd.Series(['a','a','b','b','c','c','d','d','d'],dtype='category',index=adata.obs.index)
    adata.obsm['type_obsm'] = pd.get_dummies(adata.obs['type'])
    platform_adata = ad.AnnData(X=np.array([ # fix the expression: random, but deterministic for given seed
        [ 2.1717718, 0.       ,  0.       ],
        [ 4.3435435, 0.       ,  3.4343598],
        [ 0.       , 2.6412125,  0.       ],
        [ 0.       , 3.5216165,  0.       ],
        [ 0.       , 0.       , 17.171799 ],
        [ 0.       , 1.7608083, 20.60616  ],
        [ 2.1717718, 0.8804041,  3.4343598],
        [ 2.1717718, 0.8804041,  3.4343598],
        [ 2.1717718, 0.8804041,  3.4343598],
    ]))
    platform_adata.obs['type'] = pd.Series(['a','a','b','b','c','c','d','d','d'],dtype='category',index=platform_adata.obs.index)
    platform_adata.obsm['type_obsm'] = pd.get_dummies(platform_adata.obs['type'])
    return ( adata, platform_adata )

def test_random_platform_effect(adata_and_platform_effect):
    adata, platform_adata = adata_and_platform_effect
    
    result = adata.copy() # inplace operation:
    tc.pp.apply_random_platform_effect(result, round=False)
    
    tc.testing.assert_adata_equal(result,platform_adata)
    
    result = adata.copy() # inplace operation:
    tc.pp.apply_random_platform_effect(result, round=True)
    
    rounded_platform_adata = platform_adata.copy()
    np.around(rounded_platform_adata.X, out=rounded_platform_adata.X)
    
    tc.testing.assert_adata_equal(result,rounded_platform_adata)

@pytest.mark.parametrize('anno', [None,'type','type_obsm'])
@pytest.mark.parametrize('other_anno', [None,'type','type_obsm'])
def test_normalize_platform(adata_and_platform_effect, anno, other_anno):
    adata, platform_adata = adata_and_platform_effect
    
    try:
        result = tc.pp.normalize_platform(adata=platform_adata, reference=adata, annotation_key=anno, reference_annotation_key=other_anno, inplace=False)
    except ValueError as e:
        if anno == 'type_obsm' and other_anno == anno:
            return # This exception is wanted behaviour
        else:
            raise e
    if anno == 'type_obsm' and other_anno == anno:
        assert(0) # This case should have thrown an exception
    
    result.X *= adata.X.sum() / result.X.sum() # can only normalize up to an overall factor
    
    tc.testing.assert_adata_equal(result,adata)

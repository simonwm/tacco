import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse
import tacco as tc

@pytest.fixture(scope="session")
def adata_and_split():
    adata = ad.AnnData(X=np.array([
        [1,0,0,0],
        [0,1,0,1],
        [0,0,1,3],
        [1,1,0,1],
        [1,0,1,3],
        [0,1,1,4],
        [0,1,1,5], # extra count cannot be divided equally between all contributing types
    ]))
    adata.varm['type']=pd.DataFrame(np.array([
        [1,0  ,0   ,],
        [0,0.5,0   ,],
        [0,0  ,0.25,],
        [0,0.5,0.75,],
    ]),index=adata.var.index, columns=pd.Index([0,1,2],dtype='category'))
    adata.obsm['type']=pd.DataFrame(np.array([
        [1  ,0  ,0  ],
        [0  ,1  ,0  ],
        [0  ,0  ,1  ],
        [1/3,2/3,0  ],
        [1/5,0  ,4/5],
        [0  ,2/6,4/6],
        [0  ,2/6,4/6],
    ]),index=adata.obs.index,columns=adata.varm['type'].columns)
    adata.uns['type'] = pd.Series([3,3,4],index=pd.Index([0,1,2],dtype='category'),name='type2').astype('category')
    
    sdata = ad.AnnData(X=scipy.sparse.csr_matrix(np.array([ # part of all count placement is random due to regularization in the split - but only slightly
        [1,0,0,0],
        [1,0,0,0],
        [1,0,0,0],
        [0,1,0,1],
        [0,1,0,1],
        [0,1,0,1],
        [0,1,0,1],
        [0,0,1,3],
        [0,0,1,3],
        [0,0,1,3],
        [0,0,1,4], # the placement of the extra count is random and frozen for fixed seed
    ])))
    sdata.obs=pd.DataFrame({
        'index': pd.Series([0,3,4,1,3,5,6,2,4,5,6,],index=sdata.obs.index).astype(str),
        'type':  pd.Series([0,0,0,1,1,1,1,2,2,2,2,],dtype='category',index=sdata.obs.index),
    })
    sdata.varm['type']=adata.varm['type']
    
    sdata2 = ad.AnnData(X=scipy.sparse.csr_matrix(np.array([ # part of all count placement is random due to regularization in the split - but only slightly
        [1,0,0,0],
        [0,1,0,1],
        [1,1,0,1],
        [1,0,0,0],
        [0,1,0,1],
        [0,1,0,1],
        [0,0,1,3],
        [0,0,1,3],
        [0,0,1,3],
        [0,0,1,4], # the placement of the extra count is random and frozen for fixed seed
    ])))
    sdata2.obs=pd.DataFrame({
        'index': pd.Series([0,1,3,4,5,6,2,4,5,6,],index=sdata2.obs.index).astype(str),
        'type2':  pd.Series([3,3,3,3,3,3,4,4,4,4,],dtype='category',index=sdata2.obs.index),
    })
    sdata2.varm['type']=adata.varm['type']
    
    ddata = ad.AnnData(X=np.array([
        [ 1 , 0 , 0 , 0 ],
        [ 1 , 0 , 0 , 0 ],
        [ 1 , 0 , 0 , 0 ],
        [ 0 , 1 , 0 , 1 ],
        [ 0 , 1 , 0 , 1 ],
        [ 0 , 1 , 0 , 1 ],
        [ 0 ,7/6, 0 ,7/6],
        [ 0 , 0 , 1 , 3 ],
        [ 0 , 0 , 1 , 3 ],
        [ 0 , 0 , 1 , 3 ],
        [ 0 , 0 ,7/6,7/2],
    ]))
    ddata.obs=pd.DataFrame({
        'index': pd.Series([0,3,4,1,3,5,6,2,4,5,6,],index=sdata.obs.index).astype(str),
        'type':  pd.Series([0,0,0,1,1,1,1,2,2,2,2,],dtype='category',index=sdata.obs.index),
    })
    ddata.varm['type']=adata.varm['type']
    
    ddata2 = ad.AnnData(X=np.array([
        [ 1 , 0 , 0 , 0 ],
        [ 0 , 1 , 0 , 1 ],
        [ 1 , 1 , 0 , 1 ],
        [ 1 , 0 , 0 , 0 ],
        [ 0 , 1 , 0 , 1 ],
        [ 0 ,7/6, 0 ,7/6],
        [ 0 , 0 , 1 , 3 ],
        [ 0 , 0 , 1 , 3 ],
        [ 0 , 0 , 1 , 3 ],
        [ 0 , 0 ,7/6,7/2],
    ]))
    ddata2.obs=pd.DataFrame({
        'index': pd.Series([0,1,3,4,5,6,2,4,5,6,],index=ddata2.obs.index).astype(str),
        'type2':  pd.Series([3,3,3,3,3,3,4,4,4,4,],dtype='category',index=ddata2.obs.index),
    })
    ddata2.varm['type']=adata.varm['type']
    
    bdata = adata.copy()
    bdata.X = np.array([
        [ 1 , 0 , 0 ,  0 ],
        [ 0 , 1 , 0 ,  1 ],
        [ 0 , 0 , 1 ,  3 ],
        [ 1 , 1 , 0 ,  1 ],
        [ 1 , 0 , 1 ,  3 ],
        [ 0 , 1 , 1 ,  4 ],
        [ 0 ,7/6,7/6,14/3],
    ], dtype=np.float32)
    bdata.obs['index'] = bdata.obs.index.astype(str)
    del bdata.uns['type']
    del bdata.obsm['type']
    
    return ( adata, {'exact':sdata, 'denoise':ddata, 'exact2':sdata2, 'denoise2':ddata2, 'bulk':bdata} )

@pytest.mark.parametrize('mode', ['exact','denoise','bulk'])
@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('map_all_genes', [False, True])
def test_split(adata_and_split,mode,mapped,map_all_genes):
    adata, sdatas = adata_and_split
    mode_sel = mode if not mapped or mode == 'bulk' else f'{mode}2'
    sdata = sdatas[mode_sel]
    
    adata = adata.copy() # dont change the input
    if not mapped:
        del adata.uns['type']
    
    result = tc.tl.split_observations(adata, 'type', mode=mode, seed=(42 if mode == 'exact' else None), map_all_genes=map_all_genes)
    
    tc.testing.assert_adata_equal(result, sdata, rtol=1e-14, atol=1e-14)


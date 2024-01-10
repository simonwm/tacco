import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def adata_with_data():
    adata = ad.AnnData(X=np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
    ]),dtype=np.float32, obs=pd.DataFrame(index=['A','T','C','G']))
    adata.layers['sparse'] = scipy.sparse.csr_matrix(adata.X)
    adata.layers['dense'] = adata.X
    adata.var['type']=pd.Series(['a','b','c'],index=adata.var.index,name='type')
    adata.var['blub']=pd.Series(['1','2','3'],index=adata.var.index,name='blub')
    adata.varm['type']=pd.DataFrame(np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ]),index=adata.var.index,columns=['A','B','C'])
    adata.obs['type']=pd.Series(['A','B','C','AB'],index=adata.obs.index,name='type')
    adata.obs['blub']=pd.Series(['1','2','3','12'],index=adata.obs.index,name='blub')
    adata.obsm['type']=pd.DataFrame(np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0.5,0.5,0],
    ]),index=adata.obs.index,columns=['A','B','C'])
    adata.obsm['array']=np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0.5,0.5,0],
    ])
    adata.raw = adata.copy()
    return adata

def test_get_data_from_key(adata_with_data):
    data = adata_with_data.obs['type']
    result = tc.get.data_from_key(adata_with_data, 'type')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_obs(adata_with_data):
    data = adata_with_data.obs['type']
    result = tc.get.data_from_key(adata_with_data, 'type', result_type='obs')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_var(adata_with_data):
    data = adata_with_data.var['type']
    result = tc.get.data_from_key(adata_with_data, 'type', result_type='var')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_obsm(adata_with_data):
    data = pd.DataFrame(adata_with_data.obs['type'])
    result = tc.get.data_from_key(adata_with_data, 'type', result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm(adata_with_data):
    data = pd.DataFrame(adata_with_data.var['type'])
    result = tc.get.data_from_key(adata_with_data, 'type', result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_obsm__path(adata_with_data):
    data = adata_with_data.obsm['type']
    result = tc.get.data_from_key(adata_with_data, ('obsm','type'), result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm__path(adata_with_data):
    data = adata_with_data.varm['type']
    result = tc.get.data_from_key(adata_with_data, ('varm','type'), result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_X__path(adata_with_data):
    data = adata_with_data.X
    result = tc.get.data_from_key(adata_with_data, ('X',), result_type='X')
    tc.testing.assert_dense_equal(result, data)

def test_get_data_from_key__result_type_X__path_layer(adata_with_data):
    data = adata_with_data.layers['sparse']
    result = tc.get.data_from_key(adata_with_data, ('layer','sparse'), result_type='X')
    tc.testing.assert_sparse_equal(result, data)

def test_get_data_from_key__result_type_obs__path_layer_sparse(adata_with_data):
    data = pd.Series(adata_with_data[:,'1'].layers['sparse'].toarray().flatten(), index=adata_with_data.obs.index, name='1')
    result = tc.get.data_from_key(adata_with_data, ('layer','sparse','1'), result_type='obs')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_obs__path_layer_dense(adata_with_data):
    data = pd.Series(adata_with_data[:,'1'].layers['dense'].flatten(), index=adata_with_data.obs.index, name='1')
    result = tc.get.data_from_key(adata_with_data, ('layer','dense','1'), result_type='obs')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_obsm__path_layer_sparse(adata_with_data):
    data = pd.DataFrame(adata_with_data[:,['1','2']].layers['sparse'].toarray(), index=adata_with_data.obs.index, columns=['1','2'])
    result = tc.get.data_from_key(adata_with_data, ('layer','sparse',['1','2']), result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_obsm__path_layer_dense(adata_with_data):
    data = pd.DataFrame(adata_with_data[:,['1','2']].layers['dense'], index=adata_with_data.obs.index, columns=['1','2'])
    result = tc.get.data_from_key(adata_with_data, ('layer','dense',['1','2']), result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_var__path_layer_sparse(adata_with_data):
    data = pd.Series(adata_with_data['T',:].layers['sparse'].toarray().flatten(), index=adata_with_data.var.index, name='T')
    result = tc.get.data_from_key(adata_with_data, ('layer','sparse','T'), result_type='var')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_var__path_layer_dense(adata_with_data):
    data = pd.Series(adata_with_data['T',:].layers['dense'].flatten(), index=adata_with_data.var.index, name='T')
    result = tc.get.data_from_key(adata_with_data, ('layer','dense','T'), result_type='var')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_varm__path_layer_sparse(adata_with_data):
    data = pd.DataFrame(adata_with_data[['T','G'],:].layers['sparse'].toarray().T, index=adata_with_data.var.index, columns=['T','G'])
    result = tc.get.data_from_key(adata_with_data, ('layer','sparse',['T','G']), result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm__path_layer_dense(adata_with_data):
    data = pd.DataFrame(adata_with_data[['T','G'],:].layers['dense'].T, index=adata_with_data.var.index, columns=['T','G'])
    result = tc.get.data_from_key(adata_with_data, ('layer','dense',['T','G']), result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_obsm__path_obs(adata_with_data):
    data = adata_with_data.obs[['blub','type']]
    result = tc.get.data_from_key(adata_with_data, ('obs',['blub','type']), result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm__path_var(adata_with_data):
    data = adata_with_data.var[['blub','type']]
    result = tc.get.data_from_key(adata_with_data, ('var',['blub','type']), result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm__path_raw_var(adata_with_data):
    data = adata_with_data.raw.to_adata().var[['blub','type']]
    result = tc.get.data_from_key(adata_with_data, ('raw','var',['blub','type']), result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_obsm__array(adata_with_data):
    data = pd.DataFrame(adata_with_data.obsm['array'], index=adata_with_data.obs.index)
    result = tc.get.data_from_key(adata_with_data, 'array', result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__direct_frame_obs(adata_with_data):
    data = adata_with_data.obs['type']
    result = tc.get.data_from_key(adata_with_data.obs, 'type')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_obs__direct_frame_obs(adata_with_data):
    data = adata_with_data.obs['type']
    result = tc.get.data_from_key(adata_with_data.obs, 'type', result_type='obs')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_var__direct_frame_var(adata_with_data):
    data = adata_with_data.var['type']
    result = tc.get.data_from_key(adata_with_data.var, 'type', result_type='var')
    tc.testing.assert_series_equal(result, data)

def test_get_data_from_key__result_type_obsm__direct_frame_obs(adata_with_data):
    data = pd.DataFrame(adata_with_data.obs['type'])
    result = tc.get.data_from_key(adata_with_data.obs, 'type', result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm__direct_frame_var(adata_with_data):
    data = pd.DataFrame(adata_with_data.var['type'])
    result = tc.get.data_from_key(adata_with_data.var, 'type', result_type='varm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_obsm__path_obs__direct_frame_obs(adata_with_data):
    data = adata_with_data.obs[['blub','type']]
    result = tc.get.data_from_key(adata_with_data.obs, ('obs',['blub','type']), result_type='obsm')
    tc.testing.assert_frame_equal(result, data)

def test_get_data_from_key__result_type_varm__path_var__direct_frame_var(adata_with_data):
    data = adata_with_data.var[['blub','type']]
    result = tc.get.data_from_key(adata_with_data.var, ('var',['blub','type']), result_type='varm')
    tc.testing.assert_frame_equal(result, data)


def test_get_positions(adata_with_data):
    data = adata_with_data.obsm['type']
    result = tc.get.positions(adata_with_data, 'type')
    tc.testing.assert_frame_equal(result, data)
    
def test_get_positions__array(adata_with_data):
    data = pd.DataFrame(adata_with_data.obsm['array'], index=adata_with_data.obs.index)
    result = tc.get.positions(adata_with_data, 'array')
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__single_obs(adata_with_data):
    data = pd.DataFrame(adata_with_data.obs['blub'])
    result = tc.get.positions(adata_with_data, 'blub')
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__multiple_obs(adata_with_data):
    data = adata_with_data.obs[['blub','type']]
    result = tc.get.positions(adata_with_data, ['blub','type'])
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__multiple_obs__tuple(adata_with_data):
    data = adata_with_data.obs[['blub','type']]
    result = tc.get.positions(adata_with_data, ('blub','type'))
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__single_obs__direct_frame_obs(adata_with_data):
    data = pd.DataFrame(adata_with_data.obs['blub'])
    result = tc.get.positions(adata_with_data.obs, 'blub')
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__multiple_obs__direct_frame_obs(adata_with_data):
    data = adata_with_data.obs[['blub','type']]
    result = tc.get.positions(adata_with_data.obs, ['blub','type'])
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__path_single_obs(adata_with_data):
    data = pd.DataFrame(adata_with_data.obs['type'])
    result = tc.get.positions(adata_with_data, ('obs','type'))
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__path_multiple_obs(adata_with_data):
    data = adata_with_data.obs[['blub','type']]
    result = tc.get.positions(adata_with_data, ('obs',['blub','type']))
    tc.testing.assert_frame_equal(result, data)

def test_get_positions__path_layer_sparse(adata_with_data):
    data = pd.DataFrame(adata_with_data[:,['1','2']].layers['sparse'].toarray(), index=adata_with_data.obs.index, columns=['1','2'])
    result = tc.get.positions(adata_with_data, ('layer','sparse',['1','2']))
    tc.testing.assert_frame_equal(result, data)

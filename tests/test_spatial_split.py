import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def spatial_adata_to_split():
    points = pd.DataFrame({
        'x':               pd.Series([     0,     0,    10,    10,    20,    20,    30,    30,    40,    40,    50,    50],dtype=float),
        'y':               pd.Series([     0,     1,     0,     1,     0,     1,     0,     1,     0,     1,     0,     1],dtype=float),
        'split_0_N_N':     pd.Series([  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|0']).astype('category'),
        'split_0_x_2':     pd.Series([  '|0',  '|0',  '|0',  '|0',  '|0',  '|0',  '|1',  '|1',  '|1',  '|1',  '|1',  '|1']).astype('category'),
        'split_0_x_3':     pd.Series([  '|0',  '|0',  '|0',  '|0',  '|1',  '|1',  '|1',  '|1',  '|2',  '|2',  '|2',  '|2']).astype('category'),
        'split_0_x_6':     pd.Series([  '|0',  '|0',  '|1',  '|1',  '|2',  '|2',  '|3',  '|3',  '|4',  '|4',  '|5',  '|5']).astype('category'),
        'split_0_y_2':     pd.Series([  '|0',  '|1',  '|0',  '|1',  '|0',  '|1',  '|0',  '|1',  '|0',  '|1',  '|0',  '|1']).astype('category'),
        'split_0_x_6_y_2': pd.Series(['|0|0','|0|1','|1|0','|1|1','|2|0','|2|1','|3|0','|3|1','|4|0','|4|1','|5|0','|5|1']).astype('category'),
        'split_25_x_2':    pd.Series([  '|0',  '|0',  '|0',  '|0',  None,  None,  None,  None,  '|1',  '|1',  '|1',  '|1']).astype('category'),
        'split_45_x_2':    pd.Series([  '|0',  '|0',  None,  None,  None,  None,  None,  None,  None,  None,  '|1',  '|1']).astype('category'),
    })
    adata = ad.AnnData(np.zeros((len(points),0), dtype=np.int8),
        obs=points,
        dtype=np.int8,
    )
    return adata

@pytest.mark.parametrize('label_thickness_direction_scheme', [
    ('split_0_N_N',0,None,1),
    ('split_0_N_N',0,'x',1),
    ('split_0_N_N',0,'y',1),
    ('split_0_x_2',0,None,2),
    ('split_0_x_2',0,'x',2),
    ('split_0_x_3',0,None,3),
    ('split_0_x_3',0,'x',3),
    ('split_0_x_6',0,None,6),
    ('split_0_x_6',0,'x',6),
    ('split_0_y_2',0,'y',2),
    ('split_0_x_6_y_2',0,('x','y'),(6,2)),
    ('split_0_x_6_y_2',0,None,(6,2)),
    
    ('split_0_N_N',5,None,1),
    ('split_0_N_N',5,'x',1),
    ('split_0_x_2',5,None,2),
    ('split_0_x_2',5,'x',2),
    ('split_0_x_3',5,None,3),
    ('split_0_x_3',5,'x',3),
    ('split_0_x_6',5,None,6),
    ('split_0_x_6',5,'x',6),
    
    ('split_25_x_2',25,'x',2),
    ('split_45_x_2',45,'x',2),
])
def test_distance_matrix(spatial_adata_to_split, label_thickness_direction_scheme):
    adata = spatial_adata_to_split
    
    label, thickness, direction, scheme = label_thickness_direction_scheme
    
    result = tc.utils.split_spatial_samples(adata, buffer_thickness=thickness, split_direction=direction, split_scheme=scheme)
    
    result.name = label
    tc.testing.assert_series_equal(result, adata.obs[label])

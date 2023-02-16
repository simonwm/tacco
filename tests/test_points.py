import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def points_binsize_adata_counts_reference():
    binsize = 10
    points = pd.DataFrame({
        'x': pd.Series([1,2,3,101,102,103],dtype=float),
        'y': pd.Series([1,2,11,12,101,102],dtype=float),
        'bins_x': pd.Series([0,0,0,10,10,10],dtype=np.int8),
        'bins_y': pd.Series([0,0,1,1,10,10],dtype=np.int8),
        'hash': pd.Series([0,0,1,2,3,3],dtype=np.int8),
        'gene': pd.Series([0,0,1,0,1,2]).astype('category'),
        'type': pd.Series([0,0,1,0,1,1]).astype('category'),
        'segment': pd.Series([0,0,1,2,3,3]).astype('category'),
    })
    reference = ad.AnnData(scipy.sparse.csr_matrix([
            [10,0,0],
            [0,20,10],
        ], dtype=np.int8),
        dtype=np.int8,
        obs=pd.DataFrame({'type': pd.Series([0,1]).astype('category')}),
    )
    adata = ad.AnnData(scipy.sparse.csr_matrix([
            [2,0,0],
            [0,1,0],
            [1,0,0],
            [0,1,1],
        ], dtype=np.int8),
        obs=pd.DataFrame(index=points['hash'].unique().astype(str)),
        var=pd.DataFrame(index=points['gene'].unique().astype(str)),
        dtype=np.int8,
    )
    adata.obs.index.name='hash'
    adata.var.index.name='gene'
    counts = pd.DataFrame({
        'hash': pd.Series([0,1,2,3,3]).astype(np.int8),
        'gene': pd.Series([0,1,0,1,2]).astype(str),
        'X':    pd.Series([2,1,1,1,1]).astype(np.int8),
        'type': pd.Series([0,1,0,1,1]).astype('category'),
    })
    return ( points, binsize, adata, counts, reference, )

@pytest.fixture(scope="session")
def adata_distance():
    adata = ad.AnnData(scipy.sparse.csr_matrix([
        [1,0,0,2],
        [0,1,0,2],
        [0,0,1,0],
    ], dtype=float),dtype=float)
    adata.obs['x'] = [0,3,0]
    adata.obs['y'] = [0,0,4]
    adata.obs['c'] = pd.Series([0,0,1], index=adata.obs.index).astype('category')
    adata.obsp['distance'] = scipy.sparse.csr_matrix([
        [0,3,4],
        [3,0,5],
        [4,5,0],
    ], dtype=float)
    adata.obsp['distance_c'] = scipy.sparse.csr_matrix([
        [0,3,0],
        [3,0,0],
        [0,0,0],
    ], dtype=float)
    adata.obsp['distance_c_3_5'] = scipy.sparse.csr_matrix([
        [0,3,5],
        [3,0,0],
        [5,0,0],
    ], dtype=float)
    adata.obsp['affinity'] = scipy.sparse.csr_matrix([
        [0.                 , 0.32465246735834974, 0.1353352832366127 ],
        [0.32465246735834974, 0.                 , 0.04393693362340742],
        [0.1353352832366127 , 0.04393693362340742, 0.                 ],
    ], dtype=float)
    adata.obs['s'] = pd.Series([0,0,1], index=adata.obs.index).astype('category')
    return adata

@pytest.fixture(scope="session")
def points_clusters():
    mean = np.array([[0,0],[1,0],[0,1],[1,1]])
    number = [50,80,60,100]
    rng = np.random.default_rng(seed=42)
    X = []
    y = []
    for l,(m,n) in enumerate(zip(mean,number)):
        X.append(rng.uniform(size=(n,2))*0.8+m)
        y.append(np.full(n,l))
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, pd.Series(y).astype('category')

def test_dataframe2anndata(points_binsize_adata_counts_reference):
    points, binsize, adata, counts, reference = points_binsize_adata_counts_reference
    ref = points[['hash','gene']]

    result = tc.utils.dataframe2anndata(points, obs_key='hash', var_key='gene')

    tc.testing.assert_sparse_equal(result.X, adata.X)
    tc.testing.assert_frame_equal(result.obs, adata.obs, rtol=1e-14, atol=1e-50)
    tc.testing.assert_frame_equal(result.var, adata.var, rtol=1e-14, atol=1e-50)

def test_anndata2dataframe(points_binsize_adata_counts_reference):
    points, binsize, adata, counts, reference = points_binsize_adata_counts_reference
    
    counts = counts[['hash','gene','X']].copy()
    counts['hash'] = counts['hash'].astype(str) # anndatas always have string indices...

    result = tc.utils.anndata2dataframe(adata)

    tc.testing.assert_frame_equal(result, counts[['hash','gene','X']], rtol=1e-14, atol=1e-50)

def test_map_hash_annotation(points_binsize_adata_counts_reference):
    points, binsize, adata, counts, reference = points_binsize_adata_counts_reference
    types = points['type']

    result = tc.tl._points._map_hash_annotation(points, counts, 'type')

    tc.testing.assert_series_equal(result, types, rtol=1e-14, atol=1e-50)

def test_mode():
    
    df = pd.DataFrame([
        ['a','b','c'], # the mode here is random - but given a fixed seed deterministic
        ['b','b','b'],
        ['a','a','d'],
        [np.nan,'d','d'],
        [np.nan,np.nan,'a'],
        [np.nan,np.nan,np.nan,]
    ])
    
    result = tc.utils.mode(df)

    tc.testing.assert_series_equal(result, pd.Series(['c','b','a','d','a',np.nan]))

def test_annotate_single_molecules(points_binsize_adata_counts_reference):
    points, binsize, adata, counts, reference = points_binsize_adata_counts_reference

    result = tc.tl.annotate_single_molecules(points, reference, annotation_key='type', method='projection')

    tc.testing.assert_series_equal(result, points['type'], rtol=1e-14, atol=1e-50)

@pytest.mark.filterwarnings("ignore:scipy.rand is deprecated")
def test_segment(points_clusters):
    X,y = points_clusters
    points = pd.DataFrame({'x':X[:,0],'y':X[:,1]})

    result = tc.tl.segment(points, distance_scale=0.1, max_distance=0.3, max_size=1000, min_size=100, position_scale=None)

    tc.testing.assert_series_equal(result, y, rtol=1e-14, atol=1e-50)

@pytest.mark.parametrize('low_mem', [False, True])
def test_distance_matrix(adata_distance, low_mem):
    adata = adata_distance
    
    distance = adata.obsp['distance']

    result = tc.tl.distance_matrix(adata, 10, low_mem=low_mem)
    
    tc.testing.assert_sparse_equal(result, distance)

    distance = adata.obsp['distance_c']
    
    result = tc.tl.distance_matrix(adata, 10, annotation_key='c', low_mem=low_mem)
    
    tc.testing.assert_sparse_equal(result, distance)

    distance = adata.obsp['distance_c_3_5']

    result = tc.tl.distance_matrix(adata, 5, annotation_key='c', annotation_distance=3, low_mem=low_mem)

    tc.testing.assert_sparse_equal(result, distance)

    distance = adata.obsp['distance_c_3_5']

    result = tc.tl.distance_matrix(adata, 5, annotation_key='c', annotation_distance='cosine', annotation_distance_scale=3, low_mem=low_mem)

    tc.testing.assert_sparse_equal(result, distance)
    
    distance = adata.obsp['distance']

    result = tc.tl.distance_matrix(adata, None, low_mem=low_mem)
    
    tc.testing.assert_dense_equal(result, distance.A)

    distance = adata.obsp['distance_c']
    
    result = tc.tl.distance_matrix(adata, None, annotation_key='c', low_mem=low_mem)
    
    tc.testing.assert_dense_equal(result, distance.A)

def test_affinity(adata_distance):
    adata = adata_distance
    affinity = adata.obsp['affinity']

    result = tc.tl.affinity(adata, 2.0, 'distance')

    tc.testing.assert_sparse_equal(result, affinity)

def test_spectral_clustering_adata(adata_distance):
    adata = adata_distance
    cluster = adata.obs['s'].copy()
    cluster.name = None

    result = tc.tl.spectral_clustering(adata, 2, affinity_key='affinity')

    tc.testing.assert_series_equal(result, cluster)

@pytest.mark.filterwarnings("ignore:scipy.rand is deprecated")
def test_spectral_clustering(points_clusters):
    X,y = points_clusters
    
    dist = tc.utils.sparse_distance_matrix(X, 0.3)
    aff = tc.tl.affinity(dist, 0.1)
    cluster = tc.tl.spectral_clustering(aff, max_size=1000, min_size=100, verbose=4)

    tc.testing.assert_series_equal(cluster, y)


def test_distribute_molecules():
    
    adata = ad.AnnData(scipy.sparse.csr_matrix([
        [1,0,0,2],
        [0,1,0,2],
        [0,0,1,0],
    ], dtype=float),dtype=float)
    adata.obs['x'] = [0,3,0]
    adata.obs['y'] = [0,0,4]
    
    molecules0 = pd.DataFrame({
        'x':[
            0.,0.,0.,3.,3.,3.,0.,
            ],
        'y':[
            0.,0.,0.,0.,0.,0.,4.,
            ],
        'gene':pd.Series([
            '0','3','3','1','3','3','2',
            ]).astype('category'),
        'index':pd.Series([
            '0','0','0','1','1','1','2',
        ]).astype('category')
    })
    
    result = tc.tl.distribute_molecules(adata, width=0)
    
    tc.testing.assert_frame_equal(result, molecules0)
    
    molecules1 = molecules0.copy()
    molecules1['x'] = [
         0.049671, 0.064769,-0.023415, 3.157921, 2.953053, 2.953658, 0.024196,
    ]
    molecules1['y'] = [
        -0.013826, 0.152303,-0.023414, 0.076743, 0.054256,-0.046573, 3.808672,
    ]
    result = tc.tl.distribute_molecules(adata, width=.1)

    tc.testing.assert_frame_equal(result, molecules1, atol=0.1)

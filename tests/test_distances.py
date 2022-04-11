import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse
import scipy.stats
import scipy.spatial.distance

@pytest.fixture(scope="session")
def points_distances():
    t = np.sqrt(2)
    f = np.sqrt(5)
    return {
    'points': np.array([
        [0,0,1,0],
        [2,0,0,0],
        [0,1,0,0],
    ]),
    'euclidean': np.array([
        [0,f,t],
        [f,0,f],
        [t,f,0],
    ],dtype=float),
    'cosine': np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ],dtype=float),
    'projection': np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ],dtype=float),
    'bc': np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ],dtype=float),
    'bc2': np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ],dtype=float),
    'bp': np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ],dtype=float),
    'bpd': np.array([
        [0  ,1.5,1.5],
        [1.5,0  ,1.5],
        [1.5,1.5,0  ],
    ],dtype=float),
    'hellinger': np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0],
    ],dtype=float),
    }

@pytest.fixture(scope="session")
def counts_distances():
    nCellsA = 4
    nCellsB = 6
    nGenes = 11
    meanCountsPerGeneAndCell = 15
    
    np.random.seed(42)
    rvs = scipy.stats.poisson(meanCountsPerGeneAndCell).rvs

    countsA = scipy.sparse.random(nCellsA, nGenes, density=0.8, random_state=42, data_rvs=rvs).tocsr()
    countsB = scipy.sparse.random(nCellsB, nGenes, density=0.5, random_state=42, data_rvs=rvs).tocsr()
    
    return {
    'countsA': countsA,
    'countsB': countsB,
    'euclidean': scipy.spatial.distance.cdist(countsA.A,countsB.A,metric='euclidean'),
    'cosine': scipy.spatial.distance.cdist(countsA.A,countsB.A,metric='cosine'),
    # these distances are just the frozen results of a reference run - no guarante...
    'projection': np.array([[0.9033069734004313, 0.9339160839160839, 0.9043336944745395, 0.9343117408906882, 0.9029585798816568, 0.8867565424266455],
                            [0.9027345102111457, 0.9118967452300786, 0.9061032863849765, 0.9166666666666666, 0.9112739112739112, 0.9102710958381062],
                            [0.8966211358734724, 1.                , 0.8819068255687974, 0.8874493927125506, 0.8896027049873203, 0.8951625693893734],
                            [0.8968624833110814, 0.8863636363636364, 0.9366197183098591, 0.9192669172932331, 0.9043171114599686, 0.907879234167894 ]]),
    'bc': np.array([[0.2471076136646242 , 0.555153369587005  , 0.38305350506881186, 0.5587100878629923 , 0.26353274613806676, 0.25968004424605473],
                    [0.1798770909210089 , 0.4189131505356136 , 0.315559937135646  , 0.3580043312965455 , 0.21961026361278835, 0.33401429539402794],
                    [0.3627312246765608 , 1.                 , 0.3177088162360573 , 0.33717958752269606, 0.42721304526043646, 0.4418848773365286 ],
                    [0.23022188776811014, 0.33312500483020757, 0.5685215859338625 , 0.516108579092802  , 0.2525277507269663 , 0.39882010103564647]]),
    'bc2': np.array([[0.4331530545982232 , 0.8021114754102042, 0.6193770223921214, 0.805263213446112 , 0.4576159839890628 , 0.45192636311247647],
                     [0.32739841400381287, 0.6623380733795536, 0.5315418003462391, 0.5878415613660043, 0.39099185934149827, 0.556463041260487  ],
                     [0.593888507997764  , 1.                , 0.5344787405579978, 0.5606691008034166, 0.6719151044801772 , 0.6885075098543383 ],
                     [0.407441657928708  , 0.5552777408172893, 0.8138263781949708, 0.7658490927724129, 0.4412852365667118 , 0.6385827290812096 ]]),
    'hellinger': np.array([[0.4970991990182887 , 0.745086149104253 ,  0.6189131644009617, 0.7474691216786097 , 0.5133544059790144,  0.5095881123476633],
                           [0.42411919423790395, 0.647235004102539 ,  0.5617472181823832, 0.5983346315370234 , 0.4686259314344313,  0.5779396987524114],
                           [0.6022717199707793 , 1.                ,  0.5636566474690574, 0.5806716692957355 , 0.6536153649207127,  0.6647442194833503],
                           [0.47981443055426143, 0.5771698232151501,  0.7540037041910752, 0.7184069731654906 , 0.502521393302779 ,  0.631522051107993 ]]),
    }

@pytest.fixture(scope="session")
def points_binsize():
    binsize = 10
    points = pd.DataFrame({
        'x': pd.Series([1,2,3,101,102,103],dtype=float),
        'y': pd.Series([1,2,11,12,101,102],dtype=float),
        'bins_x': pd.Series([0,0,0,10,10,10],dtype=np.int8),
        'bins_y': pd.Series([0,0,1,1,10,10],dtype=np.int8),
        'hash': pd.Series([0,0,1,2,3,3],dtype=np.int8),
        'gene': pd.Series([0,0,1,0,1,2]).astype('category'),
        'type': pd.Series([0,0,1,0,1,1]).astype('category'),
    })
    return (points, binsize)

@pytest.fixture(scope="session")
def points_consistency():
    return (
        np.array([
            [ 341.4098 , -564.2289  ],
            [ 379.60126, -560.6449  ], 
            [ 378.91278, -555.6599  ],
            [ 378.9188 , -555.8149  ],
            [ 407.1855 , -549.42365 ],
            [ 387.43665, -547.03424 ],
        ]),
        np.array([
            [ 230.5077  , -243.94992 ],
            [ 299.22757 , -264.5527  ],
            [ 230.40369 , -245.80293 ],
        ]),
    )
 
@pytest.fixture(scope="session")
def points_consistency2():
    return (
        np.array([
            [-0.06026487,  0.91973415],
            [ 0.12884895,  0.23473125],
            [ 0.37306226,  0.73771559],
            [ 1.02827034,  0.99417839],
            [ 0.92398285,  0.24521422],
            [ 0.99138707,  0.80095276],
            [ 0.57264208, -0.00685763],
        ]),
    )

@pytest.fixture(scope="session")
def hash_other():
    points = pd.DataFrame({
        'x': pd.Series([0,10,20],dtype=float),
        'y': pd.Series([0,10,20],dtype=float),
        'hash': pd.Series([3,4,5],dtype=np.int8),
    })
    other = pd.DataFrame({
        'x': pd.Series([-10,-10,0,10,20,30,20,30],dtype=float),
        'y': pd.Series([-10,0,-10,10,20,30,30,20],dtype=float),
        'hash': pd.Series([0,1,2,4,5,8,6,7],dtype=np.int8),
    })
    return (points, other)

def test_bin(points_binsize):
    points, binsize = points_binsize
    bins = points[['bins_x','bins_y']]

    result = tc.utils.bin(points, binsize)
    result.columns = bins.columns

    tc.testing.assert_frame_equal(result, bins, rtol=1e-14, atol=1e-50)

    result = tc.utils.bin(points.copy(), binsize, bin_keys=['bins_x','bins_y'])

    tc.testing.assert_frame_equal(result, points, rtol=1e-14, atol=1e-50)

def test_hash(points_binsize):
    points, binsize = points_binsize
    hash = points['hash']

    result = tc.utils.hash(points, ['bins_x','bins_y'])
    result.name = hash.name

    tc.testing.assert_series_equal(result, hash, rtol=1e-14, atol=1e-50)

    result = tc.utils.hash(points.copy(), ['bins_x','bins_y'], hash_key='hash')

    tc.testing.assert_frame_equal(result, points, rtol=1e-14, atol=1e-50)

    result, sub_result = tc.utils.hash(points.copy(), ['bins_x','bins_y'], hash_key='hash', other=points.iloc[1:3,:].copy())

    tc.testing.assert_frame_equal(result, points, rtol=1e-14, atol=1e-50)
    tc.testing.assert_frame_equal(sub_result, points.iloc[1:3,:], rtol=1e-14, atol=1e-50)

def test_hash_other(hash_other):
    points, other = hash_other
    hash = points['hash']
    other_hash = other['hash']

    result,other_result = tc.utils.hash(points, ['x','y'], other=other)
    result.name = hash.name
    other_result.name = hash.name

    tc.testing.assert_series_equal(other_result, other_hash, rtol=1e-14, atol=1e-50)
    tc.testing.assert_series_equal(result, hash, rtol=1e-14, atol=1e-50)

@pytest.mark.parametrize('metric', ['euclidean','cosine','projection','bc','bc2','hellinger'])
def test_cdist(points_distances, metric):
    points = points_distances['points']
    distance = points_distances[metric]

    result = tc.utils.cdist(points,metric=metric)

    tc.testing.assert_dense_equal(result, distance)

@pytest.mark.parametrize('metric', ['euclidean','cosine','projection','bc','bc2','hellinger'])
@pytest.mark.parametrize('sparse', ['both','only A','only B','none'])
@pytest.mark.parametrize('singleAobs', [True,False])
@pytest.mark.parametrize('singleBobs', [True,False])
def test_cdist_counts(counts_distances, metric, sparse, singleAobs, singleBobs):
    countsA = counts_distances['countsA']
    countsB = counts_distances['countsB']
    if sparse == 'only A' or sparse == 'none':
        countsB = countsB.A
    if sparse == 'only B' or sparse == 'none':
        countsA = countsA.A
    distance = counts_distances[metric]
    if singleAobs:
        countsA = countsA[:1]
        distance = distance[:1]
    if singleBobs:
        countsB = countsB[:1]
        distance = distance[:,:1]

    result = tc.utils.cdist(countsA,countsB,metric=metric)
    with np.printoptions(precision=20):
        print(result)
    
    tc.testing.assert_dense_equal(result, distance)

@pytest.mark.parametrize('method', ['scipy','numba'])
def test_sparse_distance_matrix(points_distances,method):
    points = points_distances['points']
    distance = scipy.sparse.coo_matrix(points_distances['euclidean'])

    result = tc.utils.sparse_distance_matrix(points,10,method=method)

    tc.testing.assert_sparse_equal(result, distance)

    distance2 = distance.copy()
    distance2.data[distance2.data > 2 ] = 0
    distance2.eliminate_zeros()

    result = tc.utils.sparse_distance_matrix(points,2,method=method)

    tc.testing.assert_sparse_equal(result, distance2)

@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('parallel', [False,True])
@pytest.mark.parametrize('blocksize', [None,2,10])
@pytest.mark.parametrize('bin_batching', [True,False,10,200])
@pytest.mark.parametrize('low_mem', [False, True])
@pytest.mark.parametrize('numba_experimental_dict', [0,1,2])
def test_sparse_distance_matrix_consistency(points_consistency, dtype, parallel, blocksize, bin_batching, low_mem, numba_experimental_dict):
    
    for case,points in enumerate(points_consistency):
        
        print(f'test case {case}')
    
        result_scipy  = tc.utils.sparse_distance_matrix(points, 2, method='scipy', dtype=dtype)
        
        result_numba = tc.utils.sparse_distance_matrix(points, 2, method='numba', dtype=dtype, numba_blocksize=blocksize, parallel=parallel, bin_batching=bin_batching, low_mem=low_mem, numba_experimental_dict=numba_experimental_dict)
        
        tc.testing.assert_sparse_equal(result_scipy, result_numba)

@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('parallel', [False,True])
@pytest.mark.parametrize('blocksize', [None,0.8,0.9])
@pytest.mark.parametrize('bin_batching', [True,False,10,200])
@pytest.mark.parametrize('low_mem', [False, True])
@pytest.mark.parametrize('numba_experimental_dict', [0,1,2])
def test_sparse_distance_matrix_consistency2(points_consistency2, dtype, parallel, blocksize, bin_batching, low_mem, numba_experimental_dict):
    
    for case,points in enumerate(points_consistency2):
        
        print(f'test case {case}')
    
        result_scipy  = tc.utils.sparse_distance_matrix(points, 0.3, method='scipy', dtype=dtype)
        
        result_numba = tc.utils.sparse_distance_matrix(points, 0.3, method='numba', dtype=dtype, numba_blocksize=blocksize, parallel=parallel, bin_batching=bin_batching, low_mem=low_mem, numba_experimental_dict=numba_experimental_dict)
        
        tc.testing.assert_sparse_equal(result_scipy, result_numba)

@pytest.mark.parametrize('parallel', [False,True])
def test_dense_distance_matrix(points_distances,parallel):
    points = points_distances['points']
    distance = points_distances['euclidean']

    result = tc.utils.dense_distance_matrix(points,parallel=parallel)

    tc.testing.assert_dense_equal(result, distance)

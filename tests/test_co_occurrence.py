import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def adata_coocc():
    adata = ad.AnnData(scipy.sparse.csr_matrix((10,10)))
    adata.obs['x'] =           [0,0,1,1,  0, 0, 1, 1,  0, 0,]
    adata.obs['y'] =           [0,1,0,1, 10,11,10,11, 30,31,]
    adata.obs['c'] = pd.Series([0,0,1,1,  2, 2, 1, 1,  3, 3,], index=adata.obs.index).astype('category')
    adata.obs['d'] = adata.obs['x'].copy()
    ann_c = adata.obs['c'].cat.categories
    ann_c.name = 'c'
    ann_d = adata.obs['d'].astype('category').cat.categories
    ann_d.name = 'd'
    counts = np.array([[
        [[ 2.,  0.],
         [ 8.,  0.],
         [ 4.,  0.],
         [ 0.,  4.]],

        [[ 8.,  0.],
         [12.,  0.],
         [ 8.,  0.],
         [ 0.,  8.]],

        [[ 4.,  0.],
         [ 8.,  0.],
         [ 2.,  0.],
         [ 0.,  4.]],

        [[ 0.,  4.],
         [ 0.,  8.],
         [ 0.,  4.],
         [ 2.,  0.]]
    ]])
    occ = np.array([
       [[0.68518519, 0.75      ],
        [1.15625   , 0.5       ],
        [1.14197531, 0.75      ],
        [0.68518519, 1.5       ]],

       [[1.15625   , 0.5       ],
        [0.93945312, 0.33333333],
        [1.15625   , 0.5       ],
        [0.38541667, 1.8       ]],

       [[1.14197531, 0.75      ],
        [1.15625   , 0.5       ],
        [0.68518519, 0.75      ],
        [0.68518519, 1.5       ]],

       [[0.68518519, 1.5       ],
        [0.38541667, 1.8       ],
        [0.68518519, 1.5       ],
        [6.16666667, 0.12      ]]
    ])
    log_occ = np.array([
       [[-0.37806613, -0.28768207],
        [ 0.14518201, -0.69314718],
        [ 0.13275949, -0.28768207],
        [-0.37806613,  0.40546511]],

       [[ 0.14518201, -0.69314718],
        [-0.06245735, -1.09861229],
        [ 0.14518201, -0.69314718],
        [-0.95343028,  0.58778666]],

       [[ 0.13275949, -0.28768207],
        [ 0.14518201, -0.69314718],
        [-0.37806613, -0.28768207],
        [-0.37806613,  0.40546511]],

       [[-0.37806613,  0.40546511],
        [-0.95343028,  0.58778666],
        [-0.37806613,  0.40546511],
        [ 1.81915844, -2.12026354]]
    ])
    composition = np.array([
       [[0.16666667, 0.125     ],
        [0.28125   , 0.08333333],
        [0.27777778, 0.125     ],
        [0.16666667, 0.25      ]],

       [[0.5       , 0.125     ],
        [0.40625   , 0.08333333],
        [0.5       , 0.125     ],
        [0.16666667, 0.45      ]],

       [[0.27777778, 0.125     ],
        [0.28125   , 0.08333333],
        [0.16666667, 0.125     ],
        [0.16666667, 0.25      ]],

       [[0.05555556, 0.625     ],
        [0.03125   , 0.75      ],
        [0.05555556, 0.625     ],
        [0.5       , 0.05      ]]
    ])
    log_composition = np.array([
       [[-1.79175947, -2.07944154],
        [-1.26851133, -2.48490665],
        [-1.28093385, -2.07944154],
        [-1.79175947, -1.38629436]],

       [[-0.69314718, -2.07944154],
        [-0.90078655, -2.48490665],
        [-0.69314718, -2.07944154],
        [-1.79175947, -0.7985077 ]],

       [[-1.28093385, -2.07944154],
        [-1.26851133, -2.48490665],
        [-1.79175947, -2.07944154],
        [-1.79175947, -1.38629436]],

       [[-2.89037176, -0.47000363],
        [-3.4657359 , -0.28768207],
        [-2.89037176, -0.47000363],
        [-0.69314718, -2.99573227]]
    ])
    distance_distribution = np.array([
        [[0.75      , 0.25      ],
         [0.9       , 0.1       ],
         [0.83333333, 0.16666667],
         [0.16666667, 0.83333333]],
       
        [[0.9       , 0.1       ],
         [0.92857143, 0.07142857],
         [0.9       , 0.1       ],
         [0.1       , 0.9       ]],
       
        [[0.83333333, 0.16666667],
         [0.9       , 0.1       ],
         [0.75      , 0.25      ],
         [0.16666667, 0.83333333]],
       
        [[0.16666667, 0.83333333],
         [0.1       , 0.9       ],
         [0.16666667, 0.83333333],
         [0.75      , 0.25      ]]
    ])
    log_distance_distribution = np.array([
        [[-0.28768207, -1.38629436],
         [-0.10536052, -2.30258509],
         [-0.18232156, -1.79175947],
         [-1.79175947, -0.18232156]],

        [[-0.10536052, -2.30258509],
         [-0.07410797, -2.63905733],
         [-0.10536052, -2.30258509],
         [-2.30258509, -0.10536052]],

        [[-0.18232156, -1.79175947],
         [-0.10536052, -2.30258509],
         [-0.28768207, -1.38629436],
         [-1.79175947, -0.18232156]],

        [[-1.79175947, -0.18232156],
         [-2.30258509, -0.10536052],
         [-1.79175947, -0.18232156],
         [-0.28768207, -1.38629436]]
    ])
    relative_distance_distribution = np.array([
        [[1.08333333, 0.8125    ],
         [1.2375    , 0.36666667],
         [1.2037037 , 0.54166667],
         [0.72222222, 1.08333333]],
       
        [[1.3       , 0.325     ],
         [1.27678571, 0.26190476],
         [1.3       , 0.325     ],
         [0.43333333, 1.17      ]],
       
        [[1.2037037 , 0.54166667],
         [1.2375    , 0.36666667],
         [1.08333333, 0.8125    ],
         [0.72222222, 1.08333333]],
       
        [[0.24074074, 2.70833333],
         [0.1375    , 3.3       ],
         [0.24074074, 2.70833333],
         [3.25      , 0.325     ]]
    ])
    log_relative_distance_distribution = np.array([
        [[ 0.08004271, -0.20763936],
         [ 0.21309322, -1.00330211],
         [ 0.18540322, -0.61310447],
         [-0.3254224 ,  0.08004271]],
       
        [[ 0.26236426, -1.1239301 ],
         [ 0.24434576, -1.33977435],
         [ 0.26236426, -1.1239301 ],
         [-0.83624802,  0.15700375]],
       
        [[ 0.18540322, -0.61310447],
         [ 0.21309322, -1.00330211],
         [ 0.08004271, -0.20763936],
         [-0.3254224 ,  0.08004271]],
       
        [[-1.42403469,  0.99633344],
         [-1.98413136,  1.19392247],
         [-1.42403469,  0.99633344],
         [ 1.178655  , -1.1239301 ]]
    ])
    adata.uns['coocc_c_c'] = {
        'occ': occ,
        'log_occ': log_occ,
        'z': None,
        'composition': composition,
        'log_composition': log_composition,
        'distance_distribution': distance_distribution,
        'log_distance_distribution': log_distance_distribution,
        'relative_distance_distribution': relative_distance_distribution,
        'log_relative_distance_distribution': log_relative_distance_distribution,
        'sample_counts': counts,
        'interval': np.array([ 0., 15., 32.]),
        'annotation': ann_c,
        'center': ann_c,
        'permutation_counts': None,
    }
    counts = np.array([[
        [[ 6.,  4.],
         [ 8.,  0.]],

        [[16.,  8.],
         [12.,  0.]],

        [[ 6.,  4.],
         [ 8.,  0.]],

        [[ 2.,  8.],
         [ 0.,  8.]]
    ]])
    occ = np.array([
       [[0.84926471, 1.19047619],
        [1.16015625, 0.55555556]],

       [[1.1       , 1.28571429],
        [0.89375   , 0.33333333]],

       [[0.84926471, 1.19047619],
        [1.16015625, 0.55555556]],

       [[1.45588235, 0.71428571],
        [0.515625  , 1.66666667]]
    ])
    log_occ = np.array([
       [[-0.16338436,  0.17435339],
        [ 0.14855469, -0.58778666]],

       [[ 0.09531018,  0.25131443],
        [-0.11232918, -1.09861229]],

       [[-0.16338436,  0.17435339],
        [ 0.14855469, -0.58778666]],

       [[ 0.37561214, -0.33647224],
        [-0.66237552,  0.51082562]]
    ])
    composition = np.array([
       [[0.20588235, 0.17857143],
        [0.28125   , 0.08333333]],

       [[0.5       , 0.32142857],
        [0.40625   , 0.08333333]],

       [[0.20588235, 0.17857143],
        [0.28125   , 0.08333333]],

       [[0.08823529, 0.32142857],
        [0.03125   , 0.75      ]]
    ])
    log_composition = np.array([
       [[-1.58045038, -1.7227666 ],
        [-1.26851133, -2.48490665]],

       [[-0.69314718, -1.13497993],
        [-0.90078655, -2.48490665]],

       [[-1.58045038, -1.7227666 ],
        [-1.26851133, -2.48490665]],

       [[-2.42774824, -1.13497993],
        [-3.4657359 , -0.28768207]]
    ])
    distance_distribution = np.array([
        [[0.583333333333, 0.416666666667],
         [0.9           , 0.1           ]],
       
        [[0.653846153846, 0.346153846154],
         [0.928571428571, 0.071428571429]],
       
        [[0.583333333333, 0.416666666667],
         [0.9           , 0.1           ]],
       
        [[0.25          , 0.75          ],
         [0.1           , 0.9           ]]
    ])
    log_distance_distribution = np.array([
        [[-0.538996500733, -0.875468737354],
         [-0.105360515658, -2.302585092994]],
       
        [[-0.424883193965, -1.060871960685],
         [-0.074107972154, -2.639057329615]],
       
        [[-0.538996500733, -0.875468737354],
         [-0.105360515658, -2.302585092994]],
       
        [[-1.38629436112 , -0.287682072452],
         [-2.302585092994, -0.105360515658]]
    ])
    relative_distance_distribution = np.array([
        [[1.063725490196, 0.922619047619],
         [1.2375        , 0.366666666667]],
       
        [[1.192307692308, 0.766483516484],
         [1.276785714286, 0.261904761905]],
       
        [[1.063725490196, 0.922619047619],
         [1.2375        , 0.366666666667]],
       
        [[0.455882352941, 1.660714285714],
         [0.1375        , 3.3           ]]
    ])
    log_relative_distance_distribution = np.array([
        [[ 0.061777359696, -0.080538862484],
         [ 0.213093215461, -1.003302108864]],
       
        [[ 0.175890666464, -0.265942085815],
         [ 0.244345758965, -1.339774345485]],
       
        [[ 0.061777359696, -0.080538862484],
         [ 0.213093215461, -1.003302108864]],
       
        [[-0.785520500691,  0.507247802418],
         [-1.984131361876,  1.193922468472]]
    ])
    adata.uns['coocc_c_d'] = {
        'occ': occ,
        'log_occ': log_occ,
        'z': None,
        'composition': composition,
        'log_composition': log_composition,
        'distance_distribution': distance_distribution,
        'log_distance_distribution': log_distance_distribution,
        'relative_distance_distribution': relative_distance_distribution,
        'log_relative_distance_distribution': log_relative_distance_distribution,
        'sample_counts': counts,
        'interval': np.array([ 0., 15., 32.]),
        'annotation': ann_c,
        'center': ann_d,
        'permutation_counts': None,
    }
    counts = np.array([[
        [[ 6.,  4.],
         [16.,  8.],
         [ 6.,  4.],
         [ 2.,  8.]],

        [[ 8.,  0.],
         [12.,  0.],
         [ 8.,  0.],
         [ 0.,  8.]]
    ]])
    occ = np.array([
       [[0.84926471, 1.19047619],
        [1.1       , 1.28571429],
        [0.84926471, 1.19047619],
        [1.45588235, 0.71428571]],

       [[1.16015625, 0.55555556],
        [0.89375   , 0.33333333],
        [1.16015625, 0.55555556],
        [0.515625  , 1.66666667]]
    ])
    log_occ = np.array([
       [[-0.16338436,  0.17435339],
        [ 0.09531018,  0.25131443],
        [-0.16338436,  0.17435339],
        [ 0.37561214, -0.33647224]],

       [[ 0.14855469, -0.58778666],
        [-0.11232918, -1.09861229],
        [ 0.14855469, -0.58778666],
        [-0.66237552,  0.51082562]]
    ])
    composition = np.array([
       [[0.4375    , 0.83333333],
        [0.56666667, 0.9       ],
        [0.4375    , 0.83333333],
        [0.75      , 0.5       ]],

       [[0.5625    , 0.16666667],
        [0.43333333, 0.1       ],
        [0.5625    , 0.16666667],
        [0.25      , 0.5       ]]
    ])
    log_composition = np.array([
       [[-0.82667857, -0.18232156],
        [-0.56798404, -0.10536052],
        [-0.82667857, -0.18232156],
        [-0.28768207, -0.69314718]],

       [[-0.57536414, -1.79175947],
        [-0.83624802, -2.30258509],
        [-0.57536414, -1.79175947],
        [-1.38629436, -0.69314718]]
    ])
    distance_distribution = np.array([
        [[0.58333333, 0.41666667],
         [0.65384615, 0.34615385],
         [0.58333333, 0.41666667],
         [0.25      , 0.75      ]],
      
        [[0.9       , 0.1       ],
         [0.92857143, 0.07142857],
         [0.9       , 0.1       ],
         [0.1       , 0.9       ]]
    ])
    log_distance_distribution = np.array([
        [[-0.5389965 , -0.87546874],
         [-0.42488319, -1.06087196],
         [-0.5389965 , -0.87546874],
         [-1.38629436, -0.28768207]],
       
        [[-0.10536052, -2.30258509],
         [-0.07410797, -2.63905733],
         [-0.10536052, -2.30258509],
         [-2.30258509, -0.10536052]]
    ])
    relative_distance_distribution = np.array([
        [[0.802083333333, 1.527777777778],
         [0.871794871795, 1.384615384615],
         [0.802083333333, 1.527777777778],
         [1.375         , 0.916666666667]],
       
        [[1.2375        , 0.366666666667],
         [1.238095238095, 0.285714285714],
         [1.2375        , 0.366666666667],
         [0.55          , 1.1           ]]
    ])
    log_relative_distance_distribution = np.array([
        [[-0.2205427696,  0.4238142468],
         [-0.1372011215,  0.3254224004],
         [-0.2205427696,  0.4238142468],
         [ 0.3184537311, -0.087011377 ]],
       
        [[ 0.2130932155, -1.0033021089],
         [ 0.2135741003, -1.2527629685],
         [ 0.2130932155, -1.0033021089],
         [-0.5978370008,  0.0953101798]]
    ])
    adata.uns['coocc_d_c'] = {
        'occ': occ,
        'log_occ': log_occ,
        'z': None,
        'composition': composition,
        'log_composition': log_composition,
        'distance_distribution': distance_distribution,
        'log_distance_distribution': log_distance_distribution,
        'relative_distance_distribution': relative_distance_distribution,
        'log_relative_distance_distribution': log_relative_distance_distribution,
        'sample_counts': counts,
        'interval': np.array([ 0., 15., 32.]),
        'annotation': ann_d,
        'center': ann_c,
        'permutation_counts': None,
    }
    counts = np.array([[
        [[14., 16.],
         [16.,  8.]],

        [[16.,  8.],
         [12.,  0.]]
    ]])
    occ = np.array([
       [[0.90820312, 0.90532544],
        [1.09791667, 1.24615385]],

       [[1.09791667, 1.24615385],
        [0.89555556, 0.36      ]]
    ])
    log_occ = np.array([
       [[-0.09628722, -0.09946079],
        [ 0.09341444,  0.22006188]],

       [[ 0.09341444,  0.22006188],
        [-0.11031102, -1.02165125]]
    ])
    composition = np.array([
       [[0.46875   , 0.65384615],
        [0.56666667, 0.9       ]],

       [[0.53125   , 0.34615385],
        [0.43333333, 0.1       ]]
    ])
    log_composition = np.array([
       [[-0.7576857 , -0.42488319],
        [-0.56798404, -0.10536052]],

       [[-0.63252256, -1.06087196],
        [-0.83624802, -2.30258509]]
    ])
    distance_distribution = np.array([
        [[0.46875       , 0.53125       ],
         [0.653846153846, 0.346153846154]],
       
        [[0.653846153846, 0.346153846154],
         [0.928571428571, 0.071428571429]]
    ])
    log_distance_distribution = np.array([
        [[-0.757685701698, -0.632522558744],
         [-0.424883193965, -1.060871960685]],
       
        [[-0.424883193965, -1.060871960685],
         [-0.074107972154, -2.639057329615]]
    ])
    relative_distance_distribution = np.array([
        [[0.849609375   , 1.185096153846],
         [0.871794871795, 1.384615384615]],
       
        [[1.185096153846, 0.772189349112],
         [1.238095238095, 0.285714285714]]
    ])
    log_relative_distance_distribution = np.array([
        [[-0.162978593951,  0.169823913781],
         [-0.137201121513,  0.325422400435]],
       
        [[ 0.169823913781, -0.25852548816 ],
         [ 0.213574100298, -1.252762968495]]
    ])
    adata.uns['coocc_d_d'] = {
        'occ': occ,
        'log_occ': log_occ,
        'z': None,
        'composition': composition,
        'log_composition': log_composition,
        'distance_distribution': distance_distribution,
        'log_distance_distribution': log_distance_distribution,
        'relative_distance_distribution': relative_distance_distribution,
        'log_relative_distance_distribution': log_relative_distance_distribution,
        'sample_counts': counts,
        'interval': np.array([ 0., 15., 32.]),
        'annotation': ann_d,
        'center': ann_d,
        'permutation_counts': None,
    }
    return adata

def assert_coocc_equal(left, right):
    np.set_printoptions(precision=12)
    #print(left['z'])
    #print(left['permutation_counts'])
    #print(left['comparisons']['diff']['rel_occ'])
    #print(left['comparisons']['diff']['p_t_fdr_bh'])
    #print(left['log_relative_distance_distribution'])
    for k in left:
        assert(k in right)
    for k in right:
        assert(k in left)
    tc.testing.assert_dense_equal(left['sample_counts'], right['sample_counts'])
    tc.testing.assert_dense_equal(left['interval'], right['interval'])
    tc.testing.assert_index_equal(left['annotation'], right['annotation'])
    tc.testing.assert_index_equal(left['center'], right['center'])
    tc.testing.assert_dense_equal(left['occ'], right['occ'])
    tc.testing.assert_dense_equal(left['log_occ'], right['log_occ'])
    if left['z'] is None:
        assert(right['z'] is None)
    else:
        tc.testing.assert_dense_equal(left['z'], right['z'])
    tc.testing.assert_dense_equal(left['composition'], right['composition'])
    tc.testing.assert_dense_equal(left['log_composition'], right['log_composition'])
    #tc.testing.assert_dense_equal(left['distance_distribution'], right['distance_distribution'])
    #tc.testing.assert_dense_equal(left['log_distance_distribution'], right['log_distance_distribution'])
    #tc.testing.assert_dense_equal(left['relative_distance_distribution'], right['relative_distance_distribution'])
    #tc.testing.assert_dense_equal(left['log_relative_distance_distribution'], right['log_relative_distance_distribution'])
    if left['permutation_counts'] is None:
        assert(right['permutation_counts'] is None)
    else:
        tc.testing.assert_dense_equal(left['permutation_counts'], right['permutation_counts'])

@pytest.mark.parametrize('anno', ['c','d'])
@pytest.mark.parametrize('center', ['c','d'])
@pytest.mark.parametrize('soft_anno', [False,True])
@pytest.mark.parametrize('soft_center', [False,True])
@pytest.mark.parametrize('sparse', [False,True])
def test_cooccurrence(adata_coocc, anno, center, soft_anno, soft_center, sparse):
    adata = adata_coocc.copy()
    
    if soft_anno:
        adata.obsm[anno] = pd.get_dummies(adata.obs[anno])
        del adata.obs[anno]
    if soft_center and center in adata.obs:
        adata.obsm[center] = pd.get_dummies(adata.obs[center])
        del adata.obs[center]
    
    result = tc.tl.co_occurrence(adata, annotation_key=anno, center_key=center, delta_distance=17, max_distance=32, sparse=sparse, reads=False)
    
    assert_coocc_equal(result, adata.uns[f'coocc_{anno}_{center}'])

@pytest.fixture(scope="session")
def adatas_coocc_permute():

    N = 1000
    n_samples_dir = 2
    n_samples = n_samples_dir**2

    np.random.seed(42)

    pattern = np.random.normal(size=(2,N)) * 0.2
    pattern[0] += np.arange(N) // (np.sqrt(N))
    pattern[1] += np.arange(N) % (np.sqrt(N))

    samples = tc.utils.spatial_split(pd.DataFrame(pattern.T,columns=['x','y']), position_split=n_samples_dir)

    adata0 = ad.AnnData(np.ones(shape=(N,1)),obs=pd.DataFrame({'x':pattern[0],'y':pattern[1]+0.12,'a':'A','sample':samples}))
    adata1 = ad.AnnData(np.ones(shape=(N,1)),obs=pd.DataFrame({'x':pattern[0],'y':pattern[1]+0.01,'a':'B','sample':samples}))
    adata2 = ad.AnnData(np.ones(shape=(N,1)),obs=pd.DataFrame({'x':pattern[0],'y':pattern[1]-0.10,'a':'C','sample':samples}))
    adataA = adata0.concatenate([adata1,adata2]).copy()
    adataA.obs['sample'] = adataA.obs['sample'].astype('category')
    adata0 = ad.AnnData(np.ones(shape=(N,1)),obs=pd.DataFrame({'x':pattern[0],'y':pattern[1]+0.12,'a':'A','sample':samples}))
    adata1 = ad.AnnData(np.ones(shape=(N,1)),obs=pd.DataFrame({'x':pattern[0],'y':pattern[1]+0.01,'a':'B','sample':samples}))
    adata2 = ad.AnnData(np.ones(shape=(N,1)),obs=pd.DataFrame({'x':pattern[0],'y':pattern[1]+0.10,'a':'C','sample':samples}))
    adataB = adata0.concatenate([adata1,adata2]).copy()
    adataB.obs['sample'] = adataB.obs['sample'].astype('category')
    
    adataA.uns['correct'] = {
        'occ': np.array([
            [[0.015407113095], [1.977658645013], [0.065194656208]],
            [[1.977658645013], [0.003997679548], [1.977658645013]],
            [[0.065194656208], [1.977658645013], [0.015407113095]]
        ]),
        'log_occ': np.array([
            [[-4.17298347], [ 0.68190772], [-2.82916387]],
            [[ 0.68190772], [-5.52205138], [ 0.68190772]],
            [[-2.82916387], [ 0.68190772], [-4.17298347]]
        ]),
        'z': np.array([
            [[-119.59174901], [   2.20494998], [ -71.97792357]],
            [[   2.20494998], [-331.86388086], [   2.14255343]],
            [[ -71.97792357], [   2.14255343], [-108.75929258]]
        ]),
        'composition': np.array([
            [[0.00388749], [0.49900989], [0.01647366]],
            [[0.97963886], [0.00198021], [0.97963886]],
            [[0.01647366], [0.49900989], [0.00388749]]
        ]),
        'log_composition': np.array([
            [[-5.55002055], [-0.69512936], [-4.20620095]],
            [[-0.0205954 ], [-6.22455451], [-0.0205954 ]],
            [[-4.20620095], [-0.69512936], [-5.55002055]]
        ]),
        'distance_distribution': np.array([
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]]
        ]),
        'log_distance_distribution': np.array([
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]]
        ]),
        'relative_distance_distribution': np.array([
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]]
        ]),
        'log_relative_distance_distribution': np.array([
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]]
        ]),
        'sample_counts': np.array([
           [[[  0.        ], [251.        ], [  2.        ]],
            [[251.        ], [  0.        ], [251.        ]],
            [[  2.        ], [251.        ], [  0.        ]]],
           [[[  0.        ], [251.        ], [  1.        ]],
            [[251.        ], [  0.        ], [251.        ]],
            [[  1.        ], [251.        ], [  0.        ]]],
           [[[  0.        ], [250.        ], [  5.        ]],
            [[250.        ], [  0.        ], [250.        ]],
            [[  5.        ], [250.        ], [  0.        ]]],
           [[[  0.        ], [252.        ], [  5.        ]],
            [[252.        ], [  0.        ], [252.        ]],
            [[  5.        ], [252.        ], [  0.        ]]],
        ]),
        'permutation_counts': np.array([
            [[[ 82.], [170.], [ 82.]],
             [[ 83.], [170.], [ 82.]],
             [[ 88.], [162.], [ 89.]]],
            [[[ 82.], [ 83.], [ 88.]],
             [[170.], [170.], [162.]],
             [[ 82.], [ 82.], [ 89.]]],
            [[[ 85.], [167.], [ 86.]],
             [[ 86.], [164.], [ 86.]],
             [[ 81.], [171.], [ 80.]]],
            [[[ 85.], [ 86.], [ 81.]],
             [[167.], [164.], [171.]],
             [[ 86.], [ 86.], [ 80.]]],
            [[[ 91.], [160.], [ 92.]],
             [[ 82.], [170.], [ 81.]],
             [[ 82.], [170.], [ 82.]]],
            [[[ 91.], [ 82.], [ 82.]],
             [[160.], [170.], [170.]],
             [[ 92.], [ 81.], [ 82.]]],
            [[[ 87.], [167.], [ 90.]],
             [[ 83.], [170.], [ 81.]],
             [[ 87.], [167.], [ 86.]]],
            [[[ 87.], [ 83.], [ 87.]],
             [[167.], [170.], [167.]],
             [[ 90.], [ 81.], [ 86.]]]
        ]),
        'interval': np.array([0. , 0.2]),
        'annotation': pd.Index(['A', 'B', 'C'], dtype='object', name='a'),
        'center': pd.Index(['A', 'B', 'C'], dtype='object', name='a'),
    }
    adataB.uns['correct'] = {
        'occ': np.array([
            [[0.005954361049], [1.498264374151], [1.495284250813]],
            [[1.498264374151], [0.005936713687], [1.497523267432]],
            [[1.495284250813], [1.497523267432], [0.005960260342]]]),
        'log_occ': np.array([
            [[-5.123632416687], [ 0.404307093429], [ 0.402316062746]],
            [[ 0.404307093429], [-5.126603697897], [ 0.403812533426]],
            [[ 0.402316062746], [ 0.403812533426], [-5.122641336085]]
        ]),
        'z': np.array([
            [[-157.816226292941], [  25.940853763277], [  11.523060669285]],
            [[  25.940853763277], [-346.365421236357], [  23.648108033949]],
            [[  11.523060669285], [  23.648108033949], [-211.77044466573 ]]
        ]),
        'composition': np.array([
            [[0.00198413089 ], [0.499256441549], [0.498263395816]],
            [[0.499999015806], [0.001981188049], [0.499751491054]],
            [[0.498016853304], [0.498762370402], [0.001985113131]]
        ]),
        'log_composition': np.array([
            [[-6.222575283879], [-0.694635773764], [-0.696626804447]],
            [[-0.693150133159], [-6.224060924484], [-0.693644693162]],
            [[-0.697122344748], [-0.695625874068], [-6.222079743578]]
        ]),
        'distance_distribution': np.array([
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]]
        ]),
        'log_distance_distribution': np.array([
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]]
        ]),
        'relative_distance_distribution': np.array([
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]],
            [[1.], [1.], [1.]]
        ]),
        'log_relative_distance_distribution': np.array([
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]]
        ]),
        'sample_counts': np.array([
            [[[  0.        ], [251.        ], [250.        ]], 
             [[251.        ], [  0.        ], [251.        ]], 
             [[250.        ], [251.        ], [  0.        ]]], 
            [[[  0.        ], [251.        ], [250.        ]], 
             [[251.        ], [  0.        ], [251.        ]], 
             [[250.        ], [251.        ], [  0.        ]]], 
            [[[  0.        ], [250.        ], [250.        ]], 
             [[250.        ], [  0.        ], [250.        ]], 
             [[250.        ], [250.        ], [  0.        ]]], 
            [[[  0.        ], [252.        ], [250.        ]], 
             [[252.        ], [  0.        ], [251.        ]], 
             [[250.        ], [251.        ], [  0.        ]]], 
        ]),
        'permutation_counts': np.array([
             [[[162.], [170.], [171.]],
              [[164.], [169.], [167.]],
              [[175.], [163.], [163.]]],
             [[[162.], [164.], [175.]],
              [[170.], [169.], [163.]],
              [[171.], [167.], [163.]]],
             [[[176.], [167.], [160.]],
              [[164.], [164.], [172.]],
              [[161.], [171.], [169.]]],
             [[[176.], [164.], [161.]],
              [[167.], [164.], [171.]],
              [[160.], [172.], [169.]]],
             [[[163.], [160.], [177.]],
              [[165.], [170.], [165.]],
              [[172.], [170.], [158.]]],
             [[[163.], [165.], [172.]],
              [[160.], [170.], [170.]],
              [[177.], [165.], [158.]]],
             [[[167.], [166.], [168.]],
              [[165.], [170.], [168.]],
              [[170.], [167.], [165.]]],
             [[[167.], [165.], [170.]],
              [[166.], [170.], [167.]],
              [[168.], [168.], [165.]]]
            ]),
        'interval': np.array([0. , 0.2]),
        'annotation': pd.Index(['A', 'B', 'C'], dtype='object', name='a'),
        'center': pd.Index(['A', 'B', 'C'], dtype='object', name='a'),
    }
    
    return {'adataA':adataA, 'adataB':adataB}

def test_cooccurrence_permute(adatas_coocc_permute):
    
    adatas = {k:a.copy() for k,a in adatas_coocc_permute.items()} # dont change the input

    for name,adata in adatas.items():
        tc.tl.co_occurrence(adata, annotation_key='a', sample_key='sample', max_distance=0.2, delta_distance=0.2, min_distance=0, sparse=False, result_key='results', n_permutation=2,)

    for key, adata in adatas.items():
#        print(adata.uns['results'])
        assert_coocc_equal(adata.uns['results'], adata.uns['correct'])

@pytest.fixture(scope="session")
def adata_anncoor():
    adata = ad.AnnData(scipy.sparse.csr_matrix((10,10)))
    adata.obs['x'] = [0,0,1,1,  0, 0, 1, 1,  0, 0,]
    adata.obs['y'] = [0,1,0,1, 10,11,10,11, 30,31,]
    adata.obs['c'] = [1,1,1,0,  1, 0, 0, 0,  0, 0,]
    adata.obsm['d'] = pd.DataFrame({'c':[ 1.085564, 1.085564, 1.085564, 1.085564, 9.149757,10.080241, 9.149757,10.080241,29.000077,31.326286]},index=adata.obs.index)
    return adata

@pytest.mark.parametrize('sparse', [False,True])
def test_annotation_coordinate(adata_anncoor, sparse):
    
    adata = adata_anncoor.copy()

    result = tc.tl.annotation_coordinate(adata, annotation_key='c', critical_neighbourhood_size=3, sparse=sparse)

    tc.testing.assert_frame_equal(result, adata.obsm['d'])

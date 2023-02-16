import pytest
import numpy as np
import scipy.sparse
import pandas as pd
import anndata as ad
import tacco as tc

@pytest.fixture(scope="session")
def adata_and_mixture():
    rng = np.random.Generator(np.random.PCG64(42))
    adataA = ad.AnnData(rng.poisson([0.5,8],[4,2]),dtype=np.uint8)
    adataB = ad.AnnData(rng.poisson([8,0.5],[4,2]),dtype=np.uint8)
    adata = ad.concat([adataA,adataB],label='celltype',index_unique='-')
    
    # Fixed result from reference run
    #mixture = tc.tl.mix_in_silico(adata,'celltype',n_samples=10,min_counts=None,)
    mixture = ad.AnnData(scipy.sparse.csc_matrix(np.array([
       [ 0., 10.],
       [ 0.,  0.],
       [ 8., 12.],
       [ 0.,  0.],
       [ 0.,  2.],
       [ 0.,  0.],
       [ 0.,  8.],
       [ 9.,  8.],
       [ 6.,  8.],
       [ 0.,  0.],
    ])),dtype=np.float64,obs=pd.DataFrame(np.array([
       [0.55458479, 0.466721  ],
       [0.06381726, 0.04380377],
       [0.82763117, 0.15428949],
       [0.6316644 , 0.68304895],
       [0.75808774, 0.74476216],
       [0.35452597, 0.96750973],
       [0.97069802, 0.32582536],
       [0.89312112, 0.37045971],
       [0.7783835 , 0.46955581],
       [0.19463871, 0.18947136],
    ]),index=[str(i) for i in range(10)],columns=['x','y']), var=pd.DataFrame(index=['0','1']),)
    mixture.obsm['celltype'] = pd.DataFrame(np.array([
       [0.99711234, 0.04599353],
       [0.        , 0.00254775],
       [1.01645234, 0.99988295],
       [0.        , 0.        ],
       [0.18671093, 0.00497749],
       [0.        , 0.        ],
       [0.9957687 , 0.04224399],
       [0.99999909, 1.37700033],
       [0.99209559, 1.00165128],
       [0.        , 0.        ],
    ]),index=[str(i) for i in range(10)],columns=pd.CategoricalIndex(['0','1']))
    mixture.obsm['reads_celltype'] = pd.DataFrame(np.array([
       [ 9.97112339,  0.27596119],
       [ 0.        ,  0.02292972],
       [13.13156383,  6.99918066],
       [ 0.        ,  0.        ],
       [ 2.05382024,  0.04479742],
       [ 0.        ,  0.        ],
       [ 7.96614958,  0.2936727 ],
       [ 7.99999272,  8.71255904],
       [ 7.93676469,  6.01155962],
       [ 0.        ,  0.        ],
]),index=[str(i) for i in range(10)],columns=pd.CategoricalIndex(['0','1']))

    return ( adata, mixture )

@pytest.mark.parametrize('dtype', [np.int64, np.float32])
@pytest.mark.parametrize('sparsity', ['dense','csr','csc'])
def test_mix_in_silico(adata_and_mixture, dtype, sparsity,):
    adata, mixture = adata_and_mixture
    
    adata, mixture = adata.copy(), mixture.copy() # dont change the original
    if sparsity == 'csr':
        adata.X = scipy.sparse.csr_matrix(adata.X)
    elif sparsity == 'csc':
        adata.X = scipy.sparse.csc_matrix(adata.X)
    elif sparsity == 'dense':
        mixture.X = mixture.X.A
    
    adata.X = adata.X.astype(dtype)
    
    result = tc.tl.mix_in_silico(adata,'celltype',n_samples=10,min_counts=None,)
    
    tc.testing.assert_adata_equal(result,mixture)

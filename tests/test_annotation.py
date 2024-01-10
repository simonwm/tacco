import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def adata_reference_and_typing0():
    reference = ad.AnnData(X=np.array([
        [1,0,0],
        [2,0,0],
        [3,0,0],
        [0,1,0],
        [0,2,0],
        [0,3,0],
        [0,0,1],
        [0,0,2],
        [0,0,3],
    ]))
    reference.obs=pd.DataFrame({
        'type': pd.Series([0,0,0,1,1,1,2,2,2,],dtype='category',index=pd.Index(range(reference.shape[0])).astype(str)),
    })
    adata = ad.AnnData(X=np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [1,1,0],
        [1,1,0],
        [1,1,0],
    ]))
    adata.obsm['type']=pd.DataFrame(np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0.5,0.5,0],
        [0.5,0.5,0],
        [0.5,0.5,0],
    ]),index=adata.obs.index,columns=pd.Index(reference.obs['type'].cat.categories,dtype=reference.obs['type'].dtype,name='type'))
    adata.uns['annotation_prior']=pd.Series(adata.obsm['type'].sum(axis=0).to_numpy(),index=adata.obsm['type'].columns)
    return ( adata, reference, )

@pytest.fixture(scope="session")
def adata_reference_and_typing1():
    reference = ad.AnnData(scipy.sparse.csr_matrix([
            [10,0,0],
            [0,20,10],
        ], dtype=np.int8),
        dtype=np.int8,
        obs=pd.DataFrame({'type': pd.Series([0,1]).astype('category')}),
    )
    adata = ad.AnnData(X=scipy.sparse.csr_matrix([
        [3,2,1],
    ]))
    adata.obsm['type']=pd.DataFrame(np.array([
        [0.5,0.5],
    ]),index=adata.obs.index,columns=pd.Index(reference.obs['type'].cat.categories,dtype=reference.obs['type'].dtype,name='type'))
    adata.obsm['annotation0']=pd.DataFrame(np.array([
        [0.5,0.5],
    ]),index=adata.obs.index,columns=pd.Index(reference.obs['type'].cat.categories,dtype=reference.obs['type'].dtype))
    adata.obsm['annotation1']=adata.obsm['annotation0']
    adata.obsm['annotation2']=pd.DataFrame(np.array([
        [0.5,0.5],
    ]),index=adata.obs.index,columns=pd.Index(['0-0', '1-0']))
    adata.obsm['annotation10']=adata.obsm['annotation2']
    adata.varm['profiles0']=pd.DataFrame(np.array([
        [1, 0 ],
        [0,2/3],
        [0,1/3],
    ]),index=adata.var.index,columns=pd.Index(reference.obs['type'].cat.categories.astype('category')))
    adata.varm['profiles1']=adata.varm['profiles0']
    adata.varm['profiles2']=pd.DataFrame(np.array([
        [1, 0 ],
        [0,2/3],
        [0,1/3],
    ]),index=adata.var.index,columns=pd.Index(['0', '1']))
    adata.varm['profiles10']=adata.varm['profiles2']
    adata.uns['mapping2']=pd.Series([0,1],index=['0-0', '1-0']).astype('category')
    adata.uns['mapping10']=adata.uns['mapping2']
    adata.uns['annotation_prior']=pd.Series(adata.obsm['type'].sum(axis=0).to_numpy(),index=adata.obsm['type'].columns)
    return ( adata, reference, )

@pytest.fixture(scope="session")
def adata_reference_and_typing2():
    reference = ad.AnnData(X=np.array([
        [2,0,0,0,0,0,0,0,0,0,0,0,],
        [0,2,0,0,0,0,0,0,0,0,0,0,],
        [0,0,2,0,0,0,0,0,0,0,0,0,],
        [0,0,0,2,0,0,0,0,0,0,0,0,],
        [0,0,0,0,2,0,0,0,0,0,0,0,],
        [0,0,0,0,0,2,0,0,0,0,0,0,],
        [0,0,0,0,0,0,2,0,0,0,0,0,],
        [0,0,0,0,0,0,0,2,0,0,0,0,],
        [0,0,0,0,0,0,0,0,2,0,0,0,],
        [0,0,0,0,0,0,0,0,0,2,0,0,],
        [0,0,0,0,0,0,0,0,0,0,2,0,],
        [0,0,0,0,0,0,0,0,0,0,0,2,],
    ]))
    reference.obs=pd.DataFrame({
        'type': pd.Series([0,0,0,0,1,1,1,1,2,2,2,2,],dtype='category',index=pd.Index(range(reference.shape[0])).astype(str)),
    })
    adata = ad.AnnData(X=np.array([
        [1,1,1,1,0,0,0,0,0,0,0,0,],
        [1,1,1,1,0,0,0,0,0,0,0,0,],
        [1,1,1,1,0,0,0,0,0,0,0,0,],
        [0,0,0,0,1,1,1,1,0,0,0,0,],
        [0,0,0,0,1,1,1,1,0,0,0,0,],
        [0,0,0,0,1,1,1,1,0,0,0,0,],
        [0,0,0,0,0,0,0,0,1,1,1,1,],
        [0,0,0,0,0,0,0,0,1,1,1,1,],
        [0,0,0,0,0,0,0,0,1,1,1,1,],
        [1,1,1,1,1,1,1,1,0,0,0,0,],
        [1,1,1,1,1,1,1,1,0,0,0,0,],
        [1,1,1,1,1,1,1,1,0,0,0,0,],
    ]))
    adata.obsm['type']=pd.DataFrame(np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0.5,0.5,0],
        [0.5,0.5,0],
        [0.5,0.5,0],
    ]),index=adata.obs.index,columns=pd.Index(reference.obs['type'].cat.categories,dtype=reference.obs['type'].dtype,name='type'))
    adata.obsm['annotation0']=pd.DataFrame(np.array([
        [1,0,0],
        [1,0,0],
        [1,0,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,0,1],
        [0,0,1],
        [0,0,1],
        [0.5,0.5,0],
        [0.5,0.5,0],
        [0.5,0.5,0],
    ]),index=adata.obs.index,columns=pd.Index(reference.obs['type'].cat.categories,dtype=reference.obs['type'].dtype))
    adata.obsm['annotation1'] = adata.obsm['annotation0']    
    adata.obsm['annotation2']=pd.DataFrame(np.array([
        [1/2,1/2, 0 , 0 , 0 , 0 ],
        [1/2,1/2, 0 , 0 , 0 , 0 ],
        [1/2,1/2, 0 , 0 , 0 , 0 ],
        [ 0 , 0 ,1/2,1/2, 0 , 0 ],
        [ 0 , 0 ,1/2,1/2, 0 , 0 ],
        [ 0 , 0 ,1/2,1/2, 0 , 0 ],
        [ 0 , 0 , 0 , 0 ,1/2,1/2],
        [ 0 , 0 , 0 , 0 ,1/2,1/2],
        [ 0 , 0 , 0 , 0 ,1/2,1/2],
        [1/4,1/4,1/4,1/4, 0 , 0 ],
        [1/4,1/4,1/4,1/4, 0 , 0 ],
        [1/4,1/4,1/4,1/4, 0 , 0 ],
    ]),index=adata.obs.index,columns=pd.Index(['0-0', '0-1', '1-0', '1-1', '2-0', '2-1']))
    adata.obsm['annotation10']=pd.DataFrame(np.array([
        [1/4,1/4,1/4,1/4, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [1/4,1/4,1/4,1/4, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [1/4,1/4,1/4,1/4, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 ,1/4,1/4,1/4,1/4, 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 ,1/4,1/4,1/4,1/4, 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 ,1/4,1/4,1/4,1/4, 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/4,1/4,1/4,1/4],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/4,1/4,1/4,1/4],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/4,1/4,1/4,1/4],
        [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8, 0 , 0 , 0 , 0 ],
        [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8, 0 , 0 , 0 , 0 ],
        [1/8,1/8,1/8,1/8,1/8,1/8,1/8,1/8, 0 , 0 , 0 , 0 ],
    ]),index=adata.obs.index,columns=pd.Index(['0-0', '0-1', '0-2', '0-3', '1-0', '1-1', '1-2', '1-3', '2-0', '2-1', '2-2', '2-3']))
    adata.varm['profiles0']=pd.DataFrame(np.array([
        [1/4, 0 , 0 ],
        [1/4, 0 , 0 ],
        [1/4, 0 , 0 ],
        [1/4, 0 , 0 ],
        [ 0 ,1/4, 0 ],
        [ 0 ,1/4, 0 ],
        [ 0 ,1/4, 0 ],
        [ 0 ,1/4, 0 ],
        [ 0 , 0 ,1/4],
        [ 0 , 0 ,1/4],
        [ 0 , 0 ,1/4],
        [ 0 , 0 ,1/4],
    ],dtype=np.float32),index=adata.var.index,columns=pd.Index(reference.obs['type'].cat.categories.astype('category')))
    adata.varm['profiles1']=adata.varm['profiles0']
    adata.varm['profiles2']=pd.DataFrame(np.array([
        [1/2, 0 , 0 , 0 , 0 , 0 ],
        [1/2, 0 , 0 , 0 , 0 , 0 ],
        [ 0 ,1/2, 0 , 0 , 0 , 0 ],
        [ 0 ,1/2, 0 , 0 , 0 , 0 ],
        [ 0 , 0 ,1/2, 0 , 0 , 0 ],
        [ 0 , 0 ,1/2, 0 , 0 , 0 ],
        [ 0 , 0 , 0 ,1/2, 0 , 0 ],
        [ 0 , 0 , 0 ,1/2, 0 , 0 ],
        [ 0 , 0 , 0 , 0 ,1/2, 0 ],
        [ 0 , 0 , 0 , 0 ,1/2, 0 ],
        [ 0 , 0 , 0 , 0 , 0 ,1/2],
        [ 0 , 0 , 0 , 0 , 0 ,1/2],
    ],dtype=np.float32),index=adata.var.index,columns=pd.Index(['0-0', '0-1', '1-0', '1-1', '2-0', '2-1']))
    adata.varm['profiles10']=pd.DataFrame(np.array([
        [1/1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 ,1/1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 ,1/1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 ,1/1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 ,1/1, 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 ,1/1, 0 , 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 ,1/1, 0 , 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/1, 0 , 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/1, 0 , 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/1, 0 , 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/1, 0 ],
        [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,1/1],
    ],dtype=np.float32),index=adata.var.index,columns=pd.Index(['0-0', '0-1', '0-2', '0-3', '1-0', '1-1', '1-2', '1-3', '2-0', '2-1', '2-2', '2-3']))
    adata.uns['mapping2']=pd.Series([0,0,1,1,2,2],index=['0-0', '0-1', '1-0', '1-1', '2-0', '2-1']).astype('category')
    adata.uns['mapping10']=pd.Series([0,0,0,0,1,1,1,1,2,2,2,2],index=['0-0', '0-1', '0-2', '0-3', '1-0', '1-1', '1-2', '1-3', '2-0', '2-1', '2-2', '2-3']).astype('category')
    adata.uns['annotation_prior']=pd.Series(adata.obsm['type'].sum(axis=0).to_numpy(),index=adata.obsm['type'].columns)
    return ( adata, reference, )

@pytest.fixture(scope="session")
def adata_reference_and_typing(adata_reference_and_typing0, adata_reference_and_typing1, adata_reference_and_typing2):
    return [
        adata_reference_and_typing0,
        adata_reference_and_typing1,
        adata_reference_and_typing2,
    ]

@pytest.mark.filterwarnings("ignore:Transforming to str index.")
@pytest.mark.parametrize('dataset', [0,]) # does not work for single observation data or for "composite" types
def test_annotate_NMFreg(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='NMFreg', K=3)

    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize('dataset', [0,1,2,])
def test_annotate_nnls(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='nnls')

    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize('dataset', [0,1,2,])
@pytest.mark.parametrize('decomposition', [False,True])
@pytest.mark.parametrize('metric', ['bc','bc2'])
def test_annotate_OT(adata_reference_and_typing, dataset, decomposition, metric):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']
    annotation_prior=adata.uns['annotation_prior']
    
    result = tc.tl.annotate(adata, reference, annotation_key='type', method='OT', bisections=0, annotation_prior=annotation_prior, metric=metric, decomposition=decomposition)

    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize('dataset', [0,1,2,])
@pytest.mark.parametrize('deconvolution', ['linear','nnls'])
@pytest.mark.parametrize('metric', ['naive','bc','bc2'])
def test_annotate_OT_deconvolution(adata_reference_and_typing, dataset, deconvolution, metric):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']
    annotation_prior=adata.uns['annotation_prior']
    
    result = tc.tl.annotate(adata, reference, annotation_key='type', method='OT', bisections=0, annotation_prior=annotation_prior, metric=metric, deconvolution=deconvolution)

    tc.testing.assert_frame_equal(result, typing, rtol=1e-6, atol=1e-14)

@pytest.mark.parametrize('dataset', [0,2,]) # does not work for mixed dataset 1 due to rounding errors
def test_annotate_OT_max(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']
    annotation_prior=adata.uns['annotation_prior']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='OT', bisections=0, annotation_prior=annotation_prior, max_annotation=1)
    
    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=1e-14)

@pytest.mark.parametrize('dataset', [1,2,]) # does not make sense for degenerate types
@pytest.mark.parametrize('multi_center', [0,1,2,10])
@pytest.mark.parametrize('multi_center_amplitudes', [True,False])
@pytest.mark.parametrize('reconstruction_key', [None,'reconstruction'])
def test_annotate_OT_multi_center(adata_reference_and_typing, dataset, multi_center, multi_center_amplitudes, reconstruction_key):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']
    annotation_prior=adata.uns['annotation_prior']
    
    adata = adata.copy() # dont change the input
    
    tc.tl.annotate(adata, reference, annotation_key='type', method='OT', bisections=0, annotation_prior=annotation_prior, multi_center=multi_center, multi_center_amplitudes=multi_center_amplitudes, result_key='annotation', reconstruction_key=reconstruction_key)
    result = adata.obsm['annotation']

    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=1e-14)
    
    if reconstruction_key is not None and multi_center < 10: # the overclustered case is not reproducible across platforms, as the scanpy.pp.pca is not
        # for pca non-reproducibility, see https://github.com/scverse/scanpy/issues/1187
        # maybe the choice of non-stable sort algorithm quicksort here is the source of it: https://github.com/scipy/scipy/blob/v1.8.0/scipy/sparse/linalg/_eigen/_svds.py#L388

        # can only assert some subclustering metadata as the specific subclustering is not identical for the different choices of multi_center_amplitudes
        annotation = adata.obsm[f'annotation{multi_center}']
        result_annotation = adata.obsm[reconstruction_key]
        tc.testing.assert_index_equal(result_annotation.index, annotation.index, rtol=1e-14, atol=1e-14) 
        tc.testing.assert_index_equal(result_annotation.columns, annotation.columns, rtol=1e-14, atol=1e-14)
        
        profiles = adata.varm[f'profiles{multi_center}']
        result_profiles = adata.varm[reconstruction_key]
        tc.testing.assert_index_equal(result_profiles.index, result_profiles.index, rtol=1e-14, atol=1e-14)
        tc.testing.assert_index_equal(result_profiles.columns, result_profiles.columns, rtol=1e-14, atol=1e-14)
        
        if multi_center > 1:
            mapping = adata.uns[f'mapping{multi_center}']
            result_mapping = adata.uns[reconstruction_key]
            tc.testing.assert_series_equal(result_mapping, mapping, rtol=1e-14, atol=1e-14)
        else:
            assert reconstruction_key not in adata.uns

@pytest.mark.parametrize('dataset', [0,1,2,])
@pytest.mark.parametrize('projection', ['naive','bc','bc2','weighted','nwsp'])
@pytest.mark.parametrize('bisections', [0,3])
@pytest.mark.parametrize('deconvolution', [False,'nnls','linear'])
def test_annotate_projection(adata_reference_and_typing, dataset, projection, bisections, deconvolution):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='projection', bisections=bisections, projection=projection, deconvolution=deconvolution)

    if dataset == 1 and deconvolution == 0 and projection == 'naive': # this method has huge error on the this dataset
        tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=2e-1)
    else:
        tc.testing.assert_frame_equal(result, typing, rtol=1e-7, atol=1e-14)

@pytest.mark.parametrize('dataset', [0,2,]) # does not work for single observation data
@pytest.mark.parametrize('mode', ['classification','regression'])
def test_annotate_SVM(adata_reference_and_typing, dataset, mode):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='svm', mode=mode, platform_iterations=None)

    if mode == 'classification':
        # classification can only find a single type per cell
        single_types = (typing.to_numpy() != 0).sum(axis=1) == 1
        result = result.iloc[single_types]
        typing = typing.iloc[single_types]
    
    tc.testing.assert_frame_equal(result, typing, rtol=1e-4, atol=1e-4)

@pytest.mark.skip(reason="RCTD environment is optional")
@pytest.mark.parametrize('dataset', [0,2,]) # does not work for single observation data
@pytest.mark.parametrize('doublet', ['doublet','full'])
def test_annotate_RCTD(adata_reference_and_typing, dataset, doublet):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='RCTD', min_ct=1, verbose=False, platform_iterations=None, doublet=doublet, conda_env='RCTD')

    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=2e-3)

@pytest.mark.skip(reason="tangram environment is optional")
@pytest.mark.parametrize('dataset', [0,1,2,])
def test_annotate_tangram(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='tangram', verbose=True, platform_iterations=None, conda_env='tangram')

    tc.testing.assert_frame_equal(result, typing, rtol=4e-2, atol=4e-2) # tangram gets not reproducible results with quite some distance from the ground truth

@pytest.mark.skip(reason="SingleR environment is optional")
@pytest.mark.parametrize('dataset', [0,1,2,])
def test_annotate_SingleR(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='SingleR', verbose=True, platform_iterations=None, conda_env='SingleR')

    # SingleR can only find a single type per cell
    single_types = (typing.to_numpy() != 0).sum(axis=1) == 1
    result = result.iloc[single_types]
    typing = typing.iloc[single_types]
    
    tc.testing.assert_frame_equal(result, typing, rtol=1e-14, atol=2e-3)

@pytest.mark.skipif(not tc.tl._wot.HAVE_WOT, reason="WOT is optional")
@pytest.mark.parametrize('dataset', [0,1,2,])
def test_annotate_WOT(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='WOT', verbose=True, platform_iterations=None)

    tc.testing.assert_frame_equal(result, typing, rtol=1e-7, atol=2e-2)

@pytest.mark.skipif(not tc.tl._novosparc.HAVE_NOVOSPARC, reason="NovoSpaRc is optional")
@pytest.mark.parametrize('dataset', [0,2,]) # does not work for single observation data
def test_annotate_NovoSpaRc(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']

    result = tc.tl.annotate(adata, reference, annotation_key='type', method='novosparc', verbose=True, platform_iterations=None)

    tc.testing.assert_frame_equal(result, typing, rtol=1e-7, atol=3.4e-1) # exact marginal enforcement by optimal transport makes a better result impossible...

@pytest.mark.parametrize('dataset', [0,1,2,])
@pytest.mark.skipif(not tc.benchmarking._benchmarking.BENCHMARKING_AVAILABLE, reason='Benchmarking not available on this system')
def test_benchmark_annotate(adata_reference_and_typing, dataset):
    adata, reference = adata_reference_and_typing[dataset]
    typing = adata.obsm['type']
    annotation_prior=adata.uns['annotation_prior']
    
    result = tc.benchmarking.benchmark_annotate(adata, reference, annotation_key='type', method='OT', bisections=0, annotation_prior=annotation_prior)

    tc.testing.assert_frame_equal(result['annotation'], typing, rtol=1e-14, atol=1e-14)

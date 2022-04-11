import pytest
import numpy as np
import pandas as pd
import anndata as ad
import tacco as tc
from tacco.tools import _enrichments as tc_stat

@pytest.fixture(scope="session")
def dataframe_adata_and_enrichments():
    df = pd.DataFrame({
        'group':  pd.Series([0,0,1,1,1],dtype='category'),
        'cat':  pd.Series([5,5,6,6,6],dtype='category'),
        'cont1':  pd.Series([5,5,6,6,6],dtype=np.float64),
        'cont2':  pd.Series([2,4,7,8,9],dtype=np.float64),
        'sample': pd.Series([1,1,2,3,3],dtype='category'),
    })
    enrichments = {
        'fisher_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_fisher':pd.Series([0.1,1.0,1.0,0.1,1.0,0.1,0.1,1.0],dtype=float),
            'p_fisher_fdr_bh':pd.Series([0.2,1.0,1.0,0.2,1.0,0.2,0.2,1.0],dtype=float),
            }),
        't_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_t':pd.Series([0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0],dtype=float),
            'p_t_fdr_bh':pd.Series([0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0],dtype=float),
            }),
        'welch_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_welch':pd.Series([0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0],dtype=float),
            'p_welch_fdr_bh':pd.Series([0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0],dtype=float),
            }),
        'welchl_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_welch':pd.Series([0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0],dtype=float),
            'p_welch_fdr_bh':pd.Series([0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0],dtype=float),
            }),
        'mwuc_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([0.1,1.0,1.0,0.1,1.0,0.1,0.1,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([0.2,1.0,1.0,0.2,1.0,0.2,0.2,1.0],dtype=float),
            }),
        'mwucs_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([0.1,1.0,1.0,0.1,1.0,0.1,0.1,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([0.2,1.0,1.0,0.2,1.0,0.2,0.2,1.0],dtype=float),
            }),
        'mwus_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1/3,1.0,1.0,1/3,1.0,1/3,1/3,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([2/3,1.0,1.0,2/3,1.0,2/3,2/3,1.0],dtype=float),
            }),
        'mwug_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1/3,1.0,1.0,1/3,1.0,1/3,1/3,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([2/3,1.0,1.0,2/3,1.0,2/3,2/3,1.0],dtype=float),
            }),
        'mwul_cat': pd.DataFrame({
            'cat':pd.Series([5,5,5,5,6,6,6,6],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1/3,1.0,1.0,1/3,1.0,1/3,1/3,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([2/3,1.0,1.0,2/3,1.0,2/3,2/3,1.0],dtype=float),
            }),
        't_cont1': pd.DataFrame({
            'cont1':pd.Series(['cont1','cont1','cont1','cont1'],dtype='category'),
            'group':pd.Series([0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified'],dtype='category'),
            'p_t':pd.Series([1.0,0.0,0.0,1.0],dtype=float),
            'p_t_fdr_bh':pd.Series([1.0,0.0,0.0,1.0],dtype=float),
            }),
        'welch_cont1': pd.DataFrame({
            'cont1':pd.Series(['cont1','cont1','cont1','cont1'],dtype='category'),
            'group':pd.Series([0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified'],dtype='category'),
            'p_welch':pd.Series([1.0,0.0,0.0,1.0],dtype=float),
            'p_welch_fdr_bh':pd.Series([1.0,0.0,0.0,1.0],dtype=float),
            }),
        'welchl_cont1': pd.DataFrame({
            'cont1':pd.Series(['cont1','cont1','cont1','cont1'],dtype='category'),
            'group':pd.Series([0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified'],dtype='category'),
            'p_welch':pd.Series([1.0,0.0,0.0,1.0],dtype=float),
            'p_welch_fdr_bh':pd.Series([1.0,0.0,0.0,1.0],dtype=float),
            }),
        'mwuc_cont1': pd.DataFrame({
            'cont1':pd.Series(['cont1','cont1','cont1','cont1'],dtype='category'),
            'group':pd.Series([0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1.0,0.1,0.1,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([1.0,0.2,0.2,1.0],dtype=float),
            }),
        't_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_t':pd.Series([1.0,0.0,0.0,1.0,0.9911140491029843, 0.008885950897015736, 0.008885950897015736, 0.9911140491029843],dtype=float),
            'p_t_fdr_bh':pd.Series([1.0,0.0,0.0,1.0,1.0,0.017771901794031472, 0.017771901794031472,1.0],dtype=float),
            }),
        'welch_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_welch':pd.Series([1.0,0.0,0.0,1.0,0.966751983129323, 0.03324801687067709, 0.03324801687067709, 0.966751983129323],dtype=float),
            'p_welch_fdr_bh':pd.Series([1.0,0.0,0.0,1.0,1.0,0.06649603374135418, 0.06649603374135418,1.0],dtype=float),
            }),
        'welchl_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_welch':pd.Series([0.1050634538357276,0.8949365461642724,0.8949365461642724,0.1050634538357276,0.8949365461642722,0.1050634538357278,0.1050634538357278,0.8949365461642722],dtype=float),
            'p_welch_fdr_bh':pd.Series([0.2101269076714557,0.8949365461642724,0.8949365461642724,0.2101269076714557,0.8949365461642724,0.2101269076714557,0.2101269076714557,0.8949365461642724],dtype=float),
            
        }),
        'mwuc_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1.0,0.1,0.1,1.0,1.0,0.1,0.1,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([1.0,0.2,0.2,1.0,1.0,0.2,0.2,1.0],dtype=float),
            }),
        'mwucs_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([0.1,1.0,1.0,0.1,1.0,0.1,0.1,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([0.2,1.0,1.0,0.2,1.0,0.2,0.2,1.0],dtype=float),
            }),
        'mwus_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1/3,1.0,1.0,1/3,1.0,1/3,1/3,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([2/3,1.0,1.0,2/3,1.0,2/3,2/3,1.0],dtype=float),
            }),
        'mwug_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1/3,1.0,1.0,1/3,1.0,1/3,1/3,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([2/3,1.0,1.0,2/3,1.0,2/3,2/3,1.0],dtype=float),
            }),
        'mwul_cont': pd.DataFrame({
            'cont':pd.Series(['cont1','cont1','cont1','cont1','cont2','cont2','cont2','cont2'],dtype='category'),
            'group':pd.Series([0,0,1,1,0,0,1,1],dtype='category'),
            'enrichment':pd.Series(['enriched','purified','enriched','purified','enriched','purified','enriched','purified'],dtype='category'),
            'p_mwu':pd.Series([1/3,1.0,1.0,1/3,1.0,1/3,1/3,1.0],dtype=float),
            'p_mwu_fdr_bh':pd.Series([2/3,1.0,1.0,2/3,1.0,2/3,2/3,1.0],dtype=float),
            }),
    }
    
    df = df.copy()
    df.index = df.index.astype(str)
    
    adata = ad.AnnData(df[['cont1','cont2']].to_numpy(), dtype=float, obs=df, var=df.loc[[],['cont1','cont2']].T)
    adata.obsm['cont']=df[['cont1','cont2']]
    
    return df, adata, enrichments

def _test_enrichments(dataframe_adata_and_enrichments,method,cont_cat,value_location):
    df = dataframe_adata_and_enrichments[0]
    adata = dataframe_adata_and_enrichments[1]
    cont_cat_str = cont_cat.name if hasattr(cont_cat,'name') else cont_cat
    enrichment = dataframe_adata_and_enrichments[2][method + '_' + cont_cat_str]
    
    reduction='sum'
    normalization=None
    sample_key='sample'
    if method == 'fisher':
        reduction='sum'
        normalization=None
        sample_key=None
    elif method == 't':
        reduction=None
        normalization=None
    elif method == 'welch':
        reduction=None
        normalization=None
    elif method == 'welchl':
        reduction=None
        normalization='clr'
        method='welch'
    elif method == 'mwuc':
        reduction=None
        normalization=None
        method='mwu'
    elif method == 'mwucs':
        reduction=None
        normalization='sum'
        method='mwu'
    elif method == 'mwus':
        reduction='sum'
        normalization='sum'
        method='mwu'
    elif method == 'mwug':
        reduction='sum'
        normalization='gmean'
        method='mwu'
    elif method == 'mwul':
        reduction='sum'
        normalization='clr'
        method='mwu'
    else:
        raise ValueError(f'method "{method}" is unknown!')
    
    if value_location == 'df':
        result = tc_stat.enrichments(df, cont_cat, 'group', method=method, reduction=reduction, normalization=normalization, sample_key=sample_key)
    else:
        result = tc_stat.enrichments(adata, cont_cat, 'group', method=method, reduction=reduction, normalization=normalization, sample_key=sample_key, value_location=value_location)

    tc.testing.assert_frame_equal(result, enrichment, rtol=1e-7, atol=0)

@pytest.mark.parametrize('method', ['fisher','mwuc','mwucs','mwus','mwug','mwul','t','welch','welchl'])
@pytest.mark.parametrize('value_location', ['df','obs'])
def test_enrichments_cat(dataframe_adata_and_enrichments,method,value_location):
    _test_enrichments(dataframe_adata_and_enrichments,method,'cat',value_location)

@pytest.mark.parametrize('method', ['mwuc','t','welch','welchl'])
@pytest.mark.parametrize('value_location', ['df','obs','X'])
def test_enrichments_cont1(dataframe_adata_and_enrichments,method,value_location):
    _test_enrichments(dataframe_adata_and_enrichments,method,'cont1',value_location)

@pytest.mark.parametrize('method', ['mwuc','mwucs','mwus','mwug','mwul','t','welch','welchl'])
@pytest.mark.parametrize('value_location', ['df','obs','X'])
def test_enrichments_cont12(dataframe_adata_and_enrichments,method,value_location):
    _test_enrichments(dataframe_adata_and_enrichments,method,pd.Series(['cont1','cont2'],name='cont'),value_location)

@pytest.mark.parametrize('method', ['mwuc','mwucs','mwus','mwug','mwul','t','welch','welchl'])
@pytest.mark.parametrize('value_location', ['obsm'])
def test_enrichments(dataframe_adata_and_enrichments,method,value_location):
    _test_enrichments(dataframe_adata_and_enrichments,method,'cont',value_location)

@pytest.fixture(scope="session")
def categorical_dataframe_and_contributions():
    df = pd.DataFrame({
        'group':  pd.Series([0,0,1,1,1,2,2,2],dtype='category'),
        'value':  pd.Series([5,5,6,6,6,5,6,7],dtype='category'),
        'sample': pd.Series([1,1,2,3,3,3,4,4],dtype='category'),
    })
    contributions = {
        'aggregate_sum': pd.DataFrame([
                [2,0,0],
                [0,3,0],
                [1,1,1],
            ],
            index=pd.Series([0,1,2],dtype='category',name='group'),
            columns=pd.Series([5,6,7],dtype='category',name='value'),
            ),
        'aggregate_sum_restrict_groups': pd.DataFrame([
                [2,0,0],
                [0,3,0],
            ],
            index=pd.Series([0,1],dtype='category',name='group'),
            columns=pd.Series([5,6,7],dtype='category',name='value'),
            ),
        'aggregate_sum_restrict_values': pd.DataFrame([
                [2,0],
                [0,3],
                [1,1],
            ],
            index=pd.Series([0,1,2],dtype='category',name='group'),
            columns=pd.Series([5,6],dtype='category',name='value'),
            ),
    }

    return df, contributions

def test_contributions_aggregate_sum(categorical_dataframe_and_contributions):
    df = categorical_dataframe_and_contributions[0]
    contribution = categorical_dataframe_and_contributions[1]['aggregate_sum']

    result = tc_stat.get_contributions(df, 'value', 'group', normalization=None)

    tc.testing.assert_frame_equal(result, contribution, rtol=1e-14, atol=0)

def test_contributions_aggregate_sum_restrict_groups(categorical_dataframe_and_contributions):
    df = categorical_dataframe_and_contributions[0]
    contribution = categorical_dataframe_and_contributions[1]['aggregate_sum_restrict_groups']

    result = tc_stat.get_contributions(df, 'value', 'group', normalization=None, restrict_groups=df['group'].unique()[:2])

    tc.testing.assert_frame_equal(result, contribution, rtol=1e-14, atol=0)


def test_contributions_aggregate_sum_restrict_values(categorical_dataframe_and_contributions):
    df = categorical_dataframe_and_contributions[0]
    contribution = categorical_dataframe_and_contributions[1]['aggregate_sum_restrict_values']

    result = tc_stat.get_contributions(df, 'value', 'group', normalization=None, restrict_values=df['value'].unique()[:2])

    tc.testing.assert_frame_equal(result, contribution, rtol=1e-14, atol=0)

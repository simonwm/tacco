import pytest
import numpy as np
import tacco as tc
import scipy.sparse
import pandas as pd

@pytest.fixture(scope="session")
def cosort():
    np.random.seed(42)
    arr = np.random.uniform(-10,100,250) # avoid equality collisions with undefined true sorting
    co1 = np.random.randint(-10,100,250)
    co2 = -arr

    ids = np.argsort(arr)

    sarr = arr[ids]
    sco1 = co1[ids]
    sco2 = co2[ids]
    return arr, co1, co2, sarr, sco1, sco2

@pytest.fixture(scope="session")
def csr2coo():
    coo = scipy.sparse.random(101,232,random_state=42)
    csr = coo.tocsr()
    return coo, csr

def test_cosort(cosort):
    arr, co1, co2, sarr, sco1, sco2 = cosort

    arr, co1, co2 = arr.copy(),co1.copy(),co2.copy() # dont change the input

    tc.utils.heapsort3(arr,co1,co2)

    tc.testing.assert_dense_equal(arr, sarr)
    tc.testing.assert_dense_equal(co1, sco1)
    tc.testing.assert_dense_equal(co2, sco2)

def test_coo_tocsr_inplace(csr2coo):
    coo, csr = csr2coo

    coo = coo.copy() # dont change the input

    csr2 = tc.utils.coo_tocsr_inplace(coo)

    tc.testing.assert_sparse_equal(csr2, csr)

@pytest.fixture(scope="session")
def data2split():
    np.random.seed(42)
    x = np.concatenate([
        np.random.uniform(0+s//4,1+s//4,100)
        for s in range(12)
    ])
    y = np.concatenate([
        np.random.uniform(0+s%4,1+s%4,100)
        for s in range(12)
    ])
    splitx = np.concatenate([
        np.full(fill_value=s//4,shape=100)
        for s in range(12)
    ])
    splity = np.concatenate([
        np.full(fill_value=s%4,shape=100)
        for s in range(12)
    ])
    df = pd.DataFrame({'x':x,'y':y,'split':pd.Series(splitx.astype(str))+'_'+splity.astype(str)})
    return df

def test_spatial_split(data2split):
    df = data2split
    
    result = tc.utils.spatial_split(df, position_split=(3,4))

    cmp = pd.DataFrame({'result':result,'baseline':data2split['split']})
    
    assert((cmp.value_counts() == 100).all())

import pytest
import numpy as np
import tacco as tc
import scipy.sparse

@pytest.fixture(scope="session")
def division():
    return (np.array([
        [0,0,4,0],
        [2,0,0,0],
        [0,1,0,0],
    ],dtype=float),
    np.array([
        [5,2,2,3],
        [1,3,4,1],
        [2,1,2,1],
    ],dtype=float),
    np.array([
        [0,0,2,0],
        [2,0,0,0],
        [0,1,0,0],
    ],dtype=float),)

@pytest.fixture(scope="session")
def gemmT():
    return (np.array([
        [0,0,4,0],
        [2,0,0,0],
        [0,1,0,0],
    ],dtype=float),
    np.array([
        [5,2,0,0],
        [0,0,0,1],
        [2,0,0,0],
    ],dtype=float),
    np.array([
        [ 0,0,0],
        [10,0,4],
        [ 2,0,0],
    ],dtype=float),)

@pytest.fixture(scope="session")
def row_scale():
    return (np.array([
        [0,0,4,0],
        [2,0,0,0],
        [0,1,0,0],
    ],dtype=float),
    np.array([
        [0,0,4,0],
        [4,0,0,0],
        [0,3,0,0],
    ],dtype=float),
    np.array([
        1,
        2,
        3,
    ],dtype=float),)

@pytest.fixture(scope="session")
def col_scale():
    return (np.array([
        [0,0,4,0],
        [2,0,0,0],
        [0,1,0,0],
    ],dtype=float),
    np.array([
        [0,0,12,0],
        [2,0, 0,0],
        [0,2, 0,0],
    ],dtype=float),
    np.array([
        1,2,3,4,
    ],dtype=float),)

@pytest.fixture(scope="session")
def log1p():
    return (np.array([
        [0,0,4,0],
        [2,0,0,0],
        [0,1,0,0],
    ],dtype=float),
    np.array([
        [        0,        0,np.log(5),0],
        [np.log(3),        0,        0,0],
        [        0,np.log(2),        0,0],
    ],dtype=float),)

def test_division(division):
    a,b,c = division

    result = tc.utils.divide(a,b)

    tc.testing.assert_dense_equal(result, c)
        
@pytest.mark.parametrize('Acol', ['singleAcol','allAcols'])
@pytest.mark.parametrize('Bcol', ['singleBcol','allBcols'])
@pytest.mark.parametrize('Asparse', ['sparseA','denseA'])
@pytest.mark.parametrize('Bsparse', ['sparseB','denseB'])
def test_gemmT(gemmT, Acol, Bcol, Asparse, Bsparse,):
    A,B,C = gemmT
    
    if Acol == 'singleAcol':
        A = A[:1]
        C = C[:1]
    if Bcol == 'singleBcol':
        B = B[:1]
        C = C[:,:1]
            
    if Asparse == 'sparseA':
        A = scipy.sparse.csr_matrix(A)
    if Bsparse == 'sparseB':
        B = scipy.sparse.csr_matrix(B)

    result = tc.utils.gemmT(A,B)

    tc.testing.assert_dense_equal(result, C)

def test_sparse_result_gemmT(gemmT):
    A,B,C = gemmT

    spC = scipy.sparse.coo_matrix(C)

    result = tc.utils.sparse_result_gemmT(A,B,spC,inplace=False)

    tc.testing.assert_sparse_equal(result, spC)

@pytest.mark.parametrize('input_format', ['csr', 'csc', 'coo'])
def test_sparse_result_gemmT_no_corruption(input_format):
    """Test that sparse_result_gemmT doesn't corrupt input matrices."""
    # Create test data similar to the real bug scenario
    np.random.seed(42)
    cell_type = np.random.rand(10, 5).astype(np.float64)
    average_profiles = np.random.rand(100, 5).astype(np.float32)
    
    # Create input matrix in specified format
    sparse_out = scipy.sparse.random(10, 100, density=0.1, format=input_format, dtype=np.float32)
    
    # Store original matrix for comparison
    original_sparse_out = sparse_out.copy()
    
    # Call the function with inplace=False (should NOT modify input)
    output = tc.utils.sparse_result_gemmT(cell_type, average_profiles, sparse_out, inplace=False)
    
    # Verify input was not modified using tacco's testing function
    tc.testing.assert_sparse_equal(sparse_out, original_sparse_out)

def test_row_scale(row_scale):
    A,B,s = row_scale

    result = A.copy()
    tc.utils.row_scale(result,s)

    tc.testing.assert_dense_equal(result, B)

    spA = scipy.sparse.csr_matrix(A)
    spB = scipy.sparse.csr_matrix(B)

    result = spA.copy()
    tc.utils.row_scale(result,s)

    tc.testing.assert_sparse_equal(result, spB)

def test_col_scale(col_scale):
    A,B,s = col_scale

    result = A.copy()
    tc.utils.col_scale(result,s)

    tc.testing.assert_dense_equal(result, B)

    spA = scipy.sparse.csr_matrix(A)
    spB = scipy.sparse.csr_matrix(B)

    result = spA.copy()
    tc.utils.col_scale(result,s)

    tc.testing.assert_sparse_equal(result, spB)

def test_log1p(log1p):
    A,B = log1p

    result = A.copy()
    tc.utils.log1p(result)

    tc.testing.assert_dense_equal(result, B)

    spA = scipy.sparse.csr_matrix(A)
    spB = scipy.sparse.csr_matrix(B)

    result = spA.copy()
    tc.utils.log1p(result)

    tc.testing.assert_sparse_equal(result, spB)


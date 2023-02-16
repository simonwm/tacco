"""
Specialized functions to support unit tests
"""

# expose the API
from ._assert import assert_sparse_equal, assert_dense_equal, assert_frame_equal, assert_series_equal, assert_adata_equal, assert_index_equal, assert_tuple_equal
from ._freeze import string_encode, string_decode

"""
General utility functions
"""

# expose the API
from ._utils import *
from .math import divide, sparse_result_gemmT, gemmT, row_scale, col_scale, log1p, log, sqrt, get_sum, cast_down_common
from .dist import cdist, sparse_distance_matrix, projection, bin, hash, min_dtype, dense_distance_matrix

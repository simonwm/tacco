"""
TACCO: Transfer of Annotations to Cells and their COmbinations
"""

# expose the API
from . import plots as pl
from . import preprocessing as pp
from . import tools as tl
from . import eval as ev
from . import utils
from . import testing
from . import benchmarking
# expose frequently used functions on top level for convenience
from .utils import get_sum as sum

# the standard numbda thread count is the number of vcpus - which is not good on shared systems.
import numba
try:
    numba.set_num_threads(utils.cpu_count())
except ValueError: # within joblib jobs this can blow up if nested parallelism is not allowed
    pass
del numba

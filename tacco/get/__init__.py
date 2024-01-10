"""
Accessors for frequently used data within an :class:`~anndata.AnnData`.
"""

# expose the API
from ._data import get_data_from_key as data_from_key
from ._counts import get_counts as counts
from ._positions import get_positions as positions

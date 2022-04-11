"""
Accessors for frequently used data within an :class:`~anndata.AnnData`.
"""

# expose the API
from ._counts import get_counts as counts
from ._positions import get_positions as positions

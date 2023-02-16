"""
Data preprocessing functions
"""

# expose the API
from ._reference import construct_reference_profiles, refine_reference
from ._qc import filter, filter_profiles, filter_annotation, filter_reference, check_counts_validity, filter_reference_genes
from ._platform import normalize_platform, apply_random_platform_effect, subsample_annotation

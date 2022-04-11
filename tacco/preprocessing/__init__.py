"""
Data preprocessing functions
"""

# expose the API
from ._create_reference import create_reference, normalize_reference, infer_annotation_key, get_unique_keys
from ._refine_reference import construct_reference_profiles, construct_reference_annotation, split_reference, refine_reference
from ._qc import filter, filter_profiles, filter_annotation, filter_reference, check_counts_validity, filter_reference_genes
from ._platform import normalize_platform, apply_random_platform_effect, subsample_annotation

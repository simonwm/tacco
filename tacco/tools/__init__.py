"""
Tools for annotation transfer and related tasks
"""

# expose the API
from ._find_regions import find_regions, fill_regions
from ._co_occurrence import co_occurrence, co_occurrence_matrix, annotation_coordinate
from ._split import split_observations, merge_observations
from ._annotate import annotate, annotate_single_molecules, segment
from ._OT import annotate_OT
from ._nnls import annotate_nnls
from ._svm import annotate_svm
from ._tangram import annotate_tangram
from ._novosparc import annotate_novosparc
from ._projection import annotate_projection
from ._enrichments import enrichments, get_contributions, get_compositions
from ._in_silico import mix_in_silico
from ._points import distance_matrix, affinity, spectral_clustering, distribute_molecules
from ._NMFreg import annotate_NMFreg
from ._RCTD import annotate_RCTD
from ._wot import annotate_wot
from ._SingleR import annotate_SingleR
from ._goa import setup_goa_analysis, run_goa_analysis
from ._orthology import setup_orthology_converter, run_orthology_converter

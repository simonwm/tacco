import tempfile
import subprocess
import os
import warnings
import pandas as pd
import numpy as np
import scipy.sparse

from .. import utils
from ..utils._utils import _anndata2R_header
from . import _helper as helper
from .. import get
from .. import preprocessing

def _round_X(adata):
    if scipy.sparse.issparse(adata.X.data):
        adata.X.data = np.round(adata.X.data)
    else:
        adata.X = np.round(adata.X)

def annotate_RCTD(
    adata,
    reference,
    annotation_key,
    counts_location=None,
    conda_env=None,
    x_coord_name='x',
    y_coord_name='y',
    doublet=False,
    min_ct=0,
    UMI_min_sigma=None,
    n_cores=None,
    working_directory=None,
    verbose=True,
):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by RCTD
    [Cable21]_.

    This is the direct interface to this annotation method. In practice using
    the general wrapper :func:`~tacco.tools.annotate` is recommended due to its
    higher flexibility.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X`.
    reference
        Reference data to get the annotation definition from.
    annotation_key
        The `.obs` key where the annotation is stored in the `reference`. If
        `None`, it is inferred from `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    conda_env
        The path of a conda environment where `RCTD` is installed and
        importable as 'library(RCTD)'.
    x_coord_name
        Name of an `.obs` column to forward to RCTD as x coordinates. If not
        available, forwards a new 0-only column.
    y_coord_name
        Name of an `.obs` column to forward to RCTD as y coordinates. If not
        available, forwards a new 0-only column.
    doublet
        Whether to run in "doublet" mode. Alternative is "full" mode.
    min_ct
        Minimum number of cells in a group to include in the RCTD run.
    UMI_min_sigma
        As default, RCTD has this value at `300`, which is quite large for some
        datasets, and breaks RCTD. Therefore the default heuristic for `None`
        here is at `min(300,median(total_counts_per_observation)-1)`. See RCTD
        docs for details about this parameter. 
    n_cores
        Number of cores to use for RCTD. If `None`, use all available cores.
    working_directory
        The directory where to store all the intermediates. If `None`, a
        temporary directory is used and cleaned in the end. This option is
        probably only relevant for debugging.
    verbose
        Whether to print stderr and stdout of the RCTD run.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    if conda_env is None:
        conda_env = '/ahg/regevdata/users/jklugham/src/anaconda3/envs/R_RCTD'
    if not os.path.exists(conda_env):
        raise Exception('The conda environment "%s" does not exist! A conda environment with a working `RCTD` setup is needed and can be supplied by the `conda_env` argument.' % (conda_env))
        
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=True)
    
    # since 1.2: RCTD requires integer input - to make it work also otherwise, round data here.
    _round_X(adata)
    _round_X(reference)
    
    if UMI_min_sigma is None:
        UMI_min_sigma = min(300,np.median(utils.get_sum(adata.X,axis=1))-1)
    
    if n_cores is None:
        n_cores = utils.cpu_count()
    if np.prod(adata.shape) < 100: # remove overhead for very small problems
        n_cores = 1
        
    mode = 'doublet' if (isinstance(doublet, bool) and doublet) or doublet == 'doublet' else 'full'
    
    tempdir = None
    if working_directory is None:
        tempdir = tempfile.TemporaryDirectory(prefix='temp_RCTD_',dir='.')
        working_directory = tempdir.name
    working_directory = working_directory + '/'
    
    data_file_name = 'data.h5ad'
    ref_file_name = 'reference.h5ad'
    result_file_namebase = 'result'
    result_file_name = result_file_namebase + '.tsv'

    R_script_file_name = 'run.R'
    R_script = _anndata2R_header() + """
args=commandArgs(trailingOnly = TRUE)

library(data.table)
if("spacexr" %in% rownames(installed.packages())) {
    library(spacexr)
    RCTD_API="spacexr"
} else if ("RCTD" %in% rownames(installed.packages())) {
    library(RCTD)
    RCTD_API="RCTD"
} else {
    stop('Neither "RCTD" nor "spacexr" seem to be available! Ensure that one of them is installed properly!')
}

sc_adata=args[1]

sp_adata=args[2]

out = args[3]

min_ct = as.numeric(args[4])
cores = as.numeric(args[5])
mode = args[6]
ct_col = args[7]
x_col = args[8]
y_col = args[9]
UMI_min_sigma = args[10]

print(args)

read_reference=function(adata_file,ct_col) {
    adata = read_adata(adata_file)
    
    meta_data = adata$obs
    cell_types = meta_data[,ct_col]
    names(cell_types) = row.names(meta_data)
    
    return(Reference(as(t(adata$X),'dgCMatrix'), as.factor(cell_types)))
}

read_spatial=function (adata_file,x_col,y_col) {
    adata = read_adata(adata_file)
    
    counts <- as(t(adata$X),'dgCMatrix')
    coords = adata$obs[,c(x_col,y_col)]
    colnames(coords)[1] = "x"
    colnames(coords)[2] = "y"
    
    return(SpatialRNA(coords, counts))
}

print('reading data')
puck=read_spatial(sp_adata,x_col, y_col)
print('reading reference')
reference=read_reference(sc_adata,ct_col)

print('running RCTD')
myRCTD <- create.RCTD(puck, reference, max_cores = cores, CELL_MIN_INSTANCE = min_ct,  UMI_min = 0, UMI_min_sigma=UMI_min_sigma) # UMI_min filter is already applied when data arrives here

myRCTD <- run.RCTD(myRCTD, doublet_mode = mode)

if (mode == "doublet") {
    full_df = myRCTD@results$results_df

    full_df$bc = row.names(full_df)
    full_df$w1 = myRCTD@results$weights_doublet[,1]
    full_df$w2 = myRCTD@results$weights_doublet[,2]

    full_df = full_df[full_df$spot_class != 'reject',]

    sub_df1 = data.frame(bc=full_df[,'bc'],type=full_df[,'first_type'],weight=full_df[,'w1'])
    sub_df2 = data.frame(bc=full_df[,'bc'],type=full_df[,'second_type'],weight=full_df[,'w2'])

    w1 = reshape(sub_df1, idvar = "bc", timevar = "type", direction = "wide")
    w2 = reshape(sub_df2, idvar = "bc", timevar = "type", direction = "wide")

    w1[is.na(w1)] = 0
    w2[is.na(w2)] = 0

    w1 = data.frame(w1, row.names='bc')
    w2 = data.frame(w2, row.names='bc')

    # ensure all columns appear in both data.frames
    for (col in colnames(w1)) {
        if (!(col %in% colnames(w2))) {
            w2[,col] = 0.0
        }
    }
    for (col in colnames(w2)) {
        if (!(col %in% colnames(w1))) {
            w1[,col] = 0.0
        }
    }

    weights = w1 + w2[names(w1)]

    names(weights) <- sub("^weight.", "", names(weights))
} else {
    weights=myRCTD@results$weights
}

if (is(weights, 'sparseMatrix')) {
    weights = as.data.frame(as.matrix(weights))
}

write.table(weights,paste0(out,".tsv"),sep="\t",quote=FALSE)
"""
    with open(working_directory + R_script_file_name, 'w') as f:
        f.write(R_script)

    new_x = x_coord_name not in adata.obs
    new_y = y_coord_name not in adata.obs
    if new_x or new_y:
        if new_x:
            adata.obs[x_coord_name] = 0
        if new_y:
            adata.obs[y_coord_name] = 0

    #print('writing data')
    utils.write_adata_x_var_obs(adata, filename=working_directory + data_file_name, compression='gzip')

    if new_x:
        del adata.obs[x_coord_name]
    if new_y:
        del adata.obs[y_coord_name]

    #print('writing reference')
    utils.write_adata_x_var_obs(reference, filename=working_directory + ref_file_name, compression='gzip')
    #print('running RCTD')
    process = subprocess.Popen('bash', shell=False, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    command = f'\
cd {working_directory}\n\
source $(dirname $(dirname $(which conda)))/bin/activate\n\
conda activate {conda_env}\n\
Rscript "{R_script_file_name}" "{ref_file_name}" "{data_file_name}" "{result_file_namebase}" {min_ct} {n_cores} "{mode}" "{annotation_key}" "{x_coord_name}" "{y_coord_name}" {UMI_min_sigma}\n\
'
    #print(command)
    out, err = process.communicate(command)
    #print('done')
    
    successful = os.path.isfile(working_directory + result_file_name)
    if verbose or not successful:
        print(out)
        print(err)
    try:
        if successful:
            type_fractions = pd.read_csv(working_directory + result_file_name, sep='\t')
        else:
            raise Exception('RCTD did not work properly!')
    finally:
        if tempdir is not None:
            tempdir.cleanup()
    
    type_fractions = helper.normalize_result_format(type_fractions, types=reference.obs[annotation_key].unique())
    
    return type_fractions

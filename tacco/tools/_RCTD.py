import tempfile
import subprocess
import os
import warnings
import pandas as pd
import numpy as np
import scipy.sparse

from .. import utils
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
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X` and
        profiles in `.varm` and/or annotation in `.obs` or `.obsm`.
    reference
        Reference data to get the annotation definition from. See e.g. 
        :func:`~tc.pp.create_reference` for options to create it.
    annotation_key
        The `.obs`, `.obsm`, and/or `.varm` key where the annotation and
        profiles are stored in the `reference`. If `None`, it is inferred from
        `reference`, if possible.
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tc.get.counts`.
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

    RCTD_R_script = os.path.dirname(os.path.abspath(__file__)) + '/RCTD.R'

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
Rscript "{RCTD_R_script}" "{ref_file_name}" "{data_file_name}" "{result_file_namebase}" {min_ct} {n_cores} "{mode}" "{annotation_key}" "{x_coord_name}" "{y_coord_name}" {UMI_min_sigma}\n\
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

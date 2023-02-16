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

def annotate_SingleR(
    adata,
    reference,
    annotation_key,
    counts_location=None,
    conda_env=None,
    fine_tune=True,
    aggr_ref=False,
    genes='de',
    de_method='classic',
    working_directory=None,
    verbose=True,
):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by SingleR
    [Aran19]_.

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
        The path of a conda environment where `SingleR` is installed and
        importable as 'library(SingleR)'.
    fine_tune
        Forwards the 'fine.tune' parameter to SingleR.
    aggr_ref
        Forwards the 'aggr.ref' parameter to SingleR.
    genes
        Forwards the 'genes' parameter to SingleR.
    de_method
        Forwards the 'de_method' parameter to SingleR.
    working_directory
        The directory where to store all the intermediates. If `None`, a
        temporary directory is used and cleaned in the end. This option is
        probably only relevant for debugging.
    verbose
        Whether to print stderr and stdout of the SingleR run.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    if conda_env is None or not os.path.exists(conda_env):
        raise Exception(f'The conda environment {conda_env!r} does not exist! A conda environment with a working `SingleR` setup is needed and can be supplied by the `conda_env` argument.')
        
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=True)
    
    tempdir = None
    if working_directory is None:
        tempdir = tempfile.TemporaryDirectory(prefix='temp_SingleR_',dir='.')
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
library(SingleR)

sc_adata=args[1]
sp_adata=args[2]
out = args[3]
annotation_key = args[4]
fine.tune = args[5]
aggr.ref = args[6]
genes = args[7]
de.method = args[8]

print(args)

print('reading data')
adata = read_adata(sp_adata)
adata.Xt <- t(adata$X)
print('reading reference')
reference = read_adata(sc_adata)
reference.Xt <- t(reference$X)

meta_data = reference$obs
cell_types=reference$obs[,annotation_key]
names(cell_types)=row.names(reference$obs)

print('running SingleR')
SingleR_result <- SingleR(test=adata.Xt, ref=reference.Xt, labels=cell_types, fine.tune=fine.tune, genes=genes, de.method=de.method, aggr.ref=aggr.ref)

write.table(cbind(row.names(SingleR_result),SingleR_result$labels),paste0(out,".tsv"),sep="\t",quote=FALSE,row.names=FALSE)
"""
    with open(working_directory + R_script_file_name, 'w') as f:
        f.write(R_script)
    
    # SingleR expects log-normalized data
    adata = utils.preprocess_single_cell_data(adata, hvg=False, scale=False, pca=False, inplace=False, min_cells=0, min_genes=0)
    reference = utils.preprocess_single_cell_data(reference, hvg=False, scale=False, pca=False, inplace=False, min_cells=0, min_genes=0)

    #print('writing data')
    utils.write_adata_x_var_obs(adata, filename=working_directory + data_file_name, compression='gzip')

    #print('writing reference')
    utils.write_adata_x_var_obs(reference, filename=working_directory + ref_file_name, compression='gzip')
    #print('running SingleR')
    process = subprocess.Popen('bash', shell=False, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
    command = f'\
cd {working_directory}\n\
source $(dirname $(dirname $(which conda)))/bin/activate\n\
conda activate {conda_env}\n\
Rscript "{R_script_file_name}" "{ref_file_name}" "{data_file_name}" "{result_file_namebase}" "{annotation_key}" "{fine_tune}" "{aggr_ref}" "{genes}" "{de_method}"\n\
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
            type_annotations = pd.read_csv(working_directory + result_file_name, sep='\t')
        else:
            raise Exception('SingleR did not work properly!')
    finally:
        if tempdir is not None:
            tempdir.cleanup()
    
    type_annotations = type_annotations.set_index(type_annotations.columns[0])
    type_fractions = pd.get_dummies(type_annotations[type_annotations.columns[0]])
    type_fractions.columns = type_fractions.columns.astype(str)
    type_fractions.columns.name = None
    type_fractions.index.name = None
    type_fractions = helper.normalize_result_format(type_fractions, types=reference.obs[annotation_key].unique())
    
    return type_fractions

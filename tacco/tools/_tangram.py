import tempfile
import subprocess
import os
import warnings
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path

from .. import utils
from . import _helper as helper
from .. import get
from .. import preprocessing

def _annotate_tangram(
    adata,
    reference,
    annotation_key=None,
    conda_env=None,
    result_file=None,
    cluster_mode=False,
    verbose=True,
    **kw_args,
    ):

    """\
    Implements the functionality of :func:`~annotate_tangram` without data
    integrity checks.
    """
    
    cell_type = run_tangram(adata, reference, annotation_key, conda_env, result_file=result_file, cluster_mode=cluster_mode, verbose=verbose, **kw_args)
    
    cell_type = helper.normalize_result_format(cell_type)
    
    return cell_type

def annotate_tangram(
    adata,
    reference,
    annotation_key=None,
    counts_location=None,
    conda_env=None,
    result_file=None,
    cluster_mode=True,
    verbose=True,
    **kw_args,
    ):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by Tangram
    [Biancalani20]_.

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
        The path of a conda environment where `tangram` is installed and
        importable as 'import tangram'.
    result_file
        The name of a file to contain additional results of `tangram` as
        `.h5ad`. If `None`, nothing except for the returned annotation is
        retained.
    cluster_mode
        Whether to use tangrams cluster mode for cluster level decomposition,
        instead of cell level decomposition.
    verbose
        Whether to print stderr and stdout of the tangram run.
    **kw_args
        Additional keyword arguments are forwarded to
        :func:`tangram.map_cells_to_space`. Interesting should be in
        particular 'device' to use a gpu. Note that the arguments 'mode' and
        'cluster_label' are should be used via specifying `cluster_mode==True`.
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`. Depending on\
    `result_h5ad` might also write additional results in a file.
    
    """
    
    if conda_env is None:
        conda_env = '/ahg/regevdata/users/smages/conda_envs/tangram'
    if not os.path.exists(conda_env):
        raise Exception('The conda environment "%s" does not exist! A conda environment with a working `tangram` setup is needed and can be supplied by the `conda_env` argument.' % (conda_env))
        
    adata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=True)

    # call typing without data integrity checks
    cell_type = _annotate_tangram(
        adata=adata,
        reference=reference,
        annotation_key=annotation_key,
        conda_env=conda_env,
        result_file=result_file,
        cluster_mode=cluster_mode,
        verbose=verbose,
        **kw_args,
    )
    
    return cell_type

def run_tangram(tdata, reference, annotation_key, conda_env, result_file=None, cluster_mode=True, verbose=False, **kw_args):
    
    with tempfile.TemporaryDirectory(prefix='temp_tangram_',dir='.') as tmpdirname:
        tmpdirname = tmpdirname + '/'
        data_file_name = 'data.h5ad'
        ref_file_name = 'reference.h5ad'
        result_file_name = 'result.h5ad'
        script_file_name = 'run_tangram.py'
        
        if result_file is not None:
            result_file_name  = str(Path(result_file).resolve())
        
        if not 'device' in kw_args:
            kw_args['device'] = 'cpu'
        
        if cluster_mode:
            kw_args['mode'] = 'clusters'
            kw_args['cluster_label'] = annotation_key
        
        script = f"""
import anndata as ad
import tangram as tg
import pandas as pd
ad_sp = ad.read_h5ad({data_file_name!r})
ad_sc = ad.read_h5ad({ref_file_name!r})
tg.pp_adatas(ad_sc, ad_sp, genes=None)
kw_args = {kw_args!r}
ad_map = tg.map_cells_to_space(ad_sc, ad_sp, **kw_args)
# fix sparse data quirks in tangram+adata+pandas...
bad_vars = ['rna_count_based_density']
for bad_var in bad_vars:
    new_col = pd.Series(ad_map.var[bad_var].to_numpy().flatten(), index=ad_map.var[bad_var].index, name=ad_map.var[bad_var].name)
    del ad_map.var[bad_var]
    ad_map.var[bad_var] = new_col
ad_map.write({result_file_name!r}, compression='gzip')
"""
        #print(script)
        
        with open(tmpdirname + script_file_name, "w") as script_file:
            script_file.write(script)
        
        #print('writing data')
        utils.write_adata_x_var_obs(tdata, filename=tmpdirname + data_file_name, compression='gzip')
        
        #print('writing reference')
        utils.write_adata_x_var_obs(reference, filename=tmpdirname + ref_file_name, compression='gzip')
        #print('running RCTD')
        process = subprocess.Popen('bash', shell=False, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
        command = f"""
cd {tmpdirname}
source $(dirname $(dirname $(which conda)))/bin/activate\n\
conda activate {conda_env}\n\
which python
python {script_file_name}
"""
        #print(command)
        out, err = process.communicate(command)
        #print('done')
        if verbose:
            print(out)
            print(err)
        
        #print('now its done. copy your tangram to where you want it')
        #import time
        #time.sleep(60)
        #input('now its done. copy your tangram to where you want it')
        
        if result_file is not None:
            ad_map = ad.read_h5ad(result_file)
        else:
            ad_map = ad.read_h5ad(tmpdirname + result_file_name)
        
        if annotation_key in reference.obsm:
            annotation = reference.obsm[annotation_key]
        else:
            annotation = pd.get_dummies(reference.obs[annotation_key]).astype(np.float32)
        annotation_columns = annotation.columns
        
        if cluster_mode:
            type_fractions = ad_map.X.T
        else:
            # get count-type fractions from the cell-bead matrix
            annotation = annotation.to_numpy()

            utils.row_scale(annotation, np.array(reference.X.sum(axis=1)).flatten() / annotation.sum(axis=1))

            type_fractions = utils.gemmT(ad_map.X.T, annotation.T)
        
        type_fractions = pd.DataFrame(type_fractions, index=ad_map.var.index, columns=annotation_columns)
    
    helper.normalize_result_format(type_fractions, types=reference.obs[annotation_key].unique())
    
    return type_fractions

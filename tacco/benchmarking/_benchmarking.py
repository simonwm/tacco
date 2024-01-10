import tempfile
import subprocess
import os
import shutil
import sys
import time
import gzip
import pickle
import numpy as np
import pandas as pd

TIME_PATH = None
BENCHMARKING_AVAILABLE = False

if os.path.exists('/usr/bin/time'): 
    TIME_PATH = '/usr/bin/time'
    BENCHMARKING_AVAILABLE = True
elif os.path.exists(sys.exec_prefix + '/bin/time'):
    TIME_PATH = sys.exec_prefix + '/bin/time'
    BENCHMARKING_AVAILABLE = True


def _set_up_benchmark_working_directory(
    working_directory,
):
    """\
    Set up a working directory for benchmarking.
    
    Parameters
    ----------
    working_directory
        The directory where to benchmark should run. If `None`, a temporary
        directory is created.
        
    Returns
    -------
    Returns a pair consiting of the path to the working directory and created\
    :class:`~tempfile.TemporaryDirectory` or `None` if no temporary directory\
    was created.
    
    """
    
    tmp_dir = None
    if working_directory is None:
        tmp_dir = tempfile.TemporaryDirectory(prefix='temp_benchmark_',dir='.')
        working_directory = tmp_dir.name
    elif not os.path.exists(working_directory):
        os.makedirs(working_directory)
    working_directory = working_directory + '/'
    
    return (working_directory, tmp_dir)
    
def _tear_down_temporary_directory(
    tmp_dir,
):
    """\
    Clean up a :class:`~tempfile.TemporaryDirectory`.
    
    Parameters
    ----------
    tmp_dir
        The :class:`~tempfile.TemporaryDirectory` to clean up or `None`.
        
    Returns
    -------
    `None`
    
    """
    if tmp_dir is not None:
        tmp_dir.cleanup()

def benchmark_shell(
    command,
    command_args=[],
    working_directory=None,
    verbose=0,
):
    """\
    Benchmarks time and memory consumption of a shell command.
    
    Parameters
    ----------
    command
        A string specifying the command line command to be measured.
    command_args
        A list of strings specifying the command line arguments to supply.
    working_directory
        The directory where to execute the command. If `None`, a temporary
        directory is used and cleaned in the end.
    verbose
        Whether to print stderr and stdout of the command run.
        
    Returns
    -------
    Returns a dict containing the runtime in seconds under the key\
    "shell_time_s" and the memory usage under "max_mem_usage_GB".
    
    """
    
    working_directory, tmp_dir = _set_up_benchmark_working_directory(working_directory)
    
    # run the command
    proc = subprocess.Popen(
        [TIME_PATH,'-f','wall_clock_time_seconds %e\nmax_memory_used_kbytes %M\nexit_status %x',command,*command_args,],
        cwd=working_directory,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    (stdout, stderr) = proc.communicate()

    # parse the results
    error_reason = None
    try:
        max_mem_usage_GB = [ int(l.split(b' ')[-1]) for l in stderr.split(b'\n') if l.startswith(b'max_memory_used_kbytes') ][-1] / 1e6
    except:
        max_mem_usage_GB = np.nan
    try:
        time_s = [ float(l.split(b' ')[-1]) for l in stderr.split(b'\n') if l.startswith(b'wall_clock_time_seconds') ][-1]
    except:
        time_s = np.nan
    try:
        exit_status = [ int(l.split(b' ')[-1]) for l in stderr.split(b'\n') if l.startswith(b'exit_status') ][-1]
    except:
        exit_status = np.nan
        
    if exit_status != 0 and not np.isnan(exit_status):
        error_reason = f'The command to benchmark had an error (exit_status={exit_status}).'
    elif np.isnan(max_mem_usage_GB) or np.isnan(time_s) or np.isnan(exit_status) or proc.returncode != 0:
        error_reason = 'The call to /usr/bin/time did not work.'
    
    # print the output on error or if verbose
    if verbose>0 or error_reason is not None:
        print(stdout.decode("utf-8"))
        print(stderr.decode("utf-8"))
    
    _tear_down_temporary_directory(tmp_dir)
    
    if error_reason is not None:
        raise ValueError(f'The benchmark failed: {error_reason}')
    
    return {
        'shell_time_s': time_s,
        'max_mem_usage_GB': max_mem_usage_GB,
    }

def benchmark_script(
    script,
    working_directory=None,
    verbose=0,
):
    """\
    Benchmarks time and memory consumption of a python script.
    
    Parameters
    ----------
    script
        A string specifying the python script code to be measured.
    working_directory
        The directory where to execute the command. If `None`, a temporary
        directory is used and cleaned in the end.
    verbose
        Whether to print stderr and stdout of the command run.
        
    Returns
    -------
    Returns a dict containing the runtime in seconds under the key\
    "shell_time_s" and the memory usage under "max_mem_usage_GB".
    
    """
    
    working_directory, tmp_dir = _set_up_benchmark_working_directory(working_directory)
    
    # No need to clean the script_file up at the end, as it sits either in a temporary directory or in a different directory which the user wants to be able to inspect.
    script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix='benchmark_script_', dir=working_directory, delete=False)
    script_file.write(script)
    script_file.close()
    
    command = sys.executable
    command_args = [script_file.name]
    
    results = benchmark_shell(command=command, command_args=command_args, working_directory=working_directory, verbose=verbose,)
    
    _tear_down_temporary_directory(tmp_dir)
    
    return results

def benchmark_annotate(
    adata,
    reference,
    working_directory=None,
    verbose=0,
    **kw_args,
):
    """\
    Benchmarks time and memory consumption of an annotation run. This will run
    :func:`tacco.tools.annotate` within an isolated script, in a separate
    python session on serialized `adata` and `reference` data sets. Due to this
    setup, `adata` and `reference` will not be changed by this call (in
    contrast to calling :func:`~tacco.tools.annotate` directly). This removes
    all interactions between consecutive calls to this function.
    
    Parameters
    ----------
    adata
        An :class:`~anndata.AnnData` including expression data in `.X`.
    reference
        Reference data to get the annotation definition from.
    working_directory
        The directory where to execute the command. If `None`, a temporary
        directory is used and cleaned in the end.
    verbose
        Whether to print stderr and stdout of the command run.
    **kw_args
        All extra arguments are forwarded to the :func:`~tacco.tools.annotate`
        call.
        
    Returns
    -------
    Returns a dict containing the runtime of the wrapper script including I/O\
    for reading data in seconds under the key "shell_time_s", the runtime of\
    the call to :func:`~tacco.tools.annotate` under the key\
    "annotation_time_s", the memory usage under "max_mem_usage_GB", the\
    annotation result under "annotation", and the runtime of the call to\
    :func:`~tacco.tools.benchmark_annotate` including I/O for writing and\
    reading data under the key "benchmark_time_s".
    
    """

    if not BENCHMARKING_AVAILABLE:
        raise Exception('No /usr/bin/time or conda-forge time executable found. If on macOS or linux install conda-forge time in your current conda env to run benchmarks')
    
    if working_directory is not None and 'annotation_key' not in kw_args:
        print('`working_directory` is set, but `annotation_key` is not. This frequently is a mistake.\nIf you are certain that it it not, you can deactivate this message by explicitly setting `annotation_key` to `None`.')
     
    start = time.time()

    working_directory, tmp_dir = _set_up_benchmark_working_directory(working_directory)
    
    # pickle instead of AnnData write and read to avoid AnnData's serialization limitations...
    reference_file = tempfile.NamedTemporaryFile(suffix='.pickle.gzip', prefix='reference_', dir=working_directory, delete=False)
    reference_file.close()
    with gzip.open(reference_file.name, 'wb', compresslevel=4) as f:
        pickle.dump(reference, f)
    
    adata_file = tempfile.NamedTemporaryFile(suffix='.pickle.gzip', prefix='adata_', dir=working_directory, delete=False)
    adata_file.close()
    with gzip.open(adata_file.name, 'wb', compresslevel=4) as f:
        pickle.dump(adata, f)
    
    args_file = tempfile.NamedTemporaryFile(suffix='.pickle', prefix='args_', dir=working_directory, delete=False)
    args_file.close()
    with open(args_file.name, 'wb') as f:
        pickle.dump(kw_args, f)
    
    result_file = tempfile.NamedTemporaryFile(suffix='.pickle', prefix='result_', dir=working_directory, delete=False)
    result_file.close()
    
    timing_file = tempfile.NamedTemporaryFile(suffix='.txt', prefix='timing_', dir=working_directory, delete=False)
    timing_file.close()
    
    if 'result_key' in kw_args:
        print(f'The argument "result_key" was set in a call to tc.benchmark.annotate, but will be ignored.')
        del kw_args['result_key']
    
    tacco_benchmarking_dir = os.path.dirname(os.path.abspath(__file__))
    tacco_dir = os.path.dirname(tacco_benchmarking_dir)
    tacco_parent_dir = os.path.dirname(tacco_dir)
    script = f"""
# support devel builds by providing the path to the current tacco installation folder
import sys
sys.path.insert(1, {tacco_parent_dir!r})

import time
import gzip
import pickle
import tacco as tc

with gzip.open('{reference_file.name}', 'rb') as f:
    reference = pickle.load(f)
with gzip.open('{adata_file.name}', 'rb') as f:
    adata = pickle.load(f)

with open('{args_file.name}', 'rb') as f:
    kw_args = pickle.load(f)

start = time.time()
result = tc.tl.annotate(adata, reference, **kw_args)
wall = time.time()-start

with open('{result_file.name}', 'wb') as f:
    pickle.dump(result, f)

with open('{timing_file.name}', 'w') as f:
    f.write('annotation_time_s ' + str(wall))
    print('annotation_time_s ' + str(wall))
"""
    
    results = benchmark_script(script=script, working_directory=working_directory, verbose=verbose,)
    
    try:
        with open(timing_file.name, 'r') as f:
            time_line = f.readline().split(' ')
            if len(time_line) != 2 or time_line[0] != 'annotation_time_s':
                raise ValueError(f'The annotation failed')
            else:
                results['annotation_time_s'] = float(time_line[-1])

        with open(result_file.name, 'rb') as f:
            results['annotation'] = pickle.load(f)
    finally:
        _tear_down_temporary_directory(tmp_dir)
    
    benchmark_time_s = time.time()-start
    results['benchmark_time_s'] = benchmark_time_s

    return results

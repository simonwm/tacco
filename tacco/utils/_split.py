import numpy as np
from numba import njit, prange
from scipy.sparse import issparse, hstack, vstack, coo_matrix, csr_matrix
import anndata as ad
import pandas as pd
from numpy.random import Generator, PCG64
import joblib
from ._utils import cpu_count
from . import _math
import time
import gc

def matrix_scaling(x,u,v, min_iter=3, max_iter=100, delta=1e-3, check_iter=1):
    total_u = np.array(u.sum(axis=0)).flatten()
    total_v = np.array(v.sum(axis=1)).flatten()
    if np.abs((total_u - total_v) / (total_u + total_v)).mean() > 1e-5:
        raise ValueError('"u" and "v" dont seem to be normalized identically! Make sure that the rows of u have the same sums as the columns of v.')
    if (np.array(x.sum(axis=0)).flatten() == 0).any():
        raise ValueError('"x" has zero-only columns!')
    if (np.array(x.sum(axis=1)).flatten() == 0).any():
        raise ValueError('"x" has zero-only rows!')
    
    if issparse(u):
        a = u.tocoo()
        u_data = a.data.copy()
        a.data *= 0
    else:
        a = np.zeros(u.shape,dtype=u.dtype)
    b = np.ones(v.shape,dtype=u.dtype)
    
    for i in range(max_iter):
        if i >= min_iter:
            if issparse(a):
                _a = a.data.copy()
            else:
                _a = a.copy()
        
        if issparse(a):
            _math.sparse_result_gemmT(x, b, a, parallel=False, inplace=True)
            _math.divide(u_data,a.data,out=a.data, parallel=False)
            temp2 = (a.tocsr().T)@x # tocsr is not necessary but marginally faster than directly using the coo format in the matrix multiplication
        else:
            temp1 = x@(b.T)
            _math.divide(u, temp1, out=a, parallel=False)
            temp2 = (a.T)@x
            
        if i >= min_iter:
            _b = b.copy()
        
        _math.divide(v, temp2, out=b, parallel=False)
        
        # The convergence criterion is in principle defined per problem, i.e. per bead.
        # We nevertheless run the algorithm for the whole batch for simplicity until every bead converges.
        # In effect this also give a little bonus accuracy for the earlier converging beads. 
        if i >= min_iter and (i - min_iter) % check_iter == 0:
            nonzero = b != 0
            b_rel = np.max(np.abs(_b[nonzero]/b[nonzero]-1)) # check only b first: ntypes << ngenes
            if b_rel < delta:
                if issparse(a):
                    a_rel = np.max(np.abs(_a/a.data-1))
                else:
                    nonzero = a != 0
                    a_rel = np.max(np.abs(_a[nonzero]/a[nonzero]-1))
                if a_rel < delta:
                    break

    # Depending on the batch size the deltas change sometimes unpredictably - which should not happen, as the calculations for the beads should be (mostly) independent.
    # Therefore TODO: investigate what is going on.
    # But the variation has no (visible?) effect on the results, so it is not urgent...
    #print('niter', i, 'b_rel', b_rel, 'a_rel', a_rel)
        
    return a,b

def parallel_matrix_scaling(x,u,v, min_iter=3, max_iter=100, delta=1e-6, check_iter=1, n_jobs=None, batch_size=1000):
    if n_jobs is None:
        n_jobs = cpu_count()
    
    result = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(matrix_scaling)(x,u[:,batch:(batch+batch_size)].copy(),v[batch:(batch+batch_size)].copy(), min_iter=min_iter, max_iter=max_iter, delta=delta, check_iter=check_iter)
        for batch in range(0,v.shape[0],batch_size)
    )

    if issparse(u):
        a = hstack([_a for _a,_b in result])
        a = a.tocsc() # this is necessary to have really identical results excluding all permutations of entries arising from concatenating several .coo matrices.
    else:
        a = np.hstack([_a for _a,_b in result])
    b = np.vstack([_b for _a,_b in result])
    
    #print(hash(a.indices.tobytes()),hash(a.indptr.tobytes()),hash(a.data.tobytes()),hash(b.tobytes()))
    
    return a,b

def _scale(cx,bead_type_map,type_profiles, delta=1e-6, batch_size=1000, n_jobs=None):
    # implements RAS algorithm in the factor estimation version https://en.wikipedia.org/wiki/Iterative_proportional_fitting
    # with custom optimizations
    # background to matrix scaling: https://arxiv.org/pdf/1609.06349.pdf

    # chunksize has an effect (of about the level as a different random seed) on the results as it influences the set on of the stopping criteria for the matrix scaling
    
    # bead_type_map may have zero elements, possibly leading to unsolvable situations for the matrix scaling
    # e.g. some gene is only expressed by a single celltype and there is a read at a given bead, but counts_per_bead_and_type says there is less than 1 read expected of a given celltype.
    # workaround: add epsilon expression probability for every gene in every celltype
    type_profiles = type_profiles + 1 / type_profiles.shape[0] * 1e-3
    type_profiles = type_profiles / type_profiles.sum(axis=0)
    
    cx = cx.astype(float)
    counts_per_bead_and_gene = cx
    counts_per_bead = np.array(counts_per_bead_and_gene.sum(axis=1)).flatten().astype(cx.dtype)
    counts_per_bead_and_type = bead_type_map * counts_per_bead[:,None]

    x = type_profiles.astype(cx.dtype)  # b_rescaled x will be diag(a_b).x.diag(b_b)
    
    u = counts_per_bead_and_gene.T.copy()          # per_gene_and_bead
    v = counts_per_bead_and_type.astype(cx.dtype) # per_bead_and_type

    a,b = parallel_matrix_scaling(x,u,v, batch_size=batch_size, n_jobs=n_jobs)
        
    # rescaled splitted count matrix is a_gb x_gt b_bt
    return a,b,x

@njit(fastmath=True, parallel=True, cache=True)
def _fuse(a_rows,a_cols,a_data,b,x):
    nt = x.shape[1]
    nz = len(a_data)
    layers = np.empty((nt,nz))
    for gb in prange(nz):
        g = a_rows[gb]
        _b = a_cols[gb]
        for t in range(nt):
            layers[t,gb] = a_data[gb] * x[g,t] * b[_b,t]
    return layers

@njit(cache=True)#(fastmath=True,parallel=True) # fastmath and parallel optimizations give no significant performance gain, but fastmath decreases the accuracy slightly 
def _categorical(counts, cdf, random):
    draws = np.zeros_like(cdf)
    r = 0
    for cg in prange(cdf.shape[1]):
        for i in range(counts[cg]):
            random_r = random[r]
            pick = 0
            for t in range(cdf.shape[0]-1):
                pick = random_r < cdf[t,cg]
                if pick:
                    draws[t,cg] += 1
                    break
            draws[-1,cg] += 1 - pick # make sure, that the read is accounted for somewhere even if there are rounding errors in cdf
            r += 1
    return draws
@njit(cache=True)
def multinomial_rest(counts, probabilities, seed=42):
    if len(counts) != probabilities.shape[1]:
        raise Exception('counts must be of the same length as the number of columns of probabilities!')
    np.random.seed(seed)
    random = np.random.uniform(0,1,size=counts.sum()) # randomness needed for eqivalent draws from the categorical distribution
    for i in range(1,probabilities.shape[0]): # manual version of cdf = probabilities.cumsum(axis=0) for numba
        probabilities[i] += probabilities[i-1]
    return _categorical(counts, probabilities, random)

@njit(parallel=True, cache=True)
def _fuseall(a_rows,a_cols,a_data,b,x, seed=42, batch_size=50000):
    nt = x.shape[1]
    nz = len(a_data)

    n_batches = (nz+batch_size-1)//batch_size
    
    # buffer the sparse splitted expression batch-wise
    l_rows = []
    l_cols = []
    l_data = []
    # lengthy setup to tell numba what types will be stored in the lists
    for t in range(nt):
        l_rows_i = []
        l_cols_i = []
        l_data_i = []
        for i in range(n_batches):
            l_rows_i_t = a_rows[:0]
            l_cols_i_t = a_cols[:0]
            l_data_i_t = a_data[:0]
            l_rows_i.append(l_rows_i_t)
            l_cols_i.append(l_cols_i_t)
            l_data_i.append(l_data_i_t)
        l_rows.append(l_rows_i)
        l_cols.append(l_cols_i)
        l_data.append(l_data_i)

    for i in prange(n_batches):
        batch = i*batch_size
        
        _batch_size = min(batch+batch_size,nz) - batch
        
        layers_data = np.zeros((nt,_batch_size),dtype=a_data.dtype)
        for gb in range(batch,batch+_batch_size):
            g = a_rows[gb]
            _b = a_cols[gb]
            for t in range(nt):
                layers_data[t,gb-batch] = a_data[gb] * x[g,t] * b[_b,t]
        
        if seed is not None:
            
            integer_part = np.floor(layers_data)
            layers_data -= integer_part # layers_data contains now the remainder
            remaining_reads = layers_data
            remaining_total = remaining_reads.sum(axis=0)
            for j in range(len(remaining_total)):
                remaining_reads[:,j] /= remaining_total[j] # normalize remaining_reads to remaining_probabilities

            np.around(remaining_total,0,remaining_total)
            counts = remaining_total.astype(np.int64)
            probabilities = remaining_reads
            
            # Every batch gets its own random seed, so even though it might be parallel, the results will be identical irrespective of the degree of parallelism.
            # But that means that changing the batch_size has an impact on the results, via changing the random seed for parts of the batchhes.
            layers_data = integer_part + multinomial_rest(counts, probabilities, seed=seed+batch//batch_size)
        
        for t in range(nt):
            nonzero = layers_data[t] != 0
            l_rows[t][i] = a_rows[batch:(batch+_batch_size)][nonzero].copy()
            l_cols[t][i] = a_cols[batch:(batch+_batch_size)][nonzero].copy()
            l_data[t][i] = layers_data[t][nonzero].copy()
    
    # accumulate the sparse splitted expression batch-wise
    l_rows_acc = []
    l_cols_acc = []
    l_data_acc = []
    for t in range(nt):
        nz_total = 0
        for i in range(n_batches):
            nz_total += len(l_rows[t][i])
        l_rows_acc_t = np.empty(nz_total, dtype=a_rows.dtype)
        l_cols_acc_t = np.empty(nz_total, dtype=a_cols.dtype)
        l_data_acc_t = np.empty(nz_total, dtype=a_data.dtype)
        running_gb = 0
        for i in range(n_batches):
            nz_i = len(l_rows[t][i])
            l_rows_acc_t[running_gb:(running_gb+nz_i)] = l_rows[t][i]
            l_cols_acc_t[running_gb:(running_gb+nz_i)] = l_cols[t][i]
            l_data_acc_t[running_gb:(running_gb+nz_i)] = l_data[t][i]
            running_gb += nz_i
        l_rows_acc.append(l_rows_acc_t)
        l_cols_acc.append(l_cols_acc_t)
        l_data_acc.append(l_data_acc_t)
    
    return l_rows_acc, l_cols_acc, l_data_acc

def split_beads(tdata, bead_type_map, type_profiles, min_counts=None, seed=42, delta=1e-6, scaling_jobs=None, scaling_batch_size=1000, rounding_batch_size=50000, copy=True, map_all_genes=False, return_split_profiles=False, verbose=1):
    if scaling_jobs is None:
        scaling_jobs = cpu_count()
    
    # subset to common genes, beads, and types (with non-zero counts)
    if map_all_genes:
        genes = tdata.var.index
        type_profiles = type_profiles.reindex(index=genes).fillna(0.0)
    else:
        nonzero_genes = np.array(tdata.X.sum(axis=0)).flatten() != 0
        if verbose > 0 and not nonzero_genes.all():
            print(f'removed {len(nonzero_genes)-nonzero_genes.sum()} of {len(nonzero_genes)} genes from count matrix due to zero counts in gene')
        nonzero_profile_genes = type_profiles.sum(axis=1).to_numpy() != 0
        if verbose > 0 and not nonzero_profile_genes.all():
            print(f'removed {len(nonzero_profile_genes)-nonzero_profile_genes.sum()} of {len(nonzero_profile_genes)} genes from profile definition due to zero appearance in the profiles')
        genes = tdata.var.index[nonzero_genes].intersection(type_profiles.index[nonzero_profile_genes])
        if verbose > 0 and nonzero_genes.sum() > len(genes):
            print(f'removed {(nonzero_genes).sum() - len(genes)} of {(nonzero_genes).sum()} genes from count matrix due to missing intersection with profiles')
    
    nonzero_beads = np.array(tdata.X.sum(axis=1)).flatten() != 0
    if verbose > 0 and not nonzero_beads.all():
        print(f'removed {len(nonzero_beads)-nonzero_beads.sum()} of {len(nonzero_beads)} observation from count matrix due to zero counts in observation')
    nonzero_map_beads = bead_type_map.sum(axis=1).to_numpy() != 0
    if verbose > 0 and not nonzero_map_beads.all():
        print(f'removed {len(nonzero_map_beads)-nonzero_map_beads.sum()} of {len(nonzero_map_beads)} observations from mapping due to zero mapping')
    beads = tdata.obs.index[nonzero_beads].intersection(bead_type_map.index[nonzero_map_beads])
    if verbose > 0 and nonzero_beads.sum() > len(beads):
        print(f'removed {(nonzero_beads).sum() - len(beads)} of {(nonzero_beads).sum()} observations from count matrix due to missing intersection with mapping')
    
    types = type_profiles.columns.intersection(bead_type_map.columns)
    tdata = tdata[beads,genes]
    type_profiles = type_profiles.loc[genes,types]
    bead_type_map = bead_type_map.loc[beads,types]
    
    # normalize bead_type_map: every bead has type fractions that sum to 1
    bead_type_map = bead_type_map / bead_type_map.sum(axis=1).to_numpy()[:,None]
    
    if min_counts is not None and min_counts > 0:
        bead_type_counts = bead_type_map.to_numpy() * _math.get_sum(tdata.X, axis=1)[:,None]
        keep_mask = (bead_type_counts >= min_counts) | (bead_type_counts == bead_type_counts.max(axis=1)[:,None]) # keep the highest contribution and all others which meet the threshold
        bead_type_map = bead_type_map * keep_mask
        bead_type_map = bead_type_map / bead_type_map.sum(axis=1).to_numpy()[:,None]
    
    # normalize type_profiles: every type has gene fractions that sum to 1
    type_profiles = type_profiles / type_profiles.sum(axis=0).to_numpy()
    
    type_profiles = type_profiles[types]
    
    if verbose > 0:
        print('scale..', end='...')
    start = time.time()
    a, b, x = _scale(tdata.X,bead_type_map.to_numpy(),type_profiles.to_numpy(), delta=delta, batch_size=scaling_batch_size, n_jobs=scaling_jobs)
    #print(hash(a.indices.tobytes()),hash(a.indptr.tobytes()),hash(a.data.tobytes()),hash(b.tobytes()),hash(x.tobytes()))
    # rescaled splitted count matrix is a_gb x_gt b_bt
    if verbose > 0:
        print('time', time.time() - start)
    
    if return_split_profiles:
        # sum split over beads
        ab = a @ b
        if issparse(ab):
            return ab.multiply(x)
        if issparse(x):
            return x.multiply(ab)
        return x * ab
    
    if verbose > 0:
        print('fuseall', end='...')
    start = time.time()
    if issparse(a):
        a = a.tocoo()
    else:
        a = coo_matrix(a)
    _layers_data = _fuseall(a.row,a.col,a.data,b,x, seed=seed, batch_size=rounding_batch_size)
    for typ, rows, cols, data in zip(types,*_layers_data):
        # this way of constructing a coo_matrix has slightly less overhead...
        l_coo = coo_matrix(tdata.shape,dtype=data.dtype)
        l_coo.row = cols # exchange rows and cols due to implicit transposition in _scale
        l_coo.col = rows
        l_coo.data = data
        tdata.layers[typ] = l_coo.tocsr()
    if verbose > 0:
        print('time', time.time() - start)
    
    return tdata

def combine_layers(adata, layer_mapping, remove_old_layers=False):
    
    contributions = {}
    for old, new in layer_mapping.items():
        if new not in contributions:
            contributions[new] = []
        if old in adata.layers:
            contributions[new].append(old)
        else:
            print(f'Warning: requested layer {old!r} did not exist in adata!')
    
    for new, olds in contributions.items():
        if len(olds) == 0:
            print(f'Warning: new layer {new!r} effectively did not get any mapping in adata and is set to 0!')
            adata.layers[new] = csr_matrix(adata.X.shape)
        else:
            X = adata.layers[olds[0]]
            if remove_old_layers:
                del adata.layers[olds[0]]
            for i in range(1,len(olds)):
                X = X + adata.layers[olds[i]]
                if remove_old_layers:
                    del adata.layers[olds[i]]
            adata.layers[new] = X

def merge_layers(tdata, layer_names, merge_annotation_name='layers', bead_annotation_name='bc', min_reads=None):
    X = vstack([tdata.layers[l] for l in layer_names]).tocsr()
    def assign(df,name,value,i):
        df[name] = value
        if bead_annotation_name in tdata.obs.columns:
            df.reset_index(drop=True, inplace=True)
        else:
            df.index.rename(bead_annotation_name, inplace=True)
            df.reset_index(drop=False, inplace=True)
        df.set_index(str(i) + '_' + df.index.astype(str),inplace=True) # unique index for all observations to avoid warnings at AnnData creation
        return df
    obs = pd.concat([assign(tdata.obs.copy(),merge_annotation_name,l,i) for i,l in enumerate(layer_names)])
    # conserve categorical layer_names dtype
    if hasattr(pd.Series(layer_names), 'cat'):
        obs[merge_annotation_name] = obs[merge_annotation_name].astype(layer_names.dtype)
    data = ad.AnnData(X,obs=obs,var=tdata.var,varm=tdata.varm)
    keep = None
    if min_reads is None:
        keep = _math.get_sum(data.X, axis=1) > 0
    elif min_reads > 0:
        keep = _math.get_sum(data.X, axis=1) >= min_reads
    if keep is not None:
        data = data[keep].copy()
        gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    data.obs.reset_index(drop=True, inplace=True)
    data.obs.index = data.obs.index.astype(str) # convenient for some downstream analysis
    return data

def merge_beads(mdata, merge_annotation_name='layers', bead_annotation_name='bc', min_counts=0, mean_keys=None):
    if merge_annotation_name is not None:
        joined = pd.Series([(t,bc)for t,bc in zip(mdata.obs[merge_annotation_name],mdata.obs[bead_annotation_name])])
    else:
        joined = pd.Series(mdata.obs[bead_annotation_name].to_numpy())
    joined_cat = joined.astype('category')
    
    if issparse(mdata.X):
        X = mdata.X.tocoo()
        X.row = joined_cat.cat.codes[X.row].to_numpy()
        X.resize((len(joined_cat.dtype.categories),X.shape[1]))
        X.eliminate_zeros()
        X = X.tocsr()
    else:
        joined_dummies = coo_matrix((np.ones(len(mdata.obs.index),dtype=X.dtype),(joined_cat.cat.codes,np.arange(len(mdata.obs.index)))))
        X = _math.gemmT(joined_dummies, mdata.X.T)
    
    # recover (some version of) all the observable annotation we had before.
    # if it is not consistent between the different contributions for merge_annotation_name and bead_annotation, the first one will be picked
    obs = mdata.obs.copy()
    obs['__cat_codes__'] = joined_cat.cat.codes.to_numpy()
    
    # create mean obs annotations
    if mean_keys is not None:
        if isinstance(mean_keys, str):
            mean_keys = [mean_keys]
        means = obs.groupby('__cat_codes__')[mean_keys].mean()
        
    obs.drop_duplicates(subset=['__cat_codes__'], inplace=True)
    obs.sort_values(by=['__cat_codes__'], inplace=True)
    
    if mean_keys is not None:
        for mean_key in mean_keys:
            obs[mean_key] = obs['__cat_codes__'].map(means[mean_key])
    
    del obs['__cat_codes__']
    
    data = ad.AnnData(X,obs=obs,var=mdata.var,varm=mdata.varm)
    
    if min_counts > 0:
        data = data[data.X.sum(axis=1)>=min_counts].copy()
        gc.collect() # anndata copies are not well garbage collected and accumulate in memory
    
    return data

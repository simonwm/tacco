from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
import collections
import scipy
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils import parallel_nnls
from . import _helper as helper
from .. import get

def annotate_NMFreg(
    adata,
    reference,
    annotation_key,
    K=30,
    random_state=42,
    counts_location=None,
    min_counts_per_cell=0,
    min_counts_per_gene=0,
):

    """\
    Annotates an :class:`~anndata.AnnData` using reference data by NMFreg
    [Rodriques19]_.

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
    K
        Number of NMF factors
    random_state
        Random seed
    counts_location
        A string or tuple specifying where the count matrix is stored, e.g.
        `'X'`, `('raw','X')`, `('raw','obsm','my_counts_key')`,
        `('layer','my_counts_key')`, ... For details see
        :func:`~tacco.get.counts`.
    min_counts_per_cell
        Minimum number of counts per cell
    min_counts_per_gene
        Minimum number of counts per gene
        
    Returns
    -------
    Returns the annotation in a :class:`~pandas.DataFrame`.
    
    """
    
    tdata, reference, annotation_key = helper.validate_annotation_args(adata, reference, annotation_key, counts_location, full_reference=True)
    
    atlas_dge = reference.copy()
    
    celltype_dict_rev = {cluster: ic+1 for ic,cluster in enumerate(atlas_dge.obs[annotation_key].unique())}

    idx_name = 'barcode'
    atlas_dge.obs.index.rename(idx_name, inplace=True)
    
    cell_clusters = atlas_dge.obs.reset_index()[[idx_name, annotation_key]]
    cell_clusters['cluster'] = cell_clusters[annotation_key].map(celltype_dict_rev).astype(int)

    #subset to only use really good cells
    atlas_dge = atlas_dge[(atlas_dge.X.sum(axis=1)>min_counts_per_cell),:]
    atlas_dge = atlas_dge[:,(atlas_dge.X.sum(axis=0)>min_counts_per_gene)]
    cell_clusters = cell_clusters.loc[(cell_clusters.barcode.isin(atlas_dge.obs.index)),]
    cell_clusters.reset_index(inplace=True, drop=True)

    metacell_dict = { str(celltype_dict_rev[key]): key for key in celltype_dict_rev }
    celltype_dict = { celltype_dict_rev[key]: key for key in celltype_dict_rev }
    plot_size_dict = {10:4}
    ct_names = list(metacell_dict.values())

    # need to have both the sample and the reference over a shared set of features
    gene_intersection = atlas_dge.var.index.intersection(tdata.var.index)
    atlas_dge = atlas_dge[:, gene_intersection].copy()
    # preprocess the reference for NMF
    cell_totalUMIa = np.array(atlas_dge.X.sum(axis=1)).flatten()
    sc.pp.normalize_total(atlas_dge, target_sum=1.0)
    sc.pp.scale(atlas_dge, zero_center=False)

    model = NMF(n_components=K, init='random', random_state = random_state)
    Ha = model.fit_transform(atlas_dge.X)
    Wa = model.components_

    Ha_norm = StandardScaler(with_mean=False).fit_transform(Ha)
    Ha_norm = pd.DataFrame(Ha_norm)
    Ha_norm['barcode'] = atlas_dge.obs.index.tolist()

    Ha = pd.DataFrame(Ha)
    Ha['cellname'] = atlas_dge.obs.index.tolist()

    WaT = Wa.T

    maxloc = Ha_norm.drop('barcode', axis=1).values.argmax(axis=1)
    cell_clusters['maxloc'] = maxloc

    num_atlas_clusters = np.unique(cell_clusters['cluster']).size

    factor_to_celltype_df = pd.DataFrame(0, index=range(1, num_atlas_clusters+1), 
                                         columns=range(K))
    fig,ax=plt.subplots(1,1)
    for k in range(K):
        n, bins, patches = ax.hist(cell_clusters['cluster'][cell_clusters['maxloc'] == k],
                range = (0.5, num_atlas_clusters+0.5), 
                                    bins = int(num_atlas_clusters), 
                                    facecolor='green', alpha=0.75)
        ax.set_xticks(ax.get_xticks()) # seems necessary to avoid a warning message in the line below
        ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
        factor_to_celltype_df[k] = n.astype(int)

    factor_to_celltype_df = factor_to_celltype_df.T

    factor_total = np.sum(factor_to_celltype_df, axis = 1)
    factor_to_celltype_df_norm = np.divide(factor_to_celltype_df, 
                                           factor_total.to_numpy()[:,None])

    cx = sns.clustermap(factor_to_celltype_df_norm, fmt = 'd',
                    cmap="magma_r", linewidth=0.5, col_cluster = False,
                       figsize=(5, 6),cbar_pos=None)

    #ax = sns.clustermap(factor_to_celltype_df_norm, fmt = 'd',
    #        cmap="magma_r", linewidth=0.5, col_cluster = False,
    #        annot = factor_to_celltype_df.loc[cx.dendrogram_row.reordered_ind],
    #        figsize=(7, 9))

    maxloc_fc = factor_to_celltype_df.values.argmax(axis=1)
    factor_to_celltype_dict = {factor : ctype + 1 for factor, ctype in enumerate(maxloc_fc)}

    celltype_to_factor_dict = {}
    for c in range(1, num_atlas_clusters + 1):
        celltype_to_factor_dict[c] = [k for k, v in factor_to_celltype_dict.items() if v == c]

    nonzero_beads = tdata[:, gene_intersection].X.sum(axis=1)!=0
    counts = pd.DataFrame(tdata[nonzero_beads,gene_intersection].X.toarray(), columns=gene_intersection)
    counts['barcode'] = tdata[nonzero_beads].obs.index

    co = collections.Counter(factor_to_celltype_dict.values())

    def deconv_factor_to_celltype(row, adict, K, num_atlas_clusters):
        nc = num_atlas_clusters
        tmp_list = [0]*nc
        for key in range(K):
            item = adict[key] - 1
            tmp_list[item] += row[key]**2
        return pd.Series(np.sqrt(tmp_list))

    def NMFreg(counts, size, metacell_dict, gene_intersection, 
               num_atlas_clusters, celltype_to_factor_dict, 
               celltype_dict, plot_size_dict):

        puckcounts = counts[['barcode'] + gene_intersection]
        puckcounts = puckcounts.set_index(counts['barcode'])
        puckcounts = puckcounts.drop('barcode', axis=1)

        cell_totalUMI = np.sum(puckcounts, axis = 1)
        puckcounts_cellnorm = np.divide(puckcounts, cell_totalUMI.to_numpy()[:,None])
        puckcounts_scaled = StandardScaler(with_mean=False).fit_transform(puckcounts_cellnorm)

        XsT = puckcounts_scaled.T

        Hs_hat = parallel_nnls(WaT, XsT.T)

        Hs = pd.DataFrame(Hs_hat)
        Hs['barcode'] = puckcounts.index.tolist()

        Hs_norm = StandardScaler(with_mean=False).fit_transform(Hs.drop('barcode', 
                                                                        axis=1))

        Hs_norm = pd.DataFrame(Hs_norm)
        Hs_norm['barcode'] = puckcounts.index.tolist()


        maxloc_s = Hs_norm.drop('barcode', axis=1).values.argmax(axis=1)
        barcode_clusters = pd.DataFrame()
        barcode_clusters['barcode'] = Hs_norm['barcode']
        barcode_clusters['max_factor'] = maxloc_s

        barcode_clusters['atlas_cluster'] = barcode_clusters['barcode']

        for c in range(1, num_atlas_clusters + 1):
            condition = np.isin(barcode_clusters['max_factor'], 
                                celltype_to_factor_dict[c])
            barcode_clusters.loc[condition,'atlas_cluster'] = c       

        bead_deconv_df = Hs_norm.apply(lambda x: deconv_factor_to_celltype(row=x, 
                                                adict=factor_to_celltype_dict,
                                                K=K,
                                                num_atlas_clusters=num_atlas_clusters), 
                                       axis = 1)
        bead_deconv_df.insert(0, 'barcode', Hs_norm['barcode'])
        bead_deconv_df.columns = ['barcode'] + (bead_deconv_df.columns[1:]+1).tolist()
        bead_deconv_df = pd.DataFrame(bead_deconv_df)
        bead_deconv_df = bead_deconv_df.rename(columns = celltype_dict)

        maxloc_ct = bead_deconv_df.drop('barcode', axis=1).values.argmax(axis=1)+1
        bead_maxct_df = pd.DataFrame()
        bead_maxct_df['barcode'] = bead_deconv_df['barcode']
        bead_maxct_df['max_cell_type'] = maxloc_ct

        return Hs, Hs_norm, puckcounts, bead_deconv_df, barcode_clusters, bead_maxct_df

    Hs, Hs_norm, puckcounts, bead_deconv_df, barcode_clusters, bead_maxct_df = NMFreg(counts=counts, 
                                            size=10, 
                                            metacell_dict=metacell_dict, 
                                            gene_intersection=list(gene_intersection), 
                                            num_atlas_clusters=num_atlas_clusters, 
                                            celltype_to_factor_dict=celltype_to_factor_dict, 
                                            celltype_dict=celltype_dict, 
                                            plot_size_dict=plot_size_dict)

    barcode_totalloading = np.sum(bead_deconv_df.drop('barcode', axis=1), axis = 1)
    bead_deconv_df_norm = np.true_divide(bead_deconv_df.drop('barcode', axis=1), barcode_totalloading.to_numpy()[:,None])
    bead_deconv_df_norm.index = puckcounts.index
    bead_deconv_df_norm.index.name = tdata[nonzero_beads].obs.index.name

    bead_deconv_df_norm = helper.normalize_result_format(bead_deconv_df_norm)
    
    return bead_deconv_df_norm

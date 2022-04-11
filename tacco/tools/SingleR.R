args=commandArgs(trailingOnly = TRUE)

library(data.table)
library(SingleR)

library(hdf5r)
library(Matrix) # must be loaded before SparseM to enable cast from SparseM to Matrix
library(SparseM)

read_matrix = function(file, name) {
    if (name %in% list.datasets(file,recursive=FALSE)) {
        print('dense')
        newX = t(file$open(name)$read())
    } else if ('X' %in% list.groups(file,recursive=FALSE)) {
        groupX = openGroup(file, name)
        groupX_encoding_type = h5attr(groupX,'encoding-type')
        if (groupX_encoding_type == 'csr_matrix' || groupX_encoding_type == 'csc_matrix' ) {
            ra = groupX$open('data')$read()
            ja = as.integer(groupX$open('indices')$read()+1)
            ia = as.integer(groupX$open('indptr')$read()+1)
            dim = h5attr(groupX,'shape')
            if (groupX_encoding_type == 'csr_matrix') {
                print('csr')
                newX = new("matrix.csr", ra=ra, ja=ja, ia=ia, dimension=dim)
            } else if (groupX_encoding_type == 'csc_matrix') {
                print('csc')
                newX = new("matrix.csc", ra=ra, ja=ja, ia=ia, dimension=dim)
            }
            newX = as(newX,'dgCMatrix')
        } else {
            print('unkown encoding for X...')
        }
    } else {
        print('unkown encoding for X...')
    }
    return(newX)
}
read_df = function(file, name) {
    #print(name)
    group = openGroup(file, name)
    #print('---------- group')
    #print(group)
    categories = NULL
    if (group$exists('__categories')) {
        categories = group$open('__categories')
        categories_ds = list.datasets(categories)
    }
    #print('---------- index')
    #print(h5attr(group, '_index'))
    #print('---------- group open')
    #print(group$open(h5attr(group, '_index')))
    #print('---------- group open read')
    #print(group$open(h5attr(group, '_index'))$read())
    #print('---------- df')
    df = data.frame(row.names=group$open(h5attr(group, '_index'))$read())
    if (length(list.datasets(group)) > 1) { # catch case with only the index and no additional column
        for (col in h5attr(group, 'column-order')) {
            temp = group$open(col)$read()
            if (!is.null(categories) && col %in% categories_ds) {
                temp = categories$open(col)$read()[temp+1]
            }
            df[col] = temp
        }
    }
    return(df)
}
read_adata = function(filename, transpose=FALSE) {
    file = H5File$new(filename, mode = "r")
    newX = read_matrix(file, 'X')
    obs_df = read_df(file, 'obs')
    var_df = read_df(file, 'var')
    #print('---------- newX')
    #print(newX)
    #print(colnames(newX))
    #print('---------- var_df')
    #print(var_df)
    #print(row.names(var_df))
    colnames(newX) = row.names(var_df)
    #print('---------- obs_df')
    #print(obs_df)
    row.names(newX) = row.names(obs_df)
    if (transpose) {
        return(list('X'=t(newX),'obs'=var_df,'var'=obs_df))
    } else {
        return(list('X'=newX,'obs'=obs_df,'var'=var_df))
    }
}

sc_adata=args[1]
sp_adata=args[2]
out = args[3]
annotation_key = args[4]
fine.tune = args[5]
aggr.ref = args[6]

print(args)

print('reading data')
#puck=read.SpatialRNA_custom(sp_adata,x_col, y_col)
adata = read_adata(sp_adata)
adata.Xt <- t(adata$X)
print('reading reference')
#reference=dgeToSeurat_custom(sc_adata,ct_col,min_ct)
reference = read_adata(sc_adata)
reference.Xt <- t(reference$X)

meta_data = reference$obs
cell_types=reference$obs[,annotation_key]
names(cell_types)=row.names(reference$obs)

print('running SingleR')
SingleR_result <- SingleR(test=adata.Xt, ref=reference.Xt, labels=cell_types, fine.tune=fine.tune, genes='all', de.method="wilcox", aggr.ref=aggr.ref)

write.table(cbind(row.names(SingleR_result),SingleR_result$labels),paste0(out,".tsv"),sep="\t",quote=FALSE,row.names=FALSE)

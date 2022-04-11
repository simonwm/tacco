args=commandArgs(trailingOnly = TRUE)

library(data.table)
library(RCTD)

### patch RCTD for debugging
#funcname <- function(...) {
#...
#}
#environment(funcname) <- asNamespace('RCTD')
#assignInNamespace("funcname", funcname, ns = "RCTD")

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

min_ct = as.numeric(args[4])
cores = as.numeric(args[5])
mode = args[6]
ct_col = args[7]
x_col = args[8]
y_col = args[9]
UMI_min_sigma = args[10]

print(args)

#functions
read_reference=function(adata_file,ct_col,min_ct){
    adata = read_adata(adata_file)
    #print(adata$X)
    raw.data <- as(t(adata$X),'dgCMatrix')
    meta_data = adata$obs

    #pre-filter for cell types with sufficient cells
    cell_types=meta_data[,ct_col]
    names(cell_types)=row.names(meta_data)
    stats=table(cell_types)
    good_ct=names(stats[stats>=min_ct])
    print(stats)
    keep=cell_types%in%good_ct
    print(table(keep))
    
    data_filt=raw.data[,keep]
    ct_filt=as.factor(cell_types[keep])
    counts_filt=meta_data$n_counts[keep]
    
    dref=Reference(data_filt, ct_filt, counts_filt)
    return(dref)
}

read_spatial=function (adata_file,x_col,y_col) 
{
    adata = read_adata(adata_file)
    counts <- as(t(adata$X),'dgCMatrix')
    coords = adata$obs[,c(x_col,y_col)]
    
    colnames(coords)[1] = "x"
    colnames(coords)[2] = "y"
    puck = RCTD:::SpatialRNA(coords, counts)
    RCTD:::restrict_puck(puck, colnames(puck@counts))
}

#code

print('reading data')
puck=read_spatial(sp_adata,x_col, y_col)
print('reading reference')
reference=read_reference(sc_adata,ct_col,min_ct)


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

#save(myRCTD,file=paste0(out,".RData"))

if (is(weights, 'sparseMatrix')) {
    weights = as.data.frame(as.matrix(weights))
}

write.table(weights,paste0(out,".tsv"),sep="\t",quote=FALSE)

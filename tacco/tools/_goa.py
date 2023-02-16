import pandas as pd
import os
import gzip
import shutil
import tempfile
import urllib

try: # dont fail importing the whole module, just because go term enrichment analysis is not available
    from goatools.obo_parser import GODag
    from goatools.anno.genetogo_reader import Gene2GoReader
    from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS
    HAVE_GOA = True
except ImportError:
    HAVE_GOA = False

_symbol2geneID = None
_goeaobj = None

def _get_downloaded_file(URL, working_directory):
    if os.path.isfile(URL):
        return URL
    fname = working_directory + '/' + URL.split('/')[-1]
    if os.path.isfile(fname):
        print(f'using buffered {fname}')
        return fname
    print(f'downloading {URL}')
    urllib.request.urlretrieve(URL, fname)
    return fname

def setup_goa_analysis(
    gene_index,
    gene_info_file='https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz',
    tax_id=10090,
    GO_obo_file='http://purl.obolibrary.org/obo/go/go-basic.obo',
    gene2GO_file='https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz',
    working_directory='.',
):
    """\
    Setup a GO analysis. This is a convenience wrapper around the goatools
    package [Klopfenstein18]_ and like goatools performs the enrichment
    analysis independent of the availability of webservices using a databases
    downloaded once for reproducibility.
    
    Parameters
    ----------
    gene_index
        The list of all possible genes.
    gene_info_file
        File containing a mapping from NCBI GeneIDs to gene symbols, e.g.
        downloaded from `https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz`.
        If this is not available as a local file, it is treated as an URL and
        downloaded to the `working_directory` if necessary, see below.
    tax_id
        The NCBI taxonomy ID to filter the `gene_info_file` for.
    GO_obo_file
        File containing the Gene Ontology data, e.g. downloaded from
        `http://purl.obolibrary.org/obo/go/go-basic.obo analysis/go-basic.obo`.
        If this is not available as a local file, it is treated as an URL and
        downloaded to the `working_directory` if necessary, see below.
    gene2GO_file
        File containing a mapping from NCBI GeneIDs to Gene Ontology data, e.g.
        downloaded from `https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz`.
        If this is not available as a local file, it is treated as an URL and
        downloaded to the `working_directory` if necessary, see below.
    working_directory
        Directory where to buffer downloaded files. If a file of the same name
        already exists in this directory, it is not downloaded again.
        
    Returns
    -------
    Returns a :class:`~goatools.goea.go_enrichment_ns:GOEnrichmentStudyNS` and\
    a :class:`~pandas.Series` mapping gene symbols to gene ids. Both are needed\
    to run the enrichment analyses. For convenience, they are also buffered as\
    global objects and used automatically.
    
    """
    
    if not HAVE_GOA:
        raise ImportError('To use the goatools wrapper, the goatools package must be installed!')
    
    gene_index = pd.Index(gene_index)
    
    # download files if necessary
    gene_info_file = _get_downloaded_file(gene_info_file, working_directory)
    GO_obo_file = _get_downloaded_file(GO_obo_file, working_directory)
    gene2GO_file = _get_downloaded_file(gene2GO_file, working_directory)

    # setup the symbol2geneID mapping
    symbol2geneID = pd.read_table(gene_info_file)
    symbol2geneID = symbol2geneID[symbol2geneID['#tax_id'] == tax_id]
    first = symbol2geneID.drop_duplicates(subset='Symbol',keep='first').set_index('Symbol')["GeneID"]
    last = symbol2geneID.drop_duplicates(subset='Symbol',keep='last').set_index('Symbol')["GeneID"]
    if (first[gene_index.intersection(first.index)].dropna() != last[gene_index.intersection(last.index)].dropna()).any():
        print('WARNING: There were non-unique assignments of gene symbols to gene ids. This can potentially influence the result of the GO term enrichment analysis.')
    symbol2geneID = first
    symbol2geneID = symbol2geneID[gene_index.intersection(symbol2geneID.index)]

    obodag = GODag(GO_obo_file)

    _, fin_gene2go = tempfile.mkstemp()
    with open(fin_gene2go, 'wb') as tmp_file:
        with gzip.open(gene2GO_file, 'rb') as zip_file:
            shutil.copyfileobj(zip_file, tmp_file)
    objanno = Gene2GoReader(fin_gene2go, taxids=[tax_id])
    os.remove(fin_gene2go)
    ns2assoc = objanno.get_ns2assc()

    global _symbol2geneID, _goeaobj
    
    _symbol2geneID = symbol2geneID
    _goeaobj = GOEnrichmentStudyNS(_symbol2geneID.to_numpy(), ns2assoc, obodag, propagate_counts = False, alpha = 0.05, methods = ['fdr_bh'])
    
    return _goeaobj, _symbol2geneID

def run_goa_analysis(
    genes,
    goeaobj=None,
    symbol2geneID=None,
):
    """\
    Run a GO analysis. This is a convenience wrapper around the goatools
    package [Klopfenstein18]_ and like goatools performs the enrichment
    analysis independent of the availability of webservices using a databases
    downloaded once for reproducibility.
    
    Parameters
    ----------
    genes
        A single list of gene symbols or a dict-like of gene lists.
    goeaobj
        A :class:`~goatools.goea.go_enrichment_ns:GOEnrichmentStudyNS` like the
        one returned by :func:`~tacco.tools.setup_goa_analysis`. If `None`, uses the
        instance buffered by the last call to :func:`~tacco.tools.setup_goa_analysis`.
    symbol2geneID
        a :class:`~pandas.Series` mapping gene symbols to gene ids. If `None`,
        uses the instance buffered by the last call to
        :func:`~tacco.tools.setup_goa_analysis`.
        
    Returns
    -------
    Returns a :class:`~pandas.DataFrame` containing the result of the\
    enrichment analysis.
    
    """
    global _symbol2geneID, _goeaobj
    
    if not HAVE_GOA:
        raise ImportError('To use the goatools wrapper, the goatools package must be installed!')
    
    if symbol2geneID is None:
        if _symbol2geneID is None:
            raise ValueError('"goa.perform_goa_analysis" was run before "goa.setup_goa_analysis"!')
        symbol2geneID = _symbol2geneID
    
    if goeaobj is None:
        if _goeaobj is None:
            raise ValueError('"goa.perform_goa_analysis" was run before "goa.setup_goa_analysis"!')
        goeaobj = _goeaobj

    if not hasattr(genes,'items'):
        genes = {'':genes}

    group_list = []
    p_value_list = []
    go_term_list = []
    go_name_list = []
    go_enrichment_list = []
    go_ns_list = []
    genes_list = []
    for group, ranked_genes in genes.items():
        ranked_genes = pd.Index(ranked_genes)
        ranked_genes = ranked_genes[ranked_genes.isin(symbol2geneID.index)]
        geneids_study = list(ranked_genes.map(symbol2geneID).to_numpy())
        goea_results = goeaobj.run_study(geneids_study)
        for r in goea_results:
            if r.p_fdr_bh < 0.05:
                group_list.append(group)
                p_value_list.append(r.p_fdr_bh)
                go_term_list.append(r.GO)
                go_name_list.append(r.name)
                go_enrichment_list.append('enriched' if r.enrichment == 'e' else 'purified')
                go_ns_list.append(r.goterm.namespace)
                genes_list.append(list(symbol2geneID.index[symbol2geneID.isin(r.study_items)]))
    
    return pd.DataFrame({'group':group_list,'GO_namespace':go_ns_list,'p_value':p_value_list,'finding':go_enrichment_list,'GO_term':go_term_list,'GO_name':go_name_list,'genes':genes_list})


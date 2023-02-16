import pickle
import gzip
import base64

def string_encode(
    data,
):

    """\
    Helper function to freeze complex result objects to support regression
    testing. The function freezes objects to a compressed string representation
    which can be included in the test source code.

    WARNING: The functionality is based on pickle and therefore the thawing
    becomes impossible if the object implementation changes, e.g. by
    incompatible version upgrades of packages, like Anndata.
    
    Parameters
    ----------
    data
        The object to freeze.
        
    Returns
    -------
    Returns a compressed string representation of the object to freeze.
    
    """

    return base64.b64encode(gzip.compress(pickle.dumps(data)))

def string_decode(
    data,
):

    """\
    Helper function to thaw objects frozen by :func:`~string_encode`.

    WARNING: The functionality is based on pickle and therefore the thawing
    becomes impossible if the object implementation changes, e.g. by
    incompatible version upgrades of packages, like Anndata.
    
    Parameters
    ----------
    data
        The compressed string representation of the object to freeze.
        
    Returns
    -------
    Returns the thawed object.
    
    """

    return pickle.loads(gzip.decompress(base64.b64decode(data)))


from scipy.stats import rankdata,tiecorrect,distributions
import numpy as np
import warnings
from scipy.special import comb

from collections import namedtuple
MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))

def mannwhitneyu(x, y, use_continuity=True, alternative=None, exact=False):

    """
    Compute the Mann-Whitney rank test on samples x and y.
    
    The code is identical to the :func:`scipy.stats.mannwhitneyu` function, but
    with the additional "exact" parameter and functionality to calculate exact
    values for small sample sizes.
    
    """

    if alternative is None:
        warnings.warn("Calling `mannwhitneyu` without specifying "
                      "`alternative` is deprecated.", DeprecationWarning)

    x = np.asarray(x)
    y = np.asarray(y)
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return MannwhitneyuResult(np.nan, np.nan)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1*n2 + (n1*(n1+1))/2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1*n2 - u1  # remainder is U for y
    
    if alternative is None or alternative == 'two-sided':
        bigu = max(u1, u2)
    elif alternative == 'less':
        bigu = u1
    elif alternative == 'greater':
        bigu = u2
    else:
        raise ValueError("alternative should be None, 'less', 'greater' "
                         "or 'two-sided'")

    if exact and (n1*n2 < 400):
        if alternative is None:
            # This behavior, equal to half the size of the two-sided
            # p-value, is deprecated.
            bigu = bigu - n1*n2/2.0
            p = mannwhitneyu_sf(n1,n2,abs(bigu),centered=True)
        elif alternative == 'two-sided':
            bigu = bigu - n1*n2/2.0
            p = 2 * mannwhitneyu_sf(n1,n2,abs(bigu),centered=True)
        else:
            p = mannwhitneyu_sf(n1,n2,bigu,centered=False)
        
    else: # use normal approximation
        T = tiecorrect(ranked)
        if T == 0:
            raise ValueError('All numbers are identical in mannwhitneyu')
        sd = np.sqrt(T * n1 * n2 * (n1+n2+1) / 12.0)

        meanrank = n1*n2/2.0 + 0.5 * use_continuity
        z = (bigu - meanrank) / sd
        if alternative is None:
            # This behavior, equal to half the size of the two-sided
            # p-value, is deprecated.
            p = distributions.norm.sf(abs(z))
        elif alternative == 'two-sided':
            p = 2 * distributions.norm.sf(abs(z))
        else:
            p = distributions.norm.sf(z)

    u = u2
    # This behavior is deprecated.
    if alternative is None:
        u = min(u1, u2)
    return MannwhitneyuResult(u, p)

_buffer = {}
def enum_rec(m,n,N):
    # how many possibilities do I have to represent N as a sum of m unequal integers from 1 to n inclusive? the order of the integers is irrelevant.
    min_sum = (m*(m+1))//2
    if N < min_sum:
        return 0
    if m == 1:
        return N >= 1 and N <= n
    if n == 1:
        return N == 1
    key = (m,n,N)
    if key in _buffer:
        return _buffer[key]
    else:
        result = enum_rec(m-1,n-1,N-n) + enum_rec(m,n-1,N)
        _buffer[key] = result
        return result

def dist_rec(n1,n2,half=True):
    if n2 < n1:
        _n=n2
        n2=n1
        n1=_n
    minN = (n1*(n1+1))//2
    maxN = minN+n1*n2
    if not half:
        return np.array([[maxN-N,enum_rec(m=n1,n=n1+n2,N=N)] for N in range(minN,maxN+1)]).T
    _maxN = minN+(n1*n2)//2
    half1 = np.array([[maxN-N,enum_rec(m=n1,n=n1+n2,N=N)] for N in range(minN,_maxN+1)]).T
    half2 = np.array([
        half1[0,:(n1*n2+1)//2]-(n1*n2)//2-1,
        half1[1,(n1*n2-1)//2::-1]
    ])
    return np.hstack((half1,half2))

def mannwhitneyu_sf(n1,n2,u1,centered=False):
    if centered:
        u1 = u1 + n1*n2/2.0
    [unique, counts] = dist_rec(n1,n2)
    p_val = np.sum(counts[unique >= u1])/np.sum(counts)
    return p_val

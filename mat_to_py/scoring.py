
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels import robust
from scipy.interpolate import interp1d
import statkscompute as st

def parse_args(yData=None, *varargin):
    temp = np.array(yData)
    if len(yData)==0:
        print('stats:mvksdensity:BadX')

    #yData(any((np.isnan(yData), 2), [:]).lvalue = []

    if len(yData)==0:
        print('stats:mvksdensity:NanX')

    n, d = temp.shape
    xi = []
    xispecified = False
    if len(varargin) != 0:
        if not type(varargin[0]) == np.str:
            xi = varargin[0][0]
            #varargin[0] = 0
            if not len(xi)==0:
                xispecified = True


    ymin = np.min(yData,axis=0)
    ymax = np.max(yData,axis=0)
    xi = np.array(xi)
    if d == 1:
        if xispecified and not xi.shape[1]>1:
            print('stats:ksdensity:VectorRequired2')

    elif d == 2:
        if xispecified and (not xi.shape[1]>1 or xi.shape[1] != 2):
            print('stats:mvksdensity:BadXId', 2)

    else:
        if not xispecified or not xi.shape[1]>1 or xi.shape[1] != d:
            print('stats:mvksdensity:BadXId', d)


    # Process additional name/value pair arguments
    okargs_n = ['bandwidth', 'kernel', 'support', 'weights', 'function', 'cutoff', 'boundarycorrection']
    defaults_n = [[], 'normal', 'unbounded', 1 / n, 'pdf', [], 'log']
    okargs_2 = ['plotfcn']
    defaults_2 = ['surf']
    okargs_1 = [['npoints', 'NumPoints'], 'censoring']
    defaults_1 = [100, False]

    if d == 1:
        # Process additional name/value pair arguments
        okargs = [okargs_n, okargs_1]
        defaults = [defaults_n, defaults_1]
        #internal.stats.parseArgs(okargs, defaults, varargin[:])
        u, kernelname, support, weight, ftype, cutoff, bdycorr, extra = varargin[1], defaults_n[1], defaults_n[2], \
                                                                        defaults_n[3], defaults_n[4], [], defaults_n[
                                                                            6], []
        plottype = ''
    else:
        if d == 2:
            okargs = [okargs_n, okargs_2]
            defaults = [defaults_n, defaults_2]
            u, kernelname, support, weight, ftype, cutoff, bdycorr, extra = varargin[1], defaults_n[1], defaults_n[2], \
                                                                            defaults_n[3], defaults_n[4], [], \
                                                                            defaults_n[6], []
            #internal.stats.parseArgs(okargs, defaults, varargin[:])
            if not xispecified:
                npoints = 30
            else:
                npoints = []

            plottype0 = ['contour', 'plot3', 'surf', 'surfc']
            #plottype = internal.stats.getParamVal(plottype, plottype0, ('PlotFcn'))
            plottype = plottype0[0]
        else:
            u, kernelname, support, weight, ftype, cutoff, bdycorr, extra = varargin[0][2],defaults_n[1],defaults_n[2],defaults_n[3],defaults_n[4],[],defaults_n[6],[]
            #internal.stats.parseArgs(okargs_n, defaults_n, varargin[:])
            npoints = []
            plottype = ''

        cens = np.zeros(n)


    # Check for boundary correction method
    okbdycorr = ['log', 'reflection']
    bdycorr = okbdycorr[0] #internal.stats.getParamVal(bdycorr, okbdycorr, ('BoundaryCorrection'))

    isXChunk = False
    if len(extra):
        u, isXChunk = [],[] #internal.stats.parseArgs(['width', 'isXChunk'], [u, isXChunk], extra[:])

    return yData, n, d, ymin, ymax, xispecified, xi, u, npoints, kernelname, support, weight, cens, cutoff, bdycorr, ftype, plottype, isXChunk


def standardize_weight(weight=None, n=None):
    print(type(weight))
    if weight==0:
        weight = np.ones(n)
    elif type(weight) == np.float:
        weight = np.tile(weight, (1, n))
    elif np.prod(weight) != n or np.prod(weight) > len(weight):
        print('stats:ksdensity:InputSizeMismatchWeight')
    else:
        weight = weight.cT

    weight = weight / sum(weight)

    return weight
    # -----------------------------

def standardize_cens(cens=None, n=None):
    if len(cens)==0:
        cens = 0 #np.false(1, n)
    elif any(cens[:] != np.logical_and(0, cens[:] != 1)):
        print('stats:ksdensity:BadCensoring')
    elif np.prod(cens.shape) != n or np.prod(cens.shape) > len(cens):
        print('stats:ksdensity:InputSizeMismatchCensoring')
    elif all(cens):
        print('stats:ksdensity:CompleteCensoring')

    cens = cens[:]

    return cens
    # -----------------------------

def compute_finite_support(support=None, ymin=None, ymax=None, d=None):
    if support.isnumeric():
        if d == 1:
            if np.prod(support.shape) != 2:
                print('stats:ksdensity:BadSupport1')

            L = support(1)
            U = support(2)
        else:
            if support.shape[0] != 2 or support.shape[1] != d:
                print('stats:mvksdensity:BadSupport', d)

            L = support[:,0]
            U = support[:,1]

        if any(L >= ymin) or any(U <= ymax):
            print('stats:ksdensity:BadSupport2')

    elif type(support) == np.str and not len(support)==0:
        okvals = ['unbounded', 'positive']

        support = okvals[0] #validatestring(support, okvals)


        if support=='unbounded':
            L = np.tile(-np.float('inf'), (1, d)) #-Inf(1, d)
            U = np.tile(np.float('inf'), (1, d))#Inf(1, d)
        else:
            L = np.zeros(1, d)
            U = np.float('inf') #Inf(1, d)

        if (support == 'positive') and any(ymin <= 0):
            print('stats:ksdensity:BadSupport4')
    else:
        print('stats:ksdensity:BadSupport5')

    return L,U


def apply_support(yData=None, L=None, U=None, d=None, bdycorr=None):
    # Compute transformed values of data
    if bdycorr == 'log':
        idx_unbounded =np.where((L==-np.float('inf')) & (U==np.float('inf'))) #find(L == logical_and(-Inf, U == Inf))
        idx_positive = np.where((L==0) & (U==np.float('inf'))) #find(L == logical_and(0, U == Inf))
        idx_bounded = np.setdiff1d(idx_unbounded[1][:d], idx_positive[1][:d])

        ty = np.zeros((yData.shape[0],yData.shape[1]))
        if len(idx_unbounded[1])>0:
            for j in range(0,len(yData)): # unbounded support
                for i in idx_unbounded[1]:
                    ty[j][i] = yData[j][i]

        if len(idx_positive[1])>0:        # positive support
            for i in idx_positive:
                for j in range(0, len(yData)):
                    ty[i][j] = np.log(yData[i][j])

        """if len(idx_bounded)>0:        # finite support [L, U]
            for i in idx_bounded:
                for j in range(0, len(yData)):
                    ty[i][j] = np.log((yData[i][j]-L[i])/(U[i]-yData[i][j]))"""

    else:
        ty = yData

    return ty


def apply_censoring_get_bandwidth(cens=None, yData=None, ty=None, n=None, ymax=None, weight=None, u=None, d=None):

    # Deal with censoring
    iscensored = any(cens)
    if iscensored:
        # Compute empirical cdf and create an equivalent weighted sample
        F, XF = ECDF(ty, 'censoring', cens, 'frequency', weight)
        weight = np.diff(F).T
        ty = XF[2:]
        N = sum(not cens)
        ymax = max(yData(not cens), [], 1)
        foldpoint = min(yData(np.logical_and(cens, yData >= ymax)), [], 1)    # for bias adjustment
        issubdist = not foldpoint.isempty()    # sub-distribution, integral < 1
        maxp = F[-1]
    else:
        N = n
        issubdist = False
        maxp = 1

    if not issubdist:
        foldpoint = np.float('inf')    # no bias adjustment is needed


    # Get bandwidth if not already specified
    if len(u)==0:
        if d > 2:
            print('stats:mvksdensity:BadBandwidth', d)
        else:
            if not iscensored:
                # Get a robust estimate of sigma
                sig = robust.mad(ty, 1, 1) / 0.6745
            else:
                # Estimate sigma using quantiles from the empirical cdf
                Xquant = interp1d(F, XF, [.25, .5, .75])
                if not any(np.isnan(Xquant)):
                    # Use interquartile range to estimate sigma
                    sig = (Xquant[:,2] - Xquant[:,0]) / (2 * 0.6745)
                elif not np.isnan(Xquant[:,1]):
                    # Use lower half only, if upper half is not available
                    sig = (Xquant[:,1] - Xquant[:,0]) / 0.6745
                else:
                    # Can't easily estimate sigma, just get some indication of spread
                    sig = ty[-1] - ty[0]

            idx = sig <= 0
            if any(idx):
                sig(idx).lvalue = max(ty[:, idx], [], 1) - min(ty[:,idx], [], 1)

            if sig > 0:
                # Default window parameter is optimal for normal distribution
                # Scott's rule
                u = sig * (4 / ((d + 2) * N)) ** (1 / (d + 4))
            else:
                u = np.ones(1, d)

    else:

        if len(u)<d:
            u = u * np.ones((1, d))
        elif u.shape != (d,):
            print('stats:mvksdensity:BadBandwidth', d)

    return ty, ymax, weight, u, foldpoint, maxp

class Product:
    ty: float = None
    weight: float = None
    foldpoint: float = None
    L: float = None
    U: float = None
    maxp: float = None

def mvksdensity(yData=None, *varargin):

    if len(varargin) > 1:
        varargin = list(varargin)

    [yData, n, d, ymin, ymax, xispecified, xi, u, npoints, kernelname, support, weight, cens, cutoff, bdycorr, ftype, plottype, isXChunk] = parse_args(yData, varargin[:])

    weight = standardize_weight(weight, n)
    if d == 1:
        cens = standardize_cens(cens, n)


    L, U = compute_finite_support(support, ymin, ymax, d)
    ty = apply_support(yData, L, U, d, bdycorr)

    ty, ymax, weight, u, foldpoint, maxp = apply_censoring_get_bandwidth(cens, yData, ty, n, ymax, weight, u, d)

    fout, xout, u = st.statkscompute(ftype, xi, xispecified, npoints, u, L, U, weight, cutoff, kernelname, ty, yData, foldpoint, maxp, d, isXChunk, bdycorr)


    return fout  #, xout, u, plottype, ksinfo

def scoring(sel_feature=None, pts=None, pa=None, bw=None):

    f = mvksdensity(sel_feature, pts, 'Bandwidth', pa * bw)
    score = abs(np.log10(f))
    score_01 = score
    indx_noninf = np.where(score != float('inf'))
    indx_inf = np.where(score == float('inf'))

    score_min = np.min(score)
    score_max = np.max(score)
    for i in indx_noninf[0]:
        score_01[i] = (score[i] - score_min) / (score_max - score_min)
    for i in indx_inf[0]:
        score_01[i] = 1

    return score_01, f
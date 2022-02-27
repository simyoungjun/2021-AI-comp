import numpy as np
import scipy.stats as stats



def statkscompute(ftype=None, xi=None, xispecified=None, m=None, u=None, L=None, U=None, weight=None, cutoff=None, kernelname=None, ty=None, yData=None, foldpoint=None, maxp=None, *varargin):
    #STATKSCOMPUTE Perform computations for kernel smoothing density
    #estimation.

    #   Copyright 2008-2017 The MathWorks, Inc.


    d = varargin[0]
    isXChunk = varargin[1]
    bdycorr = varargin[2]
    if bdycorr=="":
        bdycorr = 'log'

    kernel, iscdf, kernelcutoff, kernelname, ftype = kernelname,0,4,kernelname,ftype #statkskernelinfo(ftype, kernelname, d)

    if cutoff == 0:
        cutoff = kernelcutoff

    # Inverse cdf is special, so deal with it here
    """if ftype == 'icdf':    # d==1
        # Put the foldpoint and yData on the transformed scale
        yData = yData
        yData = transform(yData, L, U, 1, bdycorr)
        foldpoint = transform(foldpoint, L, U, 1, bdycorr)

        # Compute on that scale
        [fout, xout] = compute_icdf(xi, xispecified, m, u, L, U, weight, cutoff, kernelname, ty, yData, foldpoint, maxp, 1, bdycorr)

        # Transform back - f(p) is the icdf at p
        if bdycorr == 'log':
            fout = untransform(fout, L, U)

    else:"""
    [fout, xout, u] = compute_pdf_cdf(xi, xispecified, m, L, U, weight, kernel, cutoff, iscdf, u, ty, foldpoint, d, isXChunk, bdycorr)


    return fout, xout, u


def compute_pdf_cdf(xi=None, xispecified=None, m=None, L=None, U=None, weight=None, kernel=None, cutoff=None, iscdf=None, u=None, ty=None, foldpoint=None, d=None, isXChunk=None, bdycorr=None):

    foldwidth = 3 #min(cutoff, 3)
    issubdist = 0 #isfinite(foldpoint)

    # Compute transformed values of evaluation points that are in bounds
    if d == 1:
        fout = np.zeros(len(xi))    # f has the same orientation as xi
        xisize = len(xi)
    else:
        xisize = xi.shape[0]
        fout = np.zeros((xisize, 1))

    """    if iscdf and all(isfinite(U)):
        fout(all(bsxfun(@, xi, U), 2)).lvalue = sum(weight)"""

    xout = xi
    if d == 1:
        xi = xi

    #if (L == -np.float('inf')) & (U == np.float('inf')):    # unbounded support
    inbounds = np.ones((xisize, 1))
    """elif (L == 0) and (U == np.float('inf')):    # positive support
        inbounds = all(xi > 0, 2)
        xi = xi #(inbounds, [:])"""
    """else:    # finite support [L, U]
        inbounds = np.where(all(bsxfun(@, xi, L), 2) & all(bsxfun(@, xi, U), 2))
        xi = xi(inbounds, mslice[:])"""

    txi = transform(xi, L, U, d, bdycorr)
    if d == 1:
        foldpoint = transform(foldpoint, L, U, 1, bdycorr)

    # If the density is censored at the end, add new points so that we can fold
    # them back across the censoring point as a crude adjustment for bias.
    if issubdist:
        needfold = (txi >= foldpoint - foldwidth * u)
        txifold = (2 * foldpoint) - txi(needfold)
        nfold = sum(needfold)
    else:
        nfold = 0

    if len(xi) == 0:
        f = xi
    else:
        # Compute kernel estimate at the requested points
        f = dokernel(iscdf, txi, ty, u, weight, kernel, cutoff, d, L, U, xi, bdycorr)

        # If we need extra points for folding, do that now
        if nfold > 0:
            # Compute the kernel estimate at these extra points
            ffold = dokernel(iscdf, txifold, ty, u, weight, kernel, cutoff, d, L, U, xi(needfold), bdycorr)
            if iscdf:
                # Need to use upper tail for cdf at folded points
                ffold = sum(weight) - ffold

            # Fold back over the censoring point
            f(needfold).lvalue = f(needfold) + ffold

            if iscdf:
                # For cdf, extend last value horizontally
                maxf = max(f(txi <= foldpoint))
                f(txi > foldpoint).lvalue = maxf
            else:
                # For density, define a crisp upper limit with vertical line
                f(txi > foldpoint).lvalue = 0
                if not xispecified:
                    xi[+1] = xi[-1]
                    f[+ 1] = 0
                    inbounds[+ 1] = True


    if iscdf:
        if not isXChunk:
            # Guard against roundoff.  Lower boundary of 0 should be no problem.
            f = min(1, f)

    else:
        f = f / np.prod(u)

    fout = f
    if d == 1:
        xout(inbounds).lvalue = xi

    return fout, xout, u


def dokernel(iscdf=None, txi=None, ty=None, u=None, weight=None, kernel=None, cutoff=None, d=None, L=None, U=None, xi=None, bdycorr=None):
    # Now compute density estimate at selected points
    blocksize = 100 #3e4
    if d == 1:
        m = len(txi)
        n = len(ty)
    else:
        m = txi.shape[0]
        n = ty.shape[0]
    #[0,0,0,0,0,0]
    #needUntransform = np.logical_and((not np.logical_or((np.logical_and((L==np.float('inf')), L < 0)), not (U==np.float('inf')))), not np.logical_and(iscdf, (bdycorr=='log')))
    reflectionPDF = (bdycorr == 'reflection') and not iscdf #0
    reflectionCDF = (bdycorr == 'reflection') and iscdf #0

    if n * m <= blocksize and not iscdf:
        # For small problems, compute kernel density estimate in one operation
        ftemp = np.ones((n, m))
        for i in range(0,d):
            z1 = np.tile(txi[i], (n, 1))
            z2 = np.tile(ty[i], (1, m))

            z = (np.tile(txi[i], (n, 1)) - np.tile(ty[i], (1, m)))/u[i]

            if reflectionPDF:
                zleft = (np.tile(txi[i], (n, 1)) + np.tile(ty[i], (1, m)) - 2 * L[i]) / u[i]
                zright = (np.tile(txi[i], (n, 1)) + np.tile(ty[i], (1, m)) - 2 * U[i]) / u(i)
                #f = feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright)
            else:
                f = 0
                #f = feval(kernel, z)

            # Apply reverse transformation and create return value of proper size
            #if needUntransform[i]:
                #f = untransform_f(f, L[i], U[i], xi[i])

            #ftemp = ftemp *elmul* f

        f = weight * ftemp
    elif d == 1:
        # For large problems, try more selective looping

        # First sort y and carry along weights
        [ty, idx] = np.sort(ty)
        weight = weight(idx)

        # Loop over evaluation points
        f = np.zeros((1, m))

        if cutoff == np.float('inf'):
            #if reflectionCDF:
                #fc = compute_CDFreduction(L, U, u, np.float('inf'), n, ty, weight, kernel)

            for k in range(0,m):
                # Sum contributions from all
                z = (txi(k) - ty) / u
                if reflectionPDF:
                    zleft = (txi(k) + ty - 2 * L) / u
                    zright = (txi(k) + ty - 2 * U) / u
                    #f(k).lvalue = weight * (feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright))
                elif reflectionCDF:
                    zleft = (txi(k) + ty - 2 * L) / u
                    zright = (txi(k) + ty - 2 * U) / u
                    #fk = weight * (feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright))
                    #f(k).lvalue = fk - fc
                else:
                    f = 0
                    #f(k).lvalue = weight * feval(kernel, z)


        else:
            # Sort evaluation points and remember their indices
            [stxi, idx] = np.sort(txi)

            jstart = 1        # lowest nearby point
            jend = 1        # highest nearby point
            halfwidth = cutoff * u

            # Calculate reduction for reflectionCDF
            #if reflectionCDF:
                #fc = compute_CDFreduction(L, U, u, halfwidth, n, ty, weight, kernel)


            for k in range(0,m):
                # Find nearby data points for current evaluation point
                lo = stxi(k) - halfwidth
                while (ty(jstart) < lo and jstart < n):
                    jstart = jstart + 1

                hi = stxi(k) + halfwidth
                jend = max(jend, jstart)
                while (ty(jend) <= hi and jend < n):
                    jend = jend + 1

                nearby = jstart,jend

                # Sum contributions from these points
                z = (stxi(k) - ty(nearby)) / u
                if reflectionPDF:
                    zleft = (stxi(k) + ty(nearby) - 2 * L) / u
                    zright = (stxi(k) + ty(nearby) - 2 * U) / u
                    #fk = weight(nearby) * (feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright))
                elif reflectionCDF:
                    zleft = (stxi(k) + ty(nearby) - 2 * L) / u
                    zright = (stxi(k) + ty(nearby) - 2 * U) / u
                    #fk = weight(nearby) * feval(kernel, z)
                    #fk = fk + sum(weight(mslice[1:jstart - 1]))
                    if jstart == 1:
                        fk = 0
                        #fk = fk + weight(nearby) * feval(kernel, zleft)
                        #fk = fk + sum(weight(mslice[jend + 1:end]))
                    else:
                        fk = fk + sum(weight)

                    if jend == n:
                        fk = 0
                        #fk = fk + weight(nearby) * feval(kernel, zright)

                    fk = fk #- fc
                elif not iscdf:
                    fk = 0
                    #fk = weight(nearby) * feval(kernel, z)
                elif iscdf:
                    fk = 0
                    #fk = weight(nearby) * feval(kernel, z)
                    #fk = fk + sum(weight(mslice[1:jstart - 1]))

                #f(k).lvalue = fk


            # Restore original x order
            #f(idx).lvalue = f


        #if needUntransform:
            #f = untransform_f(f, L, U, xi)

    else:    # d > 1
        # Calculate reduction for reflectionCDF
        if reflectionCDF:
            cutoff = np.float('inf')
            fc = np.zeros((n, d))
            #for j in range(0,d):
                #fc[j]= compute_CDFreduction_mv(L(j), U(j), u(j), ty(j), kernel)

        if cutoff == np.float('inf'):
            f = np.zeros((1, m))
            for i in range(0,m):
                ftemp = np.ones((n, 1))
                for j in range(0,d):
                    z = (txi[i][j] - ty[j]) / u[j]
                    if reflectionPDF:
                        zleft = (txi[i][j] + ty[:, j] - 2 * L[j]) / u[j]
                        zright = (xi[i][j] + ty[:, j] - 2 * U[j]) / u[j]
                        fk = stats.norm(0, 1).pdf(z) + stats.norm(0, 1).pdf(zleft) + stats.norm(0, 1).pdf(zright)
                        #fk = feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright)
                    elif reflectionCDF:
                        zleft = (txi[i][j] + ty[:,j] - 2 * L[j]) / u[j]
                        zright = (xi[i][j] + ty[:,j] - 2 * U[j]) / u[j]
                        fk = stats.norm(0, 1).pdf(z) + stats.norm(0, 1).pdf(zleft) + stats.norm(0, 1).pdf(zright)
                        #fk = feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright)

                        fk = fk - fc[:,j]
                    else:
                        fk = stats.norm(0, 1).pdf(z)
                        #fk = feval(kernel, z)

                    #if needUntransform(j):
                        #fk = untransform_f(fk, L(j), U(j), xi(i, j))

                    ftemp = ftemp * fk

                f[i] = weight * ftemp

        else:
            halfwidth = u * 4
            index = (n,1)
            f = np.zeros((1, m))
            for i in range(0,m):
                Idx = np.ones((n, 1))
                cdfIdx =  np.ones((n, 1))
                cdfIdx_allBelow =  np.ones((n, 1))

                for j in range(0,d):
                    dist = []
                    currentIdx = []
                    for k in range(0,len(ty)):
                        dist.append(txi[i][j] -ty[k][j])
                        if abs(dist[k]) <= halfwidth[j]:
                            currentIdx.append(1)
                        else:
                            currentIdx.append(0)
                    #dist = txi(i, j) - ty(j)  # txi - ty

                    #currentIdx = abs(dist) <= halfwidth(j)
                    Idx = currentIdx #logical_and(currentIdx, Idx)                # pdf boundary
                    if iscdf:
                        currentCdfIdx = dist >= -halfwidth(j)
                        cdfIdx = currentIdx#logical_and(currentCdfIdx, cdfIdx)                    # cdf boundary1, equal or below the query point in all dimension
                        currentCdfIdx_below = dist - halfwidth(j) > 0
                        cdfIdx_allBelow = currentCdfIdx_below #logical_and(currentCdfIdx_below, cdfIdx_allBelow)                    # cdf boundary2, below the pdf lower boundary in all dimension

                if not iscdf:
                    nearby = [i for i, value in enumerate(Idx) if value == 1]#index(Idx)
                else:
                    nearby = [i for i, value in enumerate(Idx) if value == 1]
                    #nearby = index(logical_and((logical_or(Idx, cdfIdx)), (not cdfIdx_allBelow)))

                if len(nearby)>0:
                    ftemp = np.ones((len(nearby), 1))
                    for k in range(0,d):
                        z = []
                        for q in nearby:
                            z.append((txi[i][k] - ty[q][k]) / u[k])
                        if reflectionPDF:
                            zleft = (txi(i, k) + ty(nearby, k) - 2 * L(k)) / u(k)
                            zright = (txi(i, k) + ty(nearby, k) - 2 * U(k)) / u(k)
                            fk = stats.norm(0, 1).pdf(z) + stats.norm(0, 1).pdf(zleft) + stats.norm(0, 1).pdf(zright)
                            #fk = feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright)
                        else:
                            fk = stats.norm(0, 1).pdf(z)
                            #fk = feval(kernel, z)

                        #if needUntransform(k):
                            #fk = untransform_f(fk, L(k), U(k), xi(i, k))
                        fk = np.array(fk)
                        fk = fk.reshape(fk.shape[0],1)
                        ftemp =ftemp*fk
                    sum1 = 0
                    weight = np.array(weight)
                    weight = weight.reshape(1,weight.shape[1])
                    for l in nearby:
                        for cnt in range(0,len(ftemp)):
                            temp = ((weight[0][l]) * (ftemp[cnt]))
                            sum1 = sum1 + temp[0]

                    f[0][i] = sum1 # weight(nearby) * ftemp

                if iscdf and any(cdfIdx_allBelow):
                    f[i] = f[i] + sum(weight(cdfIdx_allBelow))

    return f.T


"""
def compute_icdf(xi=None, xispecified=None, m=None, u=None, Lin=None, Uin=None, weight=None, cutoff=None, kernelname=None, ty=None, yData=None, foldpoint=None, maxp=None, d=None, bdycorr=None):
    if isequal(bdycorr, mstring('log')):
        L = -Inf    # log correction has no bounds
        U = Inf
    else:
        L = Lin
        U = Uin

    if xispecified:
        p = xi
    else:
        p = (mslice[1:m]) / (m + 1)


    [Fi, xi, cutoff, u] = compute_initial_icdf(m, u, L, U, weight, cutoff, kernelname, ty, yData, foldpoint, d, bdycorr)

    [kernel_c, iscdf_c] = statkskernelinfo(mstring('cdf'), kernelname, d)
    [kernel_p, iscdf_p] = statkskernelinfo(mstring('pdf'), kernelname, d)


    # Get starting values for ICDF(p) by inverse linear interpolation of
    # the gridded CDF, plus some clean-up
    x1 = interp1(Fi, xi, p)# interpolate for p in a good range
    x1(logical_and(isnan(x1), p < min(Fi))).lvalue = min(xi)# use lowest x if p>0 too low
    x1(logical_and(isnan(x1), p > max(Fi))).lvalue = max(xi)# use highest x if p<1 too high
    x1(logical_or(p < 0, p > maxp)).lvalue = NaN# out of range
    x1(p == 0).lvalue = L# use lower bound if p==0
    x1(p == maxp).lvalue = U# and upper bound if p==1 or other max

    # Now refine the ICDF using Newton's method for cases with 0<p<1
    notdone = find(logical_and(p > 0, p < maxp))
    maxiter = 100
    min_dF0 = sqrt(eps(mclass(p)))
    for iter in mslice[1:maxiter]:
        if isempty(notdone):
            break

        x0 = x1(notdone)

        # Compute cdf and derivative (pdf) at this value
        F0 = compute_pdf_cdf(x0, true, m, L, U, weight, kernel_c, cutoff, iscdf_c, u, ty, foldpoint, d, false, bdycorr)
        dF0 = compute_pdf_cdf(x0, true, m, L, U, weight, kernel_p, cutoff, iscdf_p, u, ty, foldpoint, d, false, bdycorr)

        # dF0 is always >= 0. Prevent dF0 from becoming too small.
        dF0 = max(dF0, min_dF0)

        # Perform a Newton's step
        dp = p(notdone) - F0
        dx = dp /eldiv/ dF0
        x1(notdone).lvalue = x0 + dx

        # Continue if the x and function (probability) change are large
        notdone = notdone(logical_and(abs(dx) > 1e-6 * abs(x0), logical_and(abs(dp) > 1e-8, x0 < foldpoint)))

    if not len(notdone)==0:
        print('stats:ksdensity:NoConvergence')

    return xi,p
"""
"""
def compute_initial_icdf(m=None, u=None, L=None, U=None, weight=None, cutoff=None, kernelname=None, ty=None, yData=None, foldpoint=None, d=None, bdycorr=None):
    # To get starting x values for the ICDF evaluated at p, first create a
    # grid xi of values spanning the data on which to evaluate the CDF
    sy = sort(yData)
    xi = linspace(sy(1), sy(end), 100)

    # Estimate the CDF on the grid
    [kernel_c, iscdf_c, kernelcutoff] = statkskernelinfo(mstring('cdf'), kernelname, d)

    [Fi, xi, u] = compute_pdf_cdf(xi, true, m, L, U, weight, kernel_c, cutoff, iscdf_c, u, ty, foldpoint, d, false, bdycorr)

    if isequal(kernelname, mstring('normal')):
        # Truncation for the normal kernel creates small jumps in the CDF.
        # That's not a problem for the CDF, but it causes convergence problems
        # for ICDF calculation, so use a cutoff large enough to make the jumps
        # smaller than the convergence criterion.
        cutoff = max(cutoff, 6)
    else:
        # Other kernels have a fixed finite width.  Ignore any requested
        # truncation for these kernels; it would cause convergence problems if
        # smaller than the kernel width, and would have no effect if larger.
        cutoff = kernelcutoff


    # If there are any gaps in the data wide enough to create regions of
    # exactly zero density, include points at the edges of those regions
    # in the grid, to make sure a linear interpolation smooth of the gridded
    # CDF captures them as constant
    halfwidth = cutoff * u
    gap = find(diff(sy) > 2 * halfwidth)
    if not isempty(gap):
        sy = sy(mslice[:]).cT
        xi = sort(mcat([xi, sy(gap) + halfwidth, sy(gap + 1) - halfwidth]))
        [Fi, xi, u] = compute_pdf_cdf(xi, true, m, L, U, weight, kernel_c, cutoff, iscdf_c, u, ty, foldpoint, d, false, bdycorr)

    # Find any regions where the CDF is constant, these will cause problems
    # inverse interpolation for x at p
    t = (diff(Fi) == 0)
    if any(t):
        # Remove interior points in constant regions, they're unnecessary
        s = (logical_and(mcat([false, t]), mcat([t, false])))
        Fi(s).lvalue = mcat([])
        xi(s).lvalue = mcat([])
        # To make Fi monotonic, nudge up the CDF value at the end of each
        # constant region by the smallest amount possible.
        t = 1 + find(diff(Fi) == 0)
        Fi(t).lvalue = Fi(t) + eps(Fi(t))
        # If the CDF at the point following is that same value, just remove
        # the nudge.
        if (t(end) == length(Fi)):
            t(end).lvalue = mcat([]); print t

        s = t(Fi(t) >= Fi(t + 1))
        Fi(s).lvalue = mcat([])
        xi(s).lvalue = mcat([])

    return Fi, xi, cutoff, u
"""

def transform(y=None, L=None, U=None, d=None, bdycorr=None):
    if bdycorr== 'log':
        """idx_unbounded = find(L == logical_and(-Inf, U == Inf))
        idx_positive = find(L == logical_and(0, U == Inf))
        idx_bounded = setdiff(mslice[1:d], mcat([idx_unbounded, idx_positive]))"""
        idx_unbounded = np.where(
            (L == -np.float('inf')) & (U == np.float('inf')))  # find(L == logical_and(-Inf, U == Inf))
        idx_positive = np.where((L == 0) & (U == np.float('inf')))  # find(L == logical_and(0, U == Inf))
        idx_bounded = np.setdiff1d(idx_unbounded[1][:d], idx_positive[1][:d])

        x = np.zeros((y.shape[0],y.shape[1]))
        if len(idx_unbounded[1])>0:
            for j in range(0,len(y)): # unbounded support
                for i in idx_unbounded[1]:
                    x[j][i] = y[j][i]
        if len(idx_positive[1])>0:        # positive support
            for i in idx_positive:
                for j in range(0, len(y)):
                    x[i][j] = np.log(y[i][j])

        """if any(idx_bounded):
            y(mslice[:], idx_bounded).lvalue = bsxfun(@, L(mslice[:], idx_bounded), bsxfun(@, y(mslice[:], idx_bounded), U(mslice[:], idx_bounded)))

            x(mslice[:], idx_bounded).lvalue = log(bsxfun(@, y(mslice[:], idx_bounded, mslice[:]), L(mslice[:], idx_bounded))) - log(bsxfun(@, U(mslice[:], idx_bounded), y(mslice[:], idx_bounded)))
            [i, j] = find(y(mslice[:], idx_bounded) == Inf)
            x(i, idx_bounded(j)).lvalue = Inf"""

    else:
        x = y

    return x


"""
def untransform(x=None, L=None, U=None):
    if L == -Inf and U == Inf:    # unbounded support
        y = x
    elif L == 0 and U == Inf:    # positive support
        y = exp(x)
    else:    # finite support [L, U]
        t = x < 0
        y = x
        y(t).lvalue = (U * exp(x(t)) + L) /eldiv/ (exp(x(t)) + 1)
        t = not t
        y(t).lvalue = (U + L * exp(-x(t))) /eldiv/ (1 + exp(-x(t)))
    return y"""

"""def untransform_f(f=None, L=None, U=None, xi=None):
    if L == 0 and U == np.float('inf'):    # positive support
        f = bsxfun(@, f, 1. / xi.cT)
    elif U < np.float('inf'):    # bounded support
        tf = (U - L) /eldiv/ ((xi - L) *elmul* (U - xi))
        f = bsxfun(@, f, tf.cT)

    return f"""
"""
def compute_CDFreduction(L=None, U=None, u=None, halfwidth=None, n=None, ty=None, weight=None, kernel=None):
    jstart = 1
    jend = 1
    hi = L + halfwidth
    while (ty(jend) <= hi and jend < n):
        jend = jend + 1

    nearby = mslice[jstart:jend]
    z = (L - ty(nearby)) / u
    zleft = (ty(nearby) - L) / u
    zright = (L + ty(nearby) - 2 * U) / u
    if jend == n:
        fc = weight(nearby) * (feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright))
    else:
        fc = weight(nearby) * (feval(kernel, z) + feval(kernel, zleft))
        fc = fc + sum(weight(mslice[jend + 1:end]))
    return fc"""

"""
def compute_CDFreduction_mv(L=None, U=None, u=None, ty=None, kernel=None):
    z = (L - ty(mslice[:])) / u
    zleft = (ty(mslice[:]) - L) / u
    zright = (L + ty(mslice[:]) - 2 * U) / u
    fc = feval(kernel, z) + feval(kernel, zleft) + feval(kernel, zright)

    return fc"""
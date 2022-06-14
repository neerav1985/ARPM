from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, zeros, diag, prod, meshgrid, where, floor, ones, sum as npsum, sqrt, eye, sort, r_, ceil, \
    min as npmin, max as npmax, remainder, array, diagflat
from numpy.linalg import qr, solve, pinv
from scipy.special import factorial

from numgrad import numgrad

plt.style.use('seaborn')


def numdiagHess(fun, x0):
    # numdiagHess: diagonal elements of the Hessian matrix (vector of second partials)
    # usage: HD,err,finaldelta = numdiagHess((fun,x0))
    #
    # When all that you want are the diagonal elements of the hessian
    # matrix, it will be more efficient to call numdiagHess than numHess.
    # numdiagHess uses DERIVEST to provide both second derivative estimates
    # and error estimates. fun needs not be vectorized.
    #
    # arguments: (input)
    #  fun - SCALAR analytical function to differentiate.
    #        fun must be a function of the vector or array x0.
    #
    #  x0  - vector location at which to differentiate fun
    #        If x0 is an nxm array, then fun is assumed to be
    #        a function of n@m variables.
    #
    # arguments: (output)
    #  HD  - vector of second partial derivatives of fun.
    #        These are the diagonal elements of the Hessian
    #        matrix, evaluated at x0.
    #        HD will be a row vector of len x0.size.
    #
    #  err - vector of error estimates corresponding to
    #        each second partial derivative in HD.
    #
    #  finaldelta - vector of final step sizes chosen for
    #        each second partial derivative.
    #
    # Author: John D.TErrico
    # e-mail: woodchips@rochester.rr.com
    # Release: 1.0
    # Release date: 2/9/2007

    # get the size of x0 so we can reshape
    # later.
    sx = x0.shape

    # total number of derivatives we will need to take
    nx = x0.size

    HD = zeros((1, nx))
    err = HD.copy()
    finaldelta = HD.copy()
    for ind in range(nx):
        HD[0,ind], err[0,ind], finaldelta[0,ind] = derivest(lambda xi: fun(swapelement(x0.copy(), ind, xi)), x0[ind],deriv=2, vectorized=False)

    return HD, err, finaldelta
    # mainline function

# =======================================
#      sub-functions
# =======================================
def swapelement(vec, ind, val):
    # swaps val as element ind, into the vector vec
    vec[ind] = val
    return vec
    # sub-function 

# =======================================
#      sub-functions
# =======================================
def swap2(vec,ind1,val1,ind2,val2):
    # swaps val as element ind, into the vector vec
    vec[ind1] = val1
    vec[ind2] = val2
    return vec
    # sub-function 

# ============================================
# subfunction - romberg extrapolation
# ============================================
def rombextrap(StepRatio,der_init,rombexpon):
    # do romberg extrapolation for each estimate
    #
    #  StepRatio - Ratio decrease in step
    #  der_init - initial derivative estimates
    #  rombexpon - higher order terms to cancel using the romberg step
    #
    #  der_romb - derivative estimates returned
    #  errest - error estimates
    #  amp - noise amplification factor due to the romberg step
    
    srinv = 1/StepRatio
    
    # do nothing if no romberg terms
    nexpon = len(rombexpon)
    rombexpon = array(rombexpon)
    rmat = ones((nexpon+2,nexpon+1))
    if nexpon == 0:
        pass
        # rmat is simple: ones((2,1))
    elif nexpon == 1:
        # only one romberg term
        rmat[1,1] = srinv**rombexpon
        rmat[2,1] = srinv**( 2*rombexpon)
    elif nexpon == 2:
        # two romberg terms
        rmat[1,1:3] = srinv**rombexpon
        rmat[2,1:3] = srinv**( 2*rombexpon)
        rmat[3,1:3] = srinv**( 3*rombexpon)
    elif nexpon == 3:
        # three romberg terms
        rmat[1,1:4] = srinv**rombexpon
        rmat[2,1:4] = srinv**( 2*rombexpon)
        rmat[3,1:4] = srinv**( 3*rombexpon)
        rmat[4,1:4] = srinv**( 4*rombexpon)

    # qr factorization used for the extrapolation as well
    # as the uncertainty estimates
    qromb,rromb = qr(rmat)
    
    # the noise amplification is further amplified by the Romberg step.
    # amp = cond(rromb)
    
    # this does the extrapolation to a zero step size.
    ne = len(der_init)
    rhs = vec2mat(der_init,nexpon+2,max(1,ne - (nexpon+2)))
    rombcoefs = solve(rromb,qromb.T@rhs.astype(np.float64))
    der_romb = rombcoefs[0].T
    
    # uncertainty estimate of derivative prediction
    s = sqrt(npsum((rhs - rmat@rombcoefs)**2,0))
    rinv = solve(rromb,eye(nexpon+1))
    cov1 = npsum(rinv**2,1) # 1 spare dof
    errest = s.T*12.7062047361747**sqrt(cov1[0])

    return der_romb, errest
    # rombextrap

# ============================================
# subfunction - vec2mat
# ============================================
def vec2mat(vec,n,m):
    # forms the matrix M, such that M(i,j) = vec(i+j-1)
    i,j = meshgrid(arange(m),arange(n))
    ind = i+j
    mat = vec[ind]
    if n==1:
        mat = mat.T

    return mat

# ============================================
# subfunction - fdamat
# ============================================
def fdamat(sr,parity,nterms):
    # Compute matrix for fda derivation.
    # parity can be
    #   0 (one sided, all terms included but zeroth order)
    #   1 (only odd terms included)
    #   2 (only even terms included)
    # nterms - number of terms
    
    # sr is the ratio between successive steps
    srinv = 1/sr
    
    if parity==0:
        # single sided rule
        j, i = meshgrid(arange(nterms), arange(nterms))
        c = 1/factorial(arange(1,nterms+1))
        mat = c[j]*srinv**(i*(j+1))
    elif parity==1:
        # odd order derivative
        j, i = meshgrid(arange(nterms), arange(nterms))
        c = 1/factorial(arange(1,2*nterms+1,2))
        mat = c[j]*srinv**(i*(2*j))
    elif parity==2:
        # even order derivative
        j,i = meshgrid(arange(nterms),arange(nterms))
        c = 1/factorial(arange(2,2*nterms+1,2))
        mat = c[j]*srinv**((i)*(2*(j+1)))
    
    return mat
    # fdamat


def numHess(fun, x0):
    # numHess: estimate elements of the Hessian matrix (array of 2nd partials)
    # usage: hess,err = numHess((fun,x0))
    #
    # Hessian is NOT a tool for frequent use on an expensive
    # to evaluate objective function, especially in a large
    # number of dimensions. Its computation will use roughly
    # O( 6*n**2) function evaluations for n parameters.
    # 
    # arguments: (input)
    #  fun - SCALAR analytical function to differentiate.
    #        fun must be a function of the vector or array x0.
    #        fun does not need to be vectorized.
    # 
    #  x0  - vector location at which to compute the Hessian.
    #
    # arguments: (output)
    #  hess - nxn symmetric array of second partial derivatives
    #        of fun, evaluated at x0.
    #
    #  err - nxn array of error estimates corresponding to
    #        each second partial derivative in hess.
    #
    # Author: John D.TErrico (woodchips@rochester.rr.com)
    # Version: 02/10/2007
    # Original name: hessian

    # parameters that we might allow to change
    params = namedtuple('params', 'StepRatio RombergTerms')
    params.StepRatio = 2.0000001
    params.RombergTerms = 3

    # get the size of x0 so we can reshape
    # later.
    sx = x0.shape

    # was a string supplied?
    if isinstance(fun, str):
        fun = eval(fun)

    # total number of derivatives we will need to take
    nx = len(x0)

    # get the diagonal elements of the hessian (2nd partial
    # derivatives wrt each variable.)
    hess, err, *_ = numdiagHess(fun, x0)

    # form the eventual hessian matrix, stuffing only
    # the diagonals for now.
    hess = diagflat(hess)
    err = diagflat(err)
    if nx < 2:
        # the hessian matrix is 1x1. all done
        return

    # get the gradient vector. This is done only to decide
    # on intelligent step sizes for the mixed partials
    grad, graderr, stepsize = numgrad(fun, x0.copy())
    stepsize= stepsize.flatten()
    stepsize[0] = 1.
    # Get params.RombergTerms+1 estimates of the upper
    # triangle of the hessian matrix
    dfac = params.StepRatio ** (-arange(params.RombergTerms+1)).reshape(-1,1)
    for i in range(1,nx):
        for j in range(i):
            dij = zeros((params.RombergTerms + 1, 1))
            for k in range(params.RombergTerms + 1):
                dij[k] = fun(x0 + swap2(zeros((sx)), i,  dfac[k]*stepsize[i], j,  dfac[k]*stepsize[j])) + \
                         fun(x0 + swap2(zeros((sx)), i, -dfac[k]*stepsize[i], j, -dfac[k]*stepsize[j])) - \
                         fun(x0 + swap2(zeros((sx)), i,  dfac[k]*stepsize[i], j, -dfac[k]*stepsize[j])) - \
                         fun(x0 + swap2(zeros((sx)), i, -dfac[k]*stepsize[i], j,  dfac[k]*stepsize[j]))

            dij = dij / 4 / prod(stepsize[[i, j]])
            dij = dij / (dfac**2)

            # Romberg extrapolation step
            hess[i, j], err[i, j] = rombextrap(params.StepRatio, dij.flatten(), [2, 4])
            hess[j, i] = hess[i, j].copy()
            err[j, i] = err[i, j].copy()
    return hess,err

# mainline function 

# ============================================
# subfunction - check_params
# ============================================
def check_params(par):
    # check the parameters for acceptability
    #
    # Defaults
    # par.DerivativeOrder = 1
    # par.MethodOrder = 2
    # par.Style = central
    # par.RombergTerms = 2
    # par.FixedStep = []
    
    # DerivativeOrder == 1 by default
    if par.DerivativeOrder is None:
        par.DerivativeOrder = 1
    else:
        if not isinstance(par.DerivativeOrder,(np.float, np.int)) or not (par.DerivativeOrder in [1,2,3,4]):
            raise ValueError('DerivativeOrder must be scalar, one of [1 2 3 4].')
    
    # MethodOrder == 2 by default
    if par.MethodOrder is None:
        par.MethodOrder = 2
    else:
        if not isinstance(par.MethodOrder,(np.float, np.int)) or not (par.MethodOrder in [1,2,3,4]):
            raise ValueError('MethodOrder must be scalar, one of [1 2 3 4].')
        elif par.MethodOrder in [1, 3] and par.Style[0]=='c':
            raise ValueError('MethodOrder==1 or 3 is not possible with central difference methods.')
      
    # style is char
    valid = ['central', 'forward', 'backward']
    if par.Style is None:
        par.Style = 'central'
    elif not isinstance(par.Style,str):
        raise ValueError('Invalid Style: Must be character.')
    
    try:
        ind = valid.index(par.Style)
        par.Style = valid[ind]
    except ValueError:
        raise ValueError('Invalid Style: %s'%par.Style)
    
    
    # vectorized is char
    valid = [True, False]
    if par.Vectorized is None:
        par.Vectorized = True
    elif not isinstance(par.Vectorized,np.bool):
        raise ValueError('Invalid Vectorized: Must be boolean.')

    try:
        ind = valid.index(par.Vectorized)
        par.Vectorized = valid[ind]
    except ValueError:
        raise ValueError('Invalid Vectorized: %s'%par.Vectorized)
    
    # RombergTerms == 2 by default
    if par.RombergTerms is None:
        par.RombergTerms = 2
    else:
        if not isinstance(par.RombergTerms,(np.float, np.int)) or not (par.RombergTerms in [0,1,2,3]):
            raise ValueError('Rombererms must be scalar, one of [0 1 2 3].')

    # FixedStep == None by default
    if isinstance(par.FixedStep,(np.float,np.int)):
        if par.FixedStep<=0:
            raise ValueError('FixedStep must be empty or a scalar, >0')

    if not isinstance(par.FixedStep,(np.float,np.int)) and par.FixedStep is not None:
        raise ValueError('FixedStep must be empty or a scalar, >0')

    # MaxStep == 10 by default
    if par.MaxStep is None:
        par.MaxStep = 10
    elif not isinstance(par.MaxStep,(np.float,np.int,None)) or (par.MaxStep<=0):
        raise ValueError('MaxStep must be empty or a scalar, >0')

    return par # check_params

# ============================================
# Included subfunction - parse_pv_pairs
# ============================================
def parse_pv_pairs(params,pv_pairs):
    # parse_pv_pairs: parses sets of property value pairs, allows defaults
    # usage: params=parse_pv_pairs((default_params,pv_pairs))
    #
    # arguments: (input)
    #  default_params - structure, with one field for every potential
    #             property/value pair. Each field will contain the default
    #             value for that property. If no default is supplied for a
    #             given property, then that field must be empty.
    #
    #  pv_array - cell array of property/value pairs.
    #             Case is ignored when comparing properties to the list
    #             of field names. Also, any unambiguous shortening of a
    #             field/property name is allowed.
    #
    # arguments: (output)
    #  params   - parameter struct that reflects any updated property/value
    #             pairs in the pv_array.
    #
    # Example usage:
    # First, set default values for the parameters. Assume we
    # have four parameters that we wish to use optionally in
    # the function examplefun.
    #
    #  - viscosity, which will have a default value of 1
    #  - volume, which will default to 1
    #  - pie - which will have default value 3.141592653589793
    #  - description - a text field, left empty by default
    #
    # The first argument to examplefun is one which will always be
    # supplied.
    #
    #   function examplefun((dummyarg1,varargin))
    #   params.Viscosity = 1
    #   params.Volume = 1
    #   params.Pie = 3.141592653589793
    #
    #   params.Description = .T.T
    #   params=parse_pv_pairs((params,varargin))
    #   params
    #
    # Use examplefun, overriding the defaults for pie, viscosity
    # and description. The volume parameter is left at its default.
    #
    #   examplefun((rand(10)),vis,10,pie,3,Description,.THello world.T)
    #
    # params = 
    #     Viscosity: 10
    #        Volume: 1
    #           Pie: 3
    #   Description: .THello world.T
    #
    # Note that capitalization was ignored, and the property viscosity
    # was truncated as supplied. Also note that the order the pairs were
    # supplied was arbitrary.
    
    npv = len(pv_pairs)
    n = npv/2
    
    if n!=floor(n):
        raise ValueError('Property/value pairs must come in PAIRS.')
    
    if n<=0:
        # just return the defaults
        return
    
    
    if not isinstance(params,namedtuple):
        raise ValueError('No structure for defaults was supplied.')
    
    
    # there was at least one pv pair. process any supplied
    propnames = params._fields
    lpropnames = list(map(str.lower,propnames))
    for i in range(n):
        p_i = str.lower(pv_pairs[2*i-1])
        v_i = pv_pairs[2*i]
      
        ind = p_i==lpropnames
        if not ind:
            ind = where(p_i==lpropnames,len(p_i))
            if not ind:
                raise ValueError('No matching property found for: %s'%pv_pairs[2*i-1])
            elif len(ind)>1:
                raise ValueError('Ambiguous property name: %s'%pv_pairs[2*i-1])
    
      
        p_i = propnames[ind]
      
        # override the corresponding default in params
        params.v_i = p_i,v_i ##ok

    return params 
    # parse_pv_pairs

def derivest(fun, x0, deriv=1, vectorized=True,methodorder=4,maxstep=100,rombergterms=2,style='central',stepratio=2.0000001):
    # DERIVEST: estimate the n'th derivative of fun at x0, provide an error estimate
    # usage: der,errest = DERIVEST(fun,x0)  # first derivative
    # usage: der,errest = DERIVEST(fun,x0,prop1,val1,prop2,val2,...)
    #
    # Derivest will perform numerical differentiation of an
    # analytical function provided in fun. It will not
    # differentiate a function provided as data. Use gradient
    # for that purpose, or differentiate a spline model.
    #
    # The methods used by DERIVEST are finite difference
    # approximations of various orders, coupled with a generalized
    # (multiple term) Romberg extrapolation. This also yields
    # the error estimate provided. DERIVEST uses a semi-adaptive
    # scheme to provide the best estimate that it can by its
    # automatic choice of a differencing interval.
    #
    #
    #
    # Arguments (input)
    #  fun - function to differentiate. May be an inline function,
    #        anonymous, or an m-file. fun will be sampled at a set
    #        of distinct points for each element of x0. If there are
    #        additional parameters to be passed into fun, then use of
    #        an anonymous function is recommed.
    #
    #        fun should be vectorized to allow evaluation at multiple
    #        locations at once. This will provide the best possible
    #        speed. IF fun is not so vectorized, then you MUST set
    #        vectorized property to no, so that derivest will
    #        then call your function sequentially instead.
    #
    #        Fun is assumed to return a result of the same
    #        shape as its input x0.
    #
    #  x0  - scalar, vector, or array of points at which to
    #        differentiate fun.
    #
    # Additional inputs must be in the form of property/value pairs.
    #  Properties are character strings. They may be shortened
    #  to the extent that they are unambiguous. Properties are
    #  not case sensitive. Valid property names are:
    #
    #  DerivativeOrder, MethodOrder, Style, RombergTerms
    #  FixedStep, MaxStep
    #
    #  All properties have default values, chosen as intelligently
    #  as I could manage. Values that are character strings may
    #  also be unambiguously shortened. The legal values for each
    #  property are:
    #
    #  DerivativeOrder - specifies the derivative order estimated.
    #        Must be a positive integer from the set [1,2,3,4].
    #
    #        DEFAULT: 1 (first derivative of fun)
    #
    #  MethodOrder - specifies the order of the basic method
    #        used for the estimation.
    #
    #        For central methods, must be a positive integer
    #        from the set [2,4].
    #
    #        For forward or backward difference methods,
    #        must be a positive integer from the set [1,2,3,4].
    #
    #        DEFAULT: 4 (a second order method)
    #
    #        Note: higher order methods will generally be more
    #        accurate, but may also suffere more from numerical
    #        problems.
    #
    #        Note: First order methods would usually not be
    #        recommed.
    #
    #  Style - specifies the style of the basic method
    #        used for the estimation. central, forward,
    #        or backwards difference methods are used.
    #
    #        Must be one of Central, forward, backward.
    #
    #        DEFAULT: Central
    #
    #        Note: Central difference methods are usually the
    #        most accurate, but sometiems one must not allow
    #        evaluation in one direction or the other.
    #
    #  RombergTerms - Allows the user to specify the generalized
    #        Romberg extrapolation method used, or turn it off
    #        completely.
    #
    #        Must be a positive integer from the set [0,1,2,3].
    #
    #        DEFAULT: 2 (Two Romberg terms)
    #
    #        Note: 0 disables the Romberg step completely.
    #
    #  FixedStep - Allows the specification of a fixed step
    #        size, preventing the adaptive logic from working.
    #        This will be considerably faster, but not necessarily
    #        as accurate as allowing the adaptive logic to run.
    #
    #        DEFAULT: []
    #
    #        Note: If specified, FixedStep will define the
    #        maximum excursion from x0 that will be used.
    #
    #  Vectorized - Derivest will normally assume that your
    #        function can be safely evaluated at multiple locations
    #        in a single call. This would minimize the overhead of
    #        a loop and additional function call overhead. Some
    #        functions are not easily vectorizable, but you may
    #        (if your matlab release is new enough) be able to use
    #        arrayfun to accomplish the vectorization.
    #
    #        When all else: fails, set the vectorized property
    #        to no. This will cause derivest to loop over the
    #        successive function calls.
    #
    #        DEFAULT: yes
    #
    #
    #  MaxStep - Specifies the maximum excursion from x0 that
    #        will be allowed, as a multiple of x0.
    #
    #        DEFAULT: 100
    #
    #  StepRatio - Derivest uses a proportionally cascaded
    #        series of function evaluations, moving away from your
    #        point of evaluation. The StepRatio is the ratio used
    #        between sequential steps.
    #
    #        DEFAULT: 2.0000001
    #
    #        Note: use of a non-integer stepratio is intentional,
    #        to avoid integer multiples of the period of a periodic
    #        function under some circumstances.
    #
    #
    # Arguments: (output)
    #  der - derivative estimate for each element of x0
    #        der will have the same shape as x0.
    #
    #  errest - 95# uncertainty estimate of the derivative, such that
    #
    #        abs((der[j]) - f.T(x0[j])) < erest[j]
    #
    #  finaldelta - The final overall stepsize chosen by DERIVEST
    #

    par = namedtuple('par',
                     'DerivativeOrder MethodOrder Style RombergTerms FixedStep MaxStep StepRatio NominalStep Vectorized')

    par.DerivativeOrder = deriv
    par.MethodOrder = methodorder
    par.Style = style
    par.RombergTerms = rombergterms
    par.FixedStep = None
    par.MaxStep = maxstep
    # setting a default stepratio as a non-integer prevents
    # integer multiples of the initial point from being used.
    # In turn that avoids some problems for periodic functions.
    par.StepRatio = stepratio
    par.NominalStep = None
    par.Vectorized = vectorized

    par = check_params(par)

    # Was fun a string, or an inline/anonymous function?
    if fun is None:
        raise ValueError('fun was not supplied.')
    elif isinstance(fun, str):
        # a character function name
        fun = eval(fun)

    # no default for x0
    if x0 is None:
        raise ValueError('x0 was not supplied')

    par.NominalStep = max(x0, np.float128(0.02))

    # was a single point supplied?
    nx0 = x0.shape
    n = prod(nx0)

    # Set the steps to use.
    if par.FixedStep is None:
        # Basic sequence of steps, relative to a stepsize of 1.
        delta = par.MaxStep * par.StepRatio ** arange(0, -25 + -1, -1).T
        ndel = len(delta)
    else:
        # Fixed, user supplied absolute sequence of steps.
        ndel = 3 + ceil(par.DerivativeOrder / 2) + par.MethodOrder + par.RombergTerms
        if par.Style[0] == 'c':
            ndel = ndel - 2

        delta = par.FixedStep * par.StepRatio ** (-arange(ndel)).T

    # generate finite differencing rule in advance.
    # The rule is for a nominal unit step size, and will
    # be scaled later to reflect the local step size.
    fdarule = 1
    if par.Style == 'central':
        # for central rules, we will reduce the load by an
        # even or odd transformation as appropriate.
        if par.MethodOrder == 2:
            if par.DerivativeOrder == 1:
                # the odd transformation did all the work
                fdarule = 1
            elif par.DerivativeOrder == 2:
                # the even transformation did all the work
                fdarule = 2
            elif par.DerivativeOrder == 3:
                # the odd transformation did most of the work, but
                # we need to kill off the linear term
                fdarule = array([0, 1]).dot(pinv(fdamat(par.StepRatio, 1, 2)))
            elif par.DerivativeOrder == 4:
                # the even transformation did most of the work, but
                # we need to kill off the quadratic term
                fdarule = array([0, 1]).dot(pinv(fdamat(par.StepRatio, 2, 2)))

        else:
            # a 4th order method. We've already ruled out the 1st
            # order methods since these are central rules.
                if par.DerivativeOrder == 1:
                    # the odd transformation did most of the work, but
                    # we need to kill off the cubic term
                    fdarule = array([[1, 0]]).dot(pinv(fdamat(par.StepRatio, 1, 2)))
                elif par.DerivativeOrder == 2:
                    # the even transformation did most of the work, but
                    # we need to kill off the quartic term
                    fdarule = array([[1, 0]]).dot(pinv(fdamat(par.StepRatio, 2, 2)))
                elif par.DerivativeOrder == 3:
                    # the odd transformation did much of the work, but
                    # we need to kill off the linear & quintic terms
                    fdarule = array([[0, 1, 0]]).dot(pinv(fdamat(par.StepRatio, 1, 3)))
                elif par.DerivativeOrder == 4:
                    # the even transformation did much of the work, but
                    # we need to kill off the quadratic and 6th order terms
                    fdarule = array([[0, 1, 0]]).dot(pinv(fdamat(par.StepRatio, 2, 3)))
    elif par.Style in ['forward','backward']:
        # These two cases are identical, except at the very ,
        # where a sign will be introduced.
        
        # No odd/even trans, but we already dropped
        # off the constant term
        if par.MethodOrder == 1:
            if par.DerivativeOrder == 1:
                # an easy one
                fdarule = 1
            else:
                # [1:]4
                v = zeros((1, par.DerivativeOrder))
                v[par.DerivativeOrder] = 1
                fdarule = v / fdamat(par.StepRatio, 0, par.DerivativeOrder)
        
        else:
            # par.MethodOrder methods drop off the lower order terms,
            # plus terms directly above DerivativeOrder
            v = zeros((1, par.DerivativeOrder + par.MethodOrder - 1))
            v[par.DerivativeOrder] = 1
            fdarule = v / fdamat(par.StepRatio, 0, par.DerivativeOrder + par.MethodOrder - 1)
        
        # correct sign for the backward rule
        if par.Style[0] == 'b':
            fdarule = -fdarule

    # switch on par.style (generating fdarule)
    nfda = max(fdarule.shape)

    # will we need fun((x0))?
    if (remainder(par.DerivativeOrder, 2) == 0) or par.Style!='central':
        if par.Vectorized:
            f_x0 = fun(x0)
        else:
            # not vectorized, so loop
            f_x0 = zeros((x0.shape))
            for j in range(x0.size):
                f_x0[j] = fun(x0[j])
    else:
        f_x0 = None

    # Loop over the elements of x0, reducing it to
    # a scalar problem. Sorry, vectorization is not
    # complete here, but this IS only a single loop.
    der = zeros((nx0))
    errest = der.copy()
    finaldelta = der.copy()
    for i in range(n):
        x0i = x0[i]
        h = par.NominalStep

        # a central, forward or backwards differencing rule?
        # f_del is the set of all the function evaluations we
        # will generate. For a central rule, it will have the
        # even or odd transformation built in.
        if par.Style[0] == 'c':
            # A central rule, so we will need to evaluate
            # symmetrically around x0i.
            if par.Vectorized:
                f_plusdel = fun(x0i+h*delta)
                f_minusdel = fun(x0i-h*delta)
            else:
            # not vectorized, so loop
                f_minusdel = zeros((delta.shape),dtype=np.float128)
                f_plusdel = zeros((delta.shape),dtype=np.float128)
                for j in range(delta.size):
                    f_plusdel[j] = fun(x0i+h*delta[j])
                    f_minusdel[j] = fun(x0i-h*delta[j])
        
            if par.DerivativeOrder in [1, 3]:
                # odd transformation
                f_del = (f_plusdel - f_minusdel) / 2
            else:
                f_del = (f_plusdel + f_minusdel) / 2 - f_x0[i]
    
        elif par.Style[0] == 'f':
            # forward rule
            # drop off the constant only
            if par.Vectorized:
                f_del = fun(x0i + h*delta) - f_x0[i]
            else:
                # not vectorized, so loop
                f_del = zeros((delta.shape))
                for j in range(delta.size):
                    f_del[j] = fun(x0i+h*delta[j]) - f_x0[i]
        else:
        # backward rule
        # drop off the constant only
            if par.Vectorized:
                f_del = fun(x0i - h*delta) - f_x0[i]
            else:
                # not vectorized, so loop
                f_del = zeros((delta.shape))
                for j in range(delta.size):
                    f_del[j] = fun(x0i-h*delta[j]) - f_x0[i]
    
        # check the size of f_del to ensure it was properly vectorized.
        f_del = f_del.flatten()
        if len(f_del) != ndel:
            raise ValueError('fun did not return the correct size result(fun must be vectorized).')
    
    # Apply the finite difference rule at each delta, scaling
    # as appropriate for delta and the requested DerivativeOrder.
    # First, decide how many of these estimates we will  up with.
    ne = ndel + 1 - nfda - par.RombergTerms
    
    # Form the initial derivative estimates from the chosen
    # finite difference method.
    der_init = vec2mat(f_del, ne, nfda)@fdarule.T
    
    # scale to reflect the local delta
    der_init = der_init.flatten('F')/(h*delta[:ne])**par.DerivativeOrder
    
    # Each approximation that results is an approximation
    # of order par.DerivativeOrder to the desired derivative.
    # Additional (higher order, even or odd) terms in the
    # Taylor series also remain. Use a generalized (multi-term)
    # Romberg extrapolation to improve these estimates.
    if par.Style == 'central':
        rombexpon = 2 * arange(1,par.RombergTerms+1) + par.MethodOrder - 2
    else:
        rombexpon = arange(1,par.RombergTerms+1) + par.MethodOrder - 1
    
    der_romb, errors = rombextrap(par.StepRatio, der_init, rombexpon)
    
    # Choose which result to return
    
    # first, trim off the 
    if par.FixedStep is None:
        # trim off the estimates at each  of the scale
        nest = len(der_romb)
        if par.DerivativeOrder in [1,2]:
            trim = r_[1, 2, nest - 1, nest]
        elif par.DerivativeOrder==3:
            trim = r_[arange(1,5),nest + arange(-3,1)]
        elif par.DerivativeOrder == 4:
            trim = r_[arange(1,7), nest + arange(-5,1)]

        der_romb, tags = sort(der_romb), np.argsort(der_romb)

        np.delete(der_romb,trim)
        np.delete(tags,trim)
        errors = errors[tags]
        trimdelta = delta[tags]

        errest[i], ind = npmin(errors), np.argmin(errors)

        finaldelta[i] = h*trimdelta[ind]
        der[i] = der_romb[ind]
    else:
        [errest[i], ind] = npmin(errors), np.argmin(errors)
        finaldelta[i] = h*delta[ind]
        der[i] = der_romb(ind)

    return der, errest, finaldelta


""" Defines the GateCalc calculator class"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import numpy.linalg as _nla
import collections as _collections
import itertools as _itertools

from ..tools import compattools as _compat
from ..tools import slicetools as _slct
from ..tools import basistools as _bt
from ..tools import matrixtools as _mt
from ..baseobjs import DummyProfiler as _DummyProfiler
from . import spamvec as _sv
from . import gate as _gate
from . import labeldicts as _ld

_dummy_profiler = _DummyProfiler()

# Tolerace for matrix_rank when finding rank of a *normalized* projection
# matrix.  This is a legitimate tolerace since a normalized projection matrix
# should have just 0 and 1 eigenvalues, and thus a tolerace << 1.0 will work
# well.
P_RANK_TOL = 1e-7


class GateCalc(object):
    """
    Encapsulates a calculation tool used by gate set objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from GateSet to allow for additional
    gate set classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of gate matrices and SPAM vectors) access to these
    fundamental operations.
    """

    def __init__(self, dim, gates, preps, effects, paramvec):
        """
        Construct a new GateCalc object.

        Parameters
        ----------
        dim : int
            The gate-dimension.  All gate matrices should be dim x dim, and all
            SPAM vectors should be dim x 1.

        gates, preps, effects : OrderedDict
            Ordered dictionaries of Gate, SPAMVec, and SPAMVec objects,
            respectively.  Must be *ordered* dictionaries to specify a
            well-defined column ordering when taking derivatives.

        spamdefs : OrderedDict
            A dictionary whose keys are the allowed SPAM labels, and whose
            values are 2-tuples comprised of a state preparation label
            followed by a POVM effect label (both of which are strings,
            and keys of preps and effects, respectively, except for the
            special case when both are set to "remainder").

        paramvec : ndarray
            The parameter vector of the GateSet.
        """
        self.dim = dim
        self.gates = gates
        self.preps = preps
        self.effects = effects
        self.paramvec = paramvec
        self.Np = len(paramvec)


    def to_vector(self):
        """
        Returns the elements of the parent GateSet vectorized as a 1D array.
        Used for computing finite-difference derivatives.

        Returns
        -------
        numpy array
            The vectorized gateset parameters.
        """
        return self.paramvec


    def from_vector(self, v):
        """
        The inverse of to_vector.  Initializes the GateSet-like members of this
        calculator based on `v`. Used for computing finite-difference derivatives.
        """
        #Note: this *will* initialize the parent GateSet's objects too,
        # since only references to preps, effects, and gates are held
        # by the calculator class.  ORDER is important, as elements of
        # POVMs and Instruments rely on a fixed from_vector ordering
        # of their compiled effects/gates.
        self.paramvec = v.copy()
        for _,obj in self.iter_objs():
            obj.from_vector( v[obj.gpindices] )

            
    def iter_objs(self):
        for lbl,obj in _itertools.chain(self.preps.items(),
                                        self.effects.items(),
                                        self.gates.items()):
            yield (lbl,obj)


    def deriv_wrt_params(self):
        """
        Construct a matrix whose columns are the vectorized derivatives of all
        the gateset's raw matrix and vector *elements* (placed in a vector)
        with respect to each single gateset parameter.
        
        Returns
        -------
        numpy array
            2D array of derivatives.
        """
        num_els = sum([obj.size for _,obj in self.iter_objs()])
        num_params = self.Np
        deriv = _np.zeros( (num_els,num_params), 'd' )

        eo = 0 # element offset
        for lbl,obj in self.iter_objs():
            #Note: no overlaps possible b/c of independent *elements*
            deriv[eo:eo+obj.size,obj.gpindices] = obj.deriv_wrt_params()
            eo += obj.size
              
        return deriv


    def _buildup_dPG(self):
        """ 
        Helper function for building gauge/non-gauge projectors and 
          for computing the number of gauge/non-gauge elements.
        Returns the [ dP | dG ] matrix, i.e. np.concatenate( (dP,dG), axis=1 )
        whose nullspace gives the gauge directions in parameter space.
        """

        # ** See comments at the beginning of get_nongauge_projector for explanation **

        if any([not isinstance(gate,_gate.GateMatrix) for gate in self.gates.values()]) or \
           any([not isinstance(vec,_sv.DenseSPAMVec) for vec in self.preps.values()]) or \
           any([not isinstance(vec,_sv.DenseSPAMVec) for vec in self.effects.values()]):
            raise NotImplementedError(("Cannot (yet) extract gauge/non-gauge "
                                       "parameters for GateSets with sparse "
                                       "member representations"))

        bSkipEcs = True #Whether we should artificially skip complement-type
         # effect vecs, which is historically what we've done, even though
         # this seems somewhat wrong.  Not skipping them will alter the
         # number of "gauge params" since a complement Evec has a *fixed*
         # identity from the perspective of the GateSet params (which 
         # *varied* in gauge optimization, even though it's not a SPAMVec
         # param, creating a weird inconsistency...) SKIP
        if bSkipEcs:
            newSelf = self.copy()
            for effectlbl,EVec in self.effects.items():
                if isinstance(EVec, _sv.ComplementSPAMVec):
                    del newSelf.effects[effectlbl]
            self = newSelf #HACK!!! replacing self for remainder of this fn with version without Ecs
        
        #Use a GateSet object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        dim = self.dim 
        nParams = self.Np 
        nElements = sum([obj.size for _,obj in self.iter_objs()])

        #This was considered as optional behavior, but better to just delete qtys from GateSet
        ##whether elements of the raw gateset matrices/SPAM vectors that are not
        ## parameterized at all should be ignored.   By ignoring changes to such
        ## elements, they are treated as not being a part of the "gateset"
        #bIgnoreUnparameterizedEls = True

        #Note: gateset object (gsDeriv) must have all elements of gate
        # mxs and spam vectors as parameters (i.e. be "fully parameterized") in
        # order to match deriv_wrt_params call, which gives derivatives wrt
        # *all* elements of a gate set.

        gsDeriv_gates = _collections.OrderedDict(
            [(label,_np.zeros((dim,dim),'d')) for label in self.gates])
        gsDeriv_preps = _collections.OrderedDict(
            [(label,_np.zeros((dim,1),'d')) for label in self.preps])
        gsDeriv_effects = _collections.OrderedDict(
            [(label,_np.zeros((dim,1),'d')) for label in self.effects])
        
        dPG = _np.empty( (nElements, nParams + dim**2), 'd' )
        for i in range(dim):      # always range over all rows: this is the
            for j in range(dim):  # *generator* mx, not gauge mx itself
                unitMx = _bt.mut(i,j,dim)
                for lbl,rhoVec in self.preps.items():
                    gsDeriv_preps[lbl] = _np.dot(unitMx, rhoVec.toarray())
                for lbl,EVec in self.effects.items():
                    gsDeriv_effects[lbl] = -_np.dot(EVec.toarray().T, unitMx).T

                for lbl,gate in self.gates.items():
                    #if isinstance(gate,_gate.GateMatrix):
                    gsDeriv_gates[lbl] = _np.dot(unitMx,gate) - \
                                         _np.dot(gate,unitMx)
                    #else:
                    #    #use acton... maybe throw error if dim is too large (maybe above?)
                    #    deriv = _np.zeros((dim,dim),'d')
                    #    uv = _np.zeros((dim,1),'d') # unit vec
                    #    for k in range(dim): #FUTURE: could optimize this by bookeeping and pulling this loop outward
                    #        uv[k] = 1.0; Guv = gate.acton(uv); uv[k] = 0.0 #get k-th col of gate matrix
                    #        # termA_mn = sum( U_mk*Gkn ) so U locks m=i,k=j => termA_in = 1.0*Gjn
                    #        # termB_mn = sum( Gmk*U_kn ) so U locks k=i,n=j => termB_mj = 1.0*Gmi
                    #        deriv[i,k] += Guv[j,0] # termA contrib
                    #        if k == i: # i-th col of gate matrix goes in deriv's j-th col
                    #            deriv[:,j] -= Guv[:,0] # termB contrib
                    #    gsDeriv_gates[lbl] = deriv

                #Note: vectorize all the parameters in this full-
                # parameterization object, which gives a vector of length
                # equal to the number of gateset *elements*.
                to_vector = _np.concatenate(
                    [obj.flatten() for obj in _itertools.chain(
                        gsDeriv_preps.values(), gsDeriv_effects.values(),
                        gsDeriv_gates.values())], axis=0 )
                dPG[:,nParams + i*dim+j] = to_vector

        dPG[:, 0:nParams] = self.deriv_wrt_params()
        return dPG

    def get_nongauge_projector(self, itemWeights=None, nonGaugeMixMx=None):
        """
        Construct a projector onto the non-gauge parameter space, useful for
        isolating the gauge degrees of freedom from the non-gauge degrees of
        freedom.

        Parameters
        ----------
        itemWeights : dict, optional
            Dictionary of weighting factors for individual gates and spam operators.
            Keys can be gate, state preparation, POVM effect, spam labels, or the 
            special strings "gates" or "spam" whic represent the entire set of gate
            or SPAM operators, respectively.  Values are floating point numbers.
            These weights define the metric used to compute the non-gauge space,
            *orthogonal* the gauge space, that is projected onto.

        nonGaugeMixMx : numpy array, optional
            An array of shape (nNonGaugeParams,nGaugeParams) specifying how to
            mix the non-gauge degrees of freedom into the gauge degrees of
            freedom that are projected out by the returned object.  This argument
            essentially sets the off-diagonal block of the metric used for 
            orthogonality in the "gauge + non-gauge" space.  It is for advanced
            usage and typically left as None (the default).
.

        Returns
        -------
        numpy array
           The projection operator as a N x N matrix, where N is the number
           of parameters (obtained via num_params()).  This projector acts on
           parameter-space, and has rank equal to the number of non-gauge
           degrees of freedom.
        """

        # We want to divide the GateSet-space H (a Hilbert space, 56-dimensional in the 1Q, 3-gate, 2-vec case)
        # into the direct sum of gauge and non-gauge spaces, and find projectors onto each
        # sub-space (the non-gauge space in particular).
        #
        # Within the GateSet H-space lies a gauge-manifold of maximum chi2 (16-dimensional in 1Q case),
        #  where the gauge-invariant chi2 function is constant.  At a single point (GateSet) P on this manifold,
        #  chosen by (somewhat arbitrarily) fixing the gauge, we can talk about the tangent space
        #  at that point.  This tangent space is spanned by some basis (~16 elements for the 1Q case),
        #  which associate with the infinitesimal gauge transform ?generators? on the full space.
        #  The subspace of H spanned by the derivatives w.r.t gauge transformations at P (a GateSet) spans
        #  the gauge space, and the complement of this (in H) is the non-gauge space.
        #
        #  An element of the gauge group can be written gg = exp(-K), where K is a n x n matrix.  If K is
        #   hermitian then gg is unitary, but this need not be the case.  A gauge transform acts on a
        #   gatset via Gateset => gg^-1 G gg, i.e. G => exp(K) G exp(-K).  We care about the action of
        #   infinitesimal gauge tranformations (b/c the *derivative* vectors span the tangent space),
        #   which act as:
        #    G => (I+K) G (I-K) = G + [K,G] + ignored(K^2), where [K,G] := KG-GK
        #
        # To produce a projector onto the gauge-space, we compute the *column* vectors
        #  dG_ij = [K_ij,G], where K_ij is the i,j-th matrix unit (56x1 in the 1Q case, 16 such column vectors)
        #  and then form a projector in the standard way.
        #  (to project onto |v1>, |v2>, etc., form P = sum_i |v_i><v_i|)
        #
        #Typically nGateParams < len(dG_ij) and linear system is overconstrained
        #   and no solution is expected.  If no solution exists, simply ignore
        #
        # So we form P = sum_ij dG_ij * transpose(dG_ij) (56x56 in 1Q case)
        #              = dG * transpose(dG)              where dG_ij form the *columns* of dG (56x16 in 1Q case)
        # But dG_ij are not orthonormal, so really we need a slight modification,
        #  otherwise P'P' != P' as must be the case for a projector:
        #
        # P' = dG * (transpose(dG) * dG)^-1 * transpose(dG) (see wikipedia on projectors)
        #
        #    or equivalently (I think)
        #
        # P' = pseudo-inv(P)*P
        #
        #  since the pseudo-inv is defined by P*pseudo-inv(P) = I and so P'P' == P'
        #  and P' is our gauge-projector!

        # Note: In the general case of parameterized gates (i.e. non-fully parameterized gates), there are fewer
        #   gate parameters than the size of dG_ij ( < 56 in 1Q case).  In this general case, we want to know
        #   what (if any) change in gate parameters produces the change dG_ij of the gate matrix elements.
        #   That is, we solve dG_ij = derivWRTParams * dParams_ij for dParams_ij, where derivWRTParams is
        #   the derivative of the gate matrix elements with respect to the gate parameters (derivWRTParams
        #   is 56x(nGateParams) and dParams_ij is (nGateParams)x1 in the 1Q case) -- then substitute dG_ij
        #   with dParams_ij above to arrive at a (nGateParams)x(nGateParams) projector (compatible with
        #   hessian computed by gateset).
        #
        #   Actually, we want to know if what changes gate parameters
        #   produce changes in the span of all the dG_ij, so really we want the intersection of the space
        #   defined by the columns of derivWRTParams (the "gate-parameter range" space) and the dG_ij vectors.
        #
        #   This intersection is determined by nullspace( derivWRTParams | dG ), where the pipe denotes
        #     concatenating the matrices together.  Then, if x is a basis vector of the nullspace
        #     x[0:nGateParams] is the basis vector for the intersection space within the gate parameter space,
        #     that is, the analogue of dParams_ij in the single-dG_ij introduction above.
        #
        #   Still, we just substitue these dParams_ij vectors (as many as the nullspace dimension) for dG_ij
        #   above to get the general case projector.

        nParams = self.Np
        dPG = self._buildup_dPG()

        #print("DB: shapes = ",dP.shape,dG.shape)
        nullsp = _mt.nullspace_qr(dPG) #columns are nullspace basis vectors
        gen_dG = nullsp[0:nParams,:] #take upper (gate-param-segment) of vectors for basis
                                     # of subspace intersection in gate-parameter space
        #Note: gen_dG == "generalized dG", and is (nParams)x(nullSpaceDim==gaugeSpaceDim), so P
        #  (below) is (nParams)x(nParams) as desired.

        #DEBUG
        #nElements = self.num_elements()
        #for iRow in range(nElements):
        #    pNorm = _np.linalg.norm(dP[iRow])
        #    if pNorm < 1e-6:
        #        print "Row %d of dP is zero" % iRow
        #
        #print "------------------------------"
        #print "nParams = ",nParams
        #print "nElements = ",nElements
        #print " shape(M) = ",M.shape
        #print " shape(dG) = ",dG.shape
        #print " shape(dP) = ",dP.shape
        #print " shape(gen_dG) = ",gen_dG.shape
        #print " shape(nullsp) = ",nullsp.shape
        #print " rank(dG) = ",_np.linalg.matrix_rank(dG)
        #print " rank(dP) = ",_np.linalg.matrix_rank(dP)
        #print "------------------------------"
        #assert(_np.linalg.norm(_np.imag(gen_dG)) < 1e-9) #gen_dG is real

        if nonGaugeMixMx is not None:
            msg = "You've set both nonGaugeMixMx and itemWeights, both of which"\
                + " set the gauge metric... You probably don't want to do this."
            assert(itemWeights is None), msg

            # BEGIN GAUGE MIX ----------------------------------------
            # nullspace of gen_dG^T (mx with gauge direction vecs as rows) gives non-gauge directions
            #OLD: nonGaugeDirections = _mt.nullspace(gen_dG.T) #columns are non-gauge directions *orthogonal* to the gauge directions
            nonGaugeDirections = _mt.nullspace_qr(gen_dG.T) #columns are the non-gauge directions *orthogonal* to the gauge directions

            #for each column of gen_dG, which is a gauge direction in gateset parameter space,
            # we add some amount of non-gauge direction, given by coefficients of the
            # numNonGaugeParams non-gauge directions.
            gen_dG = gen_dG + _np.dot( nonGaugeDirections, nonGaugeMixMx) #add non-gauge direction components in
             # dims: (nParams,nGaugeParams) + (nParams,nNonGaugeParams) * (nNonGaugeParams,nGaugeParams)
             # nonGaugeMixMx is a (nNonGaugeParams,nGaugeParams) matrix whose i-th column specifies the
             #  coefficents to multipy each of the non-gauge directions by before adding them to the i-th
             #  direction to project out (i.e. what were the pure gauge directions).

            #DEBUG
            #print "gen_dG shape = ",gen_dG.shape
            #print "NGD shape = ",nonGaugeDirections.shape
            #print "NGD rank = ",_np.linalg.matrix_rank(nonGaugeDirections, P_RANK_TOL)
            #END GAUGE MIX ----------------------------------------

        # Build final non-gauge projector by getting a mx of column vectors 
        # orthogonal to the cols fo gen_dG:
        #     gen_dG^T * gen_ndG = 0 => nullspace(gen_dG^T)
        # or for a general metric W:
        #     gen_dG^T * W * gen_ndG = 0 => nullspace(gen_dG^T * W)
        # (This is instead of construction as I - gaugeSpaceProjector)

        if itemWeights is not None:
            metric_diag = _np.ones(self.Np, 'd')
            gateWeight = itemWeights.get('gates', 1.0)
            spamWeight = itemWeights.get('spam', 1.0)
            for lbl,gate in self.gates.items():
                metric_diag[gate.gpindices] = itemWeights.get(lbl, gateWeight)
            for lbl,vec in _itertools.chain(iter(self.preps.items()),
                                            iter(self.effects.items())):
                metric_diag[vec.gpindices] = itemWeights.get(lbl, spamWeight)
            metric = _np.diag(metric_diag)
            #OLD: gen_ndG = _mt.nullspace(_np.dot(gen_dG.T,metric))
            gen_ndG = _mt.nullspace_qr(_np.dot(gen_dG.T,metric))
        else:
            #OLD: gen_ndG = _mt.nullspace(gen_dG.T) #cols are non-gauge directions
            gen_ndG = _mt.nullspace_qr(gen_dG.T) #cols are non-gauge directions
            #print("DB: nullspace of gen_dG (shape = %s, rank=%d) = %s" % (str(gen_dG.shape),_np.linalg.matrix_rank(gen_dG),str(gen_ndG.shape)))
                

        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_ndG, _np.transpose(gen_ndG)) # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot( _np.linalg.pinv(P, rcond=1e-7), P ) # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_ndG), gen_ndG))
        #Pp_alt = _np.dot(gen_ndG, _np.dot(invGG, _np.transpose(gen_ndG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        #OLD: ret = _np.identity(nParams,'d') - Pp 
        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
        #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
        #       for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])
        
        try:
            rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL) # original un-normalized projector
            
            # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
            #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
            #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
            #  but different from numpy's default tolerance would be appropriate here.

            assert( rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL)) #rank shouldn't change with normalization
            #assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        except(_np.linalg.LinAlgError):
            _warnings.warn("Linear algebra error (probably a non-convergent" +
                           "SVD) ignored during matric rank checks in " +
                           "GateSet.get_nongauge_projector(...) ")
            
        return Pp 

        
    def product(self, gatestring, bScale=False):
        """
        Compute the product of a specified sequence of gate labels.

        Note: Gate matrices are multiplied in the reversed order of the tuple. That is,
        the first element of gatestring can be thought of as the first gate operation
        performed, which is on the far right of the product of matrices.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
            The sequence of gate labels.

        bScale : bool, optional
            When True, return a scaling factor (see below).

        Returns
        -------
        product : numpy array
            The product or scaled product of the gate matrices.

        scale : float
            Only returned when bScale == True, in which case the
            actual product == product * scale.  The purpose of this
            is to allow a trace or other linear operation to be done
            prior to the scaling.
        """
        raise NotImplementedError("product(...) is not implemented!")


    def dproduct(self, gatestring, flat=False, wrtFilter=None):
        """
        Compute the derivative of a specified sequence of gate labels.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        wrtFilter : list of ints, optional
          If not None, a list of integers specifying which gate parameters
          to include in the derivative.  Each element is an index into an
          array of gate parameters ordered by concatenating each gate's
          parameters (in the order specified by the gate set).  This argument
          is used internally for distributing derivative calculations across
          multiple processors.

        Returns
        -------
        deriv : numpy array
            * if flat == False, a M x G x G array, where:

              - M == length of the vectorized gateset (number of gateset parameters)
              - G == the linear dimension of a gate matrix (G x G gate matrices).

              and deriv[i,j,k] holds the derivative of the (j,k)-th entry of the product
              with respect to the i-th gateset parameter.

            * if flat == True, a N x M array, where:

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten)
              - M == length of the vectorized gateset (number of gateset parameters)

              and deriv[i,j] holds the derivative of the i-th entry of the flattened
              product with respect to the j-th gateset parameter.
        """
        raise NotImplementedError("dproduct(...) is not implemented!")


    def hproduct(self, gatestring, flat=False, wrtFilter1=None, wrtFilter2=None):
        """
        Compute the hessian of a specified sequence of gate labels.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        wrtFilter1, wrtFilter2 : list of ints, optional
          If not None, a list of integers specifying which gate parameters
          to differentiate with respect to in the first (row) and second (col)
          derivative operations, respectively.  Each element is an index into an
          array of gate parameters ordered by concatenating each gate's
          parameters (in the order specified by the gate set).  This argument
          is used internally for distributing derivative calculations across
          multiple processors.

        Returns
        -------
        hessian : numpy array
            * if flat == False, a  M x M x G x G numpy array, where:

              - M == length of the vectorized gateset (number of gateset parameters)
              - G == the linear dimension of a gate matrix (G x G gate matrices).

              and hessian[i,j,k,l] holds the derivative of the (k,l)-th entry of the product
              with respect to the j-th then i-th gateset parameters.

            * if flat == True, a  N x M x M numpy array, where:

              - N == the number of entries in a single flattened gate (ordered as numpy.flatten)
              - M == length of the vectorized gateset (number of gateset parameters)

              and hessian[i,j,k] holds the derivative of the i-th entry of the flattened
              product with respect to the k-th then k-th gateset parameters.
        """
        raise NotImplementedError("hproduct(...) is not implemented!")


#    def pr(self, spamLabel, gatestring, clipTo=None, bUseScaling=True):
#        """
#        Compute the probability of the given gate sequence, where initialization
#        & measurement operations are together specified by spamLabel.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        gatestring : GateString or tuple of gate labels
#          The sequence of gate labels specifying the gate string.
#
#        clipTo : 2-tuple, optional
#          (min,max) to clip return value if not None.
#
#        bUseScaling : bool, optional
#          Whether to use a post-scaled product internally.  If False, this
#          routine will run slightly faster, but with a chance that the
#          product will overflow and the subsequent trace operation will
#          yield nan as the returned probability.
#
#        Returns
#        -------
#        float
#        """
#        if self._is_remainder_spamlabel(spamLabel):
#            #then compute 1.0 - (all other spam label probabilities)
#            otherSpamdefs = list(self.spamdefs.keys())[:]; del otherSpamdefs[ otherSpamdefs.index(spamLabel) ]
#            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamdefs]) )
#            return 1.0 - sum( [self._pr_nr(sl, gatestring, clipTo, bUseScaling) for sl in otherSpamdefs] )
#        else:
#            return self._pr_nr(spamLabel, gatestring, clipTo, bUseScaling)
#
#    def _pr_nr(self, spamLabel, gatestring, clipTo, bUseScaling):
#        """ non-remainder version of pr(...) overridden by derived clases """
#        raise NotImplementedError("_pr_nr must be implemented by the derived class") 
#
#
#    def dpr(self, spamLabel, gatestring, returnPr=False,clipTo=None):
#        """
#        Compute the derivative of a probability generated by a gate string and
#        spam label as a 1 x M numpy array, where M is the number of gateset
#        parameters.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        gatestring : GateString or tuple of gate labels
#          The sequence of gate labels specifying the gate string.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probability itself.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip returned probability to if not None.
#           Only relevant when returnPr == True.
#
#        Returns
#        -------
#        derivative : numpy array
#            a 1 x M numpy array of derivatives of the probability w.r.t.
#            each gateset parameter (M is the length of the vectorized gateset).
#
#        probability : float
#            only returned if returnPr == True.
#        """
#
#        if self._is_remainder_spamlabel(spamLabel):
#            #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
#            otherSpamdefs = list(self.spamdefs.keys())[:]; del otherSpamdefs[ otherSpamdefs.index(spamLabel) ]
#            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamdefs]) )
#            otherResults = [self._dpr_nr(sl, gatestring, returnPr, clipTo) for sl in otherSpamdefs]
#            if returnPr:
#                return -1.0 * sum([dpr for dpr,p in otherResults]), 1.0 - sum([p for dpr,p in otherResults])
#            else:
#                return -1.0 * sum(otherResults)
#        else:
#            return self._dpr_nr(spamLabel, gatestring, returnPr, clipTo)
#
#        
#    def _dpr_nr(self, spamLabel, gatestring, returnPr, clipTo):
#        """ non-remainder version of dpr(...) overridden by derived clases """
#        raise NotImplementedError("_dpr_nr must be implemented by the derived class") 
#
#
#    def hpr(self, spamLabel, gatestring, returnPr=False, returnDeriv=False,clipTo=None):
#        """
#        Compute the Hessian of a probability generated by a gate string and
#        spam label as a 1 x M x M array, where M is the number of gateset
#        parameters.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        gatestring : GateString or tuple of gate labels
#          The sequence of gate labels specifying the gate string.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probability itself.
#
#        returnDeriv : bool, optional
#          when set to True, additionally return the derivative of the
#          probability.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip returned probability to if not None.
#           Only relevant when returnPr == True.
#
#        Returns
#        -------
#        hessian : numpy array
#            a 1 x M x M array, where M is the number of gateset parameters.
#            hessian[0,j,k] is the derivative of the probability w.r.t. the
#            k-th then the j-th gateset parameter.
#
#        derivative : numpy array
#            only returned if returnDeriv == True. A 1 x M numpy array of
#            derivatives of the probability w.r.t. each gateset parameter.
#
#        probability : float
#            only returned if returnPr == True.
#        """
#
#        if self._is_remainder_spamlabel(spamLabel):
#            #then compute Hessian[ 1.0 - (all other spam label probabilities) ]
#            otherSpamdefs = list(self.spamdefs.keys())[:]; del otherSpamdefs[ otherSpamdefs.index(spamLabel) ]
#            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamdefs]) )
#            otherResults = [self._hpr_nr(sl, gatestring, returnPr, returnDeriv, clipTo) for sl in otherSpamdefs]
#            if returnDeriv:
#                if returnPr: return ( -1.0 * sum([hpr for hpr,dpr,p in otherResults]),
#                                      -1.0 * sum([dpr for hpr,dpr,p in otherResults]),
#                                       1.0 - sum([p   for hpr,dpr,p in otherResults])   )
#                else:        return ( -1.0 * sum([hpr for hpr,dpr in otherResults]),
#                                      -1.0 * sum([dpr for hpr,dpr in otherResults])     )
#            else:
#                if returnPr: return ( -1.0 * sum([hpr for hpr,p in otherResults]),
#                                       1.0 - sum([p   for hpr,p in otherResults])   )
#                else:        return   -1.0 * sum(otherResults)
#        else:
#            return self._hpr_nr(spamLabel, gatestring, returnPr, returnDeriv, clipTo)
#
#            
#    def _hpr_nr(self, spamLabel, gatestring, returnPr, returnDeriv, clipTo):
#        """ non-remainder version of hpr(...) overridden by derived clases """
#        raise NotImplementedError("_hpr_nr must be implemented by the derived class") 


    def probs(self, compiled_gatestring, clipTo=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        clipTo : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,gatestring,clipTo)
            for each spam label (string) SL.
        """
        probs = _ld.OutcomeLabelDict()
        raw_dict, outcomeLbls = compiled_gatestring
        iOut = 0 #outcome index
        for raw_gatestring, spamTuples in raw_dict.items():
            for spamTuple in spamTuples:
                probs[outcomeLbls[iOut]] = self.pr(
                    spamTuple, raw_gatestring, clipTo, False)
                iOut += 1
        return probs


    def dprobs(self, compiled_gatestring, returnPr=False,clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            dprobs[SL] = dpr(SL,gatestring,gates,G0,SPAM,SP0,returnPr,clipTo)
            for each spam label (string) SL.
        """
        dprobs = { }
        raw_dict, outcomeLbls = compiled_gatestring
        iOut = 0 #outcome index
        for raw_gatestring, spamTuples in raw_dict.items():
            for spamTuple in spamTuples:
                dprobs[outcomeLbls[iOut]] = self.dpr(
                    spamTuple, raw_gatestring, returnPr, clipTo)
                iOut += 1
        return dprobs



    def hprobs(self, compiled_gatestring, returnPr=False, returnDeriv=False, clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given gate string.

        Parameters
        ----------
        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the derivatives of the
          probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        hprobs : dictionary
            A dictionary such that
            hprobs[SL] = hpr(SL,gatestring,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo)
            for each spam label (string) SL.
        """
        hprobs = { }
        raw_dict, outcomeLbls = compiled_gatestring
        iOut = 0 #outcome index
        for raw_gatestring, spamTuples in raw_dict.items():
            for spamTuple in spamTuples:
                hprobs[outcomeLbls[iOut]] = self.hpr(
                    spamTuple, raw_gatestring, returnPr,returnDeriv,clipTo)
                iOut += 1
        return hprobs

    def construct_evaltree(self):
        """
        Constructs an EvalTree object appropriate for this calculator.
        """
        raise NotImplementedError("construct_evaltree(...) is not implemented!")

    
    def bulk_product(self, evalTree, bScale=False, comm=None):
        """
        Compute the products of many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        bScale : bool, optional
           When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  This is done over gate strings when a
           *split* evalTree is given, otherwise no parallelization is performed.

        Returns
        -------
        prods : numpy array
            Array of shape S x G x G, where:

            - S == the number of gate strings
            - G == the linear dimension of a gate matrix (G x G gate matrices).

        scaleValues : numpy array
            Only returned when bScale == True. A length-S array specifying
            the scaling that needs to be applied to the resulting products
            (final_product[i] = scaleValues[i] * prods[i]).
        """
        raise NotImplementedError("bulk_product(...) is not implemented!")

    
    def bulk_dproduct(self, evalTree, flat=False, bReturnProds=False,
                      bScale=False, comm=None, wrtFilter=None):
        """
        Compute the derivative of a many gate strings at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnProds : bool, optional
          when set to True, additionally return the probabilities.

        bScale : bool, optional
          When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first done over the
           set of parameters being differentiated with respect to.  If there are
           more processors than gateset parameters, distribution over a split
           evalTree (if given) is possible.

        wrtFilter : list of ints, optional
          If not None, a list of integers specifying which gate parameters
          to include in the derivative.  Each element is an index into an
          array of gate parameters ordered by concatenating each gate's
          parameters (in the order specified by the gate set).  This argument
          is used internally for distributing derivative calculations across
          multiple processors.


        Returns
        -------
        derivs : numpy array

          * if flat == False, an array of shape S x M x G x G, where:

            - S == len(gatestring_list)
            - M == the length of the vectorized gateset
            - G == the linear dimension of a gate matrix (G x G gate matrices)

            and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
            of the i-th gate string product with respect to the j-th gateset
            parameter.

          * if flat == True, an array of shape S*N x M where:

            - N == the number of entries in a single flattened gate (ordering same as numpy.flatten),
            - S,M == as above,

            and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
            the (i / G^2)-th flattened gate string product  with respect to
            the j-th gateset parameter.

        products : numpy array
          Only returned when bReturnProds == True.  An array of shape
          S x G x G; products[i] is the i-th gate string product.

        scaleVals : numpy array
          Only returned when bScale == True.  An array of shape S such that
          scaleVals[i] contains the multiplicative scaling needed for
          the derivatives and/or products for the i-th gate string.
        """
        raise NotImplementedError("bulk_dproduct(...) is not implemented!")


    def bulk_hproduct(self, evalTree, flat=False, bReturnDProdsAndProds=False,
                      bScale=False, comm=None, wrtFilter1=None, wrtFilter2=None):

        """
        Return the Hessian of many gate string products at once.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        flat : bool, optional
          Affects the shape of the returned derivative array (see below).

        bReturnDProdsAndProds : bool, optional
          when set to True, additionally return the probabilities and
          their derivatives (see below).

        bScale : bool, optional
          When True, return a scaling factor (see below).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first done over the
           set of parameters being differentiated with respect to when the
           *second* derivative is taken.  If there are more processors than
           gateset parameters, distribution over a split evalTree (if given)
           is possible.

        wrtFilter1, wrtFilter2 : list of ints, optional
          If not None, a list of integers specifying which gate parameters
          to differentiate with respect to in the first (row) and second (col)
          derivative operations, respectively.  Each element is an index into an
          array of gate parameters ordered by concatenating each gate's
          parameters (in the order specified by the gate set).  This argument
          is used internally for distributing derivative calculations across
          multiple processors.

        Returns
        -------
        hessians : numpy array
            * if flat == False, an  array of shape S x M x M x G x G, where

              - S == len(gatestring_list)
              - M == the length of the vectorized gateset
              - G == the linear dimension of a gate matrix (G x G gate matrices)

              and hessians[i,j,k,l,m] holds the derivative of the (l,m)-th entry
              of the i-th gate string product with respect to the k-th then j-th
              gateset parameters.

            * if flat == True, an array of shape S*N x M x M where

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten),
              - S,M == as above,

              and hessians[i,j,k] holds the derivative of the (i % G^2)-th entry
              of the (i / G^2)-th flattened gate string product with respect to
              the k-th then j-th gateset parameters.

        derivs1, derivs2 : numpy array
          Only returned if bReturnDProdsAndProds == True.

          * if flat == False, two arrays of shape S x M x G x G, where

            - S == len(gatestring_list)
            - M == the number of gateset params or wrtFilter1 or 2, respectively
            - G == the linear dimension of a gate matrix (G x G gate matrices)

            and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
            of the i-th gate string product with respect to the j-th gateset
            parameter.

          * if flat == True, an array of shape S*N x M where

            - N == the number of entries in a single flattened gate (ordering is
                   the same as that used by numpy.flatten),
            - S,M == as above,

            and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
            the (i / G^2)-th flattened gate string product  with respect to
            the j-th gateset parameter.

        products : numpy array
          Only returned when bReturnDProdsAndProds == True.  An array of shape
          S x G x G; products[i] is the i-th gate string product.

        scaleVals : numpy array
          Only returned when bScale == True.  An array of shape S such that
          scaleVals[i] contains the multiplicative scaling needed for
          the hessians, derivatives, and/or products for the i-th gate string.

        """
        raise NotImplementedError("bulk_hproduct(...) is not implemented!")
    

    def _fill_result_tuple(self, result_tup, evalTree,
                           param_slice1, param_slice2, calc_and_fill_fn):
        """ 
        This function takes a "calc-and-fill" function, which computes
        and *fills* (i.e. doesn't return to save copying) some arrays. The
        arrays that are filled internally to `calc_and_fill_fn` must be the 
        same as the elements of `result_tup`.  The fill function computes
        values for only a single spam label (specified to it by the first
        two arguments), and in general only a specified slice of the values
        for this spam label (given by the subsequent arguments, except for
        the last).  The final argument is a boolean specifying whether 
        the filling should overwrite or add to the existing array values, 
        which is a functionality needed to correctly handle the remainder
        spam label.
        """
        
        pslc1 = param_slice1
        pslc2 = param_slice2
        for spamTuple, (fInds,gInds) in evalTree.spamtuple_indices.items():
            # fInds = "final indices" = the "element" indices in the final
            #          filled quantity combining both spam and gate-sequence indices
            # gInds  = "gate sequence indices" = indices into the (tree-) list of
            #          all of the raw gate sequences which need to be computed
            #          for the current spamTuple (this list has the SAME length as fInds).            
            calc_and_fill_fn(spamTuple,fInds,gInds,pslc1,pslc2,False) #TODO: remove SumInto == True cases
                    
        return

    def _fill_result_tuple_collectrho(self, result_tup, evalTree,
                           param_slice1, param_slice2, calc_and_fill_fn):
        """ 
        Similar to :method:`_fill_result_tuple`, but collects common-rho
        spamtuples together for speeding up map-based evaluation.  Thus, where
        `_fill_result_tuple` makes a separate call to `calc_and_fill_fn` for 
        each `(rhoLabel,Elabel)` spamtuple, this function calls 
        `calc_and_fill_fn(rhoLabel, Elabels, ...)`.
        """ 
        
        pslc1 = param_slice1
        pslc2 = param_slice2
        collected = _collections.OrderedDict() # keys are rho labels
        for spamTuple, (fInds,gInds) in evalTree.spamtuple_indices.items():
            # fInds = "final indices" = the "element" indices in the final
            #          filled quantity combining both spam and gate-sequence indices
            # gInds  = "gate sequence indices" = indices into the (tree-) list of
            #          all of the raw gate sequences which need to be computed
            #          for the current spamTuple (this list has the SAME length as fInds).
            rholabel,elabel = spamTuple #this should always be the case... (no "custom" / "raw" labels)
            if rholabel not in collected: collected[rholabel] = [list(),list(),list()]
            collected[rholabel][0].append(elabel)
            collected[rholabel][1].append(fInds)
            collected[rholabel][2].append(gInds)
            
        for rholabel, (elabels, fIndsList, gIndsList) in collected.items():
            calc_and_fill_fn(rholabel, elabels, fIndsList, gIndsList,pslc1,pslc2, False) 
                    
        return

    

    def bulk_fill_probs(self, mxToFill, evalTree,
                        clipTo=None, check=False, comm=None):

        """
        Compute the outcome probabilities for an entire tree of gate strings.

        This routine fills a 1D array, `mxToFill` with the probabilities
        corresponding to the *compiled* gate strings found in an evaluation
        tree, `evalTree`.  An initial list of (general) :class:`GateString`
        objects is *compiled* into a lists of gate-only sequences along with
        a mapping of final elements (i.e. probabilities) to gate-only sequence
        and prep/effect pairs.  The evaluation tree organizes how to efficiently
        compute the gate-only sequences.  This routine fills in `mxToFill`, which
        must have length equal to the number of final elements (this can be 
        obtained by `evalTree.num_final_elements()`.  To interpret which elements
        correspond to which strings and outcomes, you'll need the mappings 
        generated when the original list of `GateStrings` was compiled.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated 1D numpy array of length equal to the
          total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).


        Returns
        -------
        None
        """
        raise NotImplementedError("bulk_fill_probs(...) is not implemented!")


    def bulk_fill_dprobs(self, mxToFill, evalTree,
                         prMxToFill=None,clipTo=None,check=False,
                         comm=None, wrtFilter=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):

        """
        Compute the outcome probability-derivatives for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
        probability-derivatives for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M is the 
          number of gate set parameters.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter : list of ints, optional
          If not None, a list of integers specifying which parameters
          to include in the derivative dimension. This argument is used
          internally for distributing calculations across multiple
          processors and to control memory usage.  Cannot be specified
          in conjuction with wrtBlockSize.

        wrtBlockSize : int or float, optional
          The maximum number of derivative columns to compute *products*
          for simultaneously.  None means compute all requested columns
          at once.  The  minimum of wrtBlockSize and the size that makes
          maximal use of available processors is used as the final block size.
          This argument must be None if wrtFilter is not None.  Set this to
          non-None to reduce amount of intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """
        raise NotImplementedError("bulk_fill_dprobs(...) is not implemented!")


    def bulk_fill_hprobs(self, mxToFill, evalTree,
                         prMxToFill=None, deriv1MxToFill=None, deriv2MxToFill=None, 
                         clipTo=None, check=False,comm=None, wrtFilter1=None, wrtFilter2=None,
                         wrtBlockSize1=None, wrtBlockSize2=None, gatherMemLimit=None):
        """
        Compute the outcome probability-Hessians for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        probability-Hessians for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExMxM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M1 & M2 are
          the number of selected gate-set parameters (by wrtFilter1 and wrtFilter2).

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *compiled* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated ExM numpy array that is filled
          with probability derivatives, similar to bulk_fill_dprobs(...), but
          where M is the number of gateset parameters selected for the 1st and 2nd
          differentiation, respectively (i.e. by wrtFilter1 and wrtFilter2).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter1, wrtFilter2 : list of ints, optional
          If not None, a list of integers specifying which gate set parameters
          to differentiate with respect to in the first (row) and second (col)
          derivative operations, respectively.

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """
        raise NotImplementedError("bulk_fill_hprobs(...) is not implemented!")        


    #def bulk_pr(self, spamLabel, evalTree, clipTo=None,
    #            check=False, comm=None):
    #    """
    #    Compute the probabilities of the gate sequences given by evalTree,
    #    where initialization & measurement operations are always the same
    #    and are together specified by spamLabel.
    #
    #    Parameters
    #    ----------
    #    spamLabel : string
    #       the label specifying the state prep and measure operations
    #
    #    evalTree : EvalTree
    #       given by a prior call to bulk_evaltree.  Specifies the gate strings
    #       to compute the bulk operation on.
    #
    #    clipTo : 2-tuple, optional
    #       (min,max) to clip return value if not None.
    #
    #    check : boolean, optional
    #      If True, perform extra checks within code to verify correctness,
    #      generating warnings when checks fail.  Used for testing, and runs
    #      much slower when True.
    #
    #    comm : mpi4py.MPI.Comm, optional
    #       When not None, an MPI communicator for distributing the computation
    #       across multiple processors.  Distribution is performed over
    #       subtrees of evalTree (if it is split).
    #
    #
    #    Returns
    #    -------
    #    numpy array
    #      An array of length equal to the number of gate strings containing
    #      the (float) probabilities.
    #    """
    #    vp = _np.empty( (1,evalTree.num_final_strings()), 'd' )
    #    self.bulk_fill_probs(vp, { spamLabel: 0}, evalTree,
    #                         clipTo, check, comm)
    #    return vp[0]


    def bulk_probs(self, gatestrings, evalTree, elIndices, outcomes,
                   clipTo=None, check=False, comm=None):
        """
        Construct a dictionary containing the probabilities
        for an entire list of gate sequences.

        Parameters
        ----------
        gatestrings : list of GateStrings
            The list of (uncompiled) original gate strings.

        evalTree : EvalTree
            An evalution tree corresponding to `gatestrings`.

        elIndices : dict
            A dictionary of indices for each original gate string.

        outcomes : dict
            A dictionary of outcome labels (string or tuple) for each original
            gate string.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.


        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[gstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        vp = _np.empty(evalTree.num_final_elements(),'d')
        self.bulk_fill_probs(vp, evalTree, clipTo, check, comm)

        ret = _collections.OrderedDict()
        for i, gstr in enumerate(gatestrings):
            elInds = _slct.indices(elIndices[i]) \
                     if isinstance(elIndices[i],slice) else elIndices[i]
            ret[gstr] = _collections.OrderedDict(
                [(outLbl,vp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret


#    def bulk_dpr(self, spamLabel, evalTree,
#                 returnPr=False,clipTo=None,check=False,
#                 comm=None, wrtFilter=None, wrtBlockSize=None):
#        """
#        Compute the derivatives of the probabilities generated by a each gate
#        sequence given by evalTree, where initialization
#        & measurement operations are always the same and are
#        together specified by spamLabel.
#
#        Parameters
#        ----------
#        spamLabel : string
#           the label specifying the state prep and measure operations
#
#        evalTree : EvalTree
#           given by a prior call to bulk_evaltree.  Specifies the gate strings
#           to compute the bulk operation on.
#
#        returnPr : bool, optional
#          when set to True, additionally return the probabilities.
#
#        clipTo : 2-tuple, optional
#           (min,max) to clip returned probability to if not None.
#           Only relevant when returnPr == True.
#
#        check : boolean, optional
#          If True, perform extra checks within code to verify correctness,
#          generating warnings when checks fail.  Used for testing, and runs
#          much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#           When not None, an MPI communicator for distributing the computation
#           across multiple processors.  Distribution is first performed over
#           subtrees of evalTree (if it is split), and then over blocks (subsets)
#           of the parameters being differentiated with respect to (see
#           wrtBlockSize).
#
#        wrtFilter : list of ints, optional
#          If not None, a list of integers specifying which parameters
#          to include in the derivative dimension. This argument is used
#          internally for distributing calculations across multiple
#          processors and to control memory usage.  Cannot be specified
#          in conjuction with wrtBlockSize.
#
#        wrtBlockSize : int or float, optional
#          The maximum average number of derivative columns to compute *products*
#          for simultaneously.  None means compute all requested columns
#          at once.  The  minimum of wrtBlockSize and the size that makes
#          maximal use of available processors is used as the final block size.
#          This argument must be None if wrtFilter is not None.  Set this to
#          non-None to reduce amount of intermediate memory required.
#
#        Returns
#        -------
#        dprobs : numpy array
#            An array of shape S x M, where
#
#            - S == the number of gate strings
#            - M == the length of the vectorized gateset
#
#            and dprobs[i,j] holds the derivative of the i-th probability w.r.t.
#            the j-th gateset parameter.
#
#        probs : numpy array
#            Only returned when returnPr == True. An array of shape S containing
#            the probabilities of each gate string.
#        """
#        nGateStrings = evalTree.num_final_strings()
#        nDerivCols = self.Np
#
#        vdp = _np.empty( (1,nGateStrings,nDerivCols), 'd' )
#        vp = _np.empty( (1,nGateStrings), 'd' ) if returnPr else None
#
#        self.bulk_fill_dprobs(vdp, {spamLabel: 0}, evalTree,
#                              vp, clipTo, check, comm,
#                              wrtFilter, wrtBlockSize)
#        return (vdp[0], vp[0]) if returnPr else vdp[0]


    def bulk_dprobs(self, gatestrings, evalTree, elIndices, outcomes,
                    returnPr=False,clipTo=None,
                    check=False,comm=None,
                    wrtFilter=None, wrtBlockSize=None):

        """
        Construct a dictionary containing the probability-derivatives
        for an entire list of gate sequences.

        Parameters
        ----------
        gatestrings : list of GateStrings
            The list of (uncompiled) original gate strings.

        evalTree : EvalTree
            An evalution tree corresponding to `gatestrings`.

        elIndices : dict
            A dictionary of indices for each original gate string.

        outcomes : dict
            A dictionary of outcome labels (string or tuple) for each original
            gate string.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter : list of ints, optional
          If not None, a list of integers specifying which parameters
          to include in the derivative dimension. This argument is used
          internally for distributing calculations across multiple
          processors and to control memory usage.  Cannot be specified
          in conjuction with wrtBlockSize.

        wrtBlockSize : int or float, optional
          The maximum average number of derivative columns to compute *products*
          for simultaneously.  None means compute all requested columns
          at once.  The  minimum of wrtBlockSize and the size that makes
          maximal use of available processors is used as the final block size.
          This argument must be None if wrtFilter is not None.  Set this to
          non-None to reduce amount of intermediate memory required.


        Returns
        -------
        dprobs : dictionary
            A dictionary such that `probs[gstr]` is an ordered dictionary of
            `(outcome, dp, p)` tuples, where `outcome` is a tuple of labels,
            `p` is the corresponding probability, and `dp` is an array containing
            the derivative of `p` with respect to each parameter.  If `returnPr`
            if False, then `p` is not included in the tuples (so they're just
            `(outcome, dp)`).
        """
        nElements = evalTree.num_final_elements()
        nDerivCols = self.Np

        vdp = _np.empty( (nElements,nDerivCols), 'd' )
        vp = _np.empty( nElements, 'd' ) if returnPr else None

        self.bulk_fill_dprobs(vdp, evalTree,
                              vp, clipTo, check, comm,
                              wrtFilter, wrtBlockSize)

        ret = _collections.OrderedDict()
        for i, gstr in enumerate(gatestrings):
            elInds = _slct.indices(elIndices[i]) \
                     if isinstance(elIndices[i],slice) else elIndices[i]
            if returnPr:
                ret[gstr] = _collections.OrderedDict(
                    [(outLbl,(vdp[ei],vp[ei])) for ei, outLbl in zip(elInds, outcomes[i])])
            else:
                ret[gstr] = _collections.OrderedDict(
                    [(outLbl,vdp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret


#    def bulk_hpr(self, spamLabel, evalTree,
#                 returnPr=False,returnDeriv=False,
#                 clipTo=None,check=False,comm=None,
#                 wrtFilter1=None, wrtFilter2=None,
#                 wrtBlockSize1=None, wrtBlockSize2=None):
#    
#        """
#        Compute the 2nd derivatives of the probabilities generated by a each gate
#        sequence given by evalTree, where initialization & measurement
#        operations are always the same and are together specified by spamLabel.
#    
#        Parameters
#        ----------
#        spamLabel : string
#          the label specifying the state prep and measure operations
#    
#        evalTree : EvalTree
#          given by a prior call to bulk_evaltree.  Specifies the gate strings
#          to compute the bulk operation on.
#    
#        returnPr : bool, optional
#          when set to True, additionally return the probabilities.
#    
#        returnDeriv : bool, optional
#          when set to True, additionally return the probability derivatives.
#    
#        clipTo : 2-tuple, optional
#          (min,max) to clip returned probability to if not None.
#          Only relevant when returnPr == True.
#    
#        check : boolean, optional
#          If True, perform extra checks within code to verify correctness,
#          generating warnings when checks fail.  Used for testing, and runs
#          much slower when True.
#    
#        comm : mpi4py.MPI.Comm, optional
#           When not None, an MPI communicator for distributing the computation
#           across multiple processors.  Distribution is first performed over
#           subtrees of evalTree (if it is split), and then over blocks (subsets)
#           of the parameters being differentiated with respect to (see
#           wrtBlockSize).
#    
#        wrtFilter1, wrtFilter2 : list of ints, optional
#          If not None, a list of integers specifying which gate set parameters
#          to differentiate with respect to in the first (row) and second (col)
#          derivative operations, respectively.
#    
#        wrtBlockSize2, wrtBlockSize2 : int or float, optional
#          The maximum number of 1st (row) and 2nd (col) derivatives to compute
#          *products* for simultaneously.  None means compute all requested
#          rows or columns at once.  The  minimum of wrtBlockSize and the size
#          that makes maximal use of available processors is used as the final
#          block size.  These arguments must be None if the corresponding
#          wrtFilter is not None.  Set this to non-None to reduce amount of
#          intermediate memory required.
#    
#    
#        Returns
#        -------
#        hessians : numpy array
#            a S x M x M array, where
#    
#            - S == the number of gate strings
#            - M == the length of the vectorized gateset
#    
#            and hessians[i,j,k] is the derivative of the i-th probability
#            w.r.t. the k-th then the j-th gateset parameter.
#    
#        derivs1, derivs2 : numpy array
#            only returned if returnDeriv == True. Two S x M arrays where
#            derivsX[i,j] holds the derivative of the i-th probability
#            w.r.t. the j-th gateset parameter, where j is taken from the
#            first and second sets of filtered parameters (i.e. by
#            wrtFilter1 and wrtFilter2).  If `wrtFilter1 == wrtFilter2`,
#            then derivs2 is not returned (to save memory, since it's the
#            same as derivs1).
#    
#        probabilities : numpy array
#            only returned if returnPr == True.  A length-S array
#            containing the probabilities for each gate string.
#        """
#        nGateStrings = evalTree.num_final_strings()
#        nDerivCols1 = self.Np if (wrtFilter1 is None) \
#                           else len(wrtFilter1)
#        nDerivCols2 = self.Np if (wrtFilter2 is None) \
#                           else len(wrtFilter2)
#    
#        vhp = _np.empty( (1,nGateStrings,nDerivCols1,nDerivCols2), 'd' )
#        vdp1 = _np.empty( (1,nGateStrings,self.Np), 'd' ) \
#            if returnDeriv else None
#        vdp2 = vdp1.copy() if (returnDeriv and wrtFilter1!=wrtFilter2) else None
#        vp = _np.empty( (1,nGateStrings), 'd' ) if returnPr else None
#    
#        self.bulk_fill_hprobs(vhp, {spamLabel: 0}, evalTree,
#                              vp, vdp1, vdp2, clipTo, check, comm,
#                              wrtFilter1,wrtFilter2,wrtBlockSize1,wrtBlockSize2)
#        if returnDeriv:
#            if vdp2 is None:
#                return (vhp[0], vdp1[0], vp[0]) if returnPr else (vhp[0],vdp1[0])
#            else:
#                return (vhp[0], vdp1[0], vdp2[0], vp[0]) if returnPr else (vhp[0],vdp1[0],vdp2[0])
#        else:
#            return (vhp[0], vp[0]) if returnPr else vhp[0]


    def bulk_hprobs(self, gatestrings, evalTree, elIndices, outcomes,
                    returnPr=False,returnDeriv=False,clipTo=None,
                    check=False,comm=None,
                    wrtFilter1=None, wrtFilter2=None,
                    wrtBlockSize1=None, wrtBlockSize2=None):

        """
        Construct a dictionary containing the probability-Hessians
        for an entire list of gate sequences.

        Parameters
        ----------
        gatestrings : list of GateStrings
            The list of (uncompiled) original gate strings.

        evalTree : EvalTree
            An evalution tree corresponding to `gatestrings`.

        elIndices : dict
            A dictionary of indices for each original gate string.

        outcomes : dict
            A dictionary of outcome labels (string or tuple) for each original
            gate string.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the probability derivatives.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter1, wrtFilter2 : list of ints, optional
          If not None, a list of integers specifying which gate set parameters
          to differentiate with respect to in the first (row) and second (col)
          derivative operations, respectively.

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.


        Returns
        -------
        hprobs : dictionary
            A dictionary such that `probs[gstr]` is an ordered dictionary of
            `(outcome, hp, dp, p)` tuples, where `outcome` is a tuple of labels,
            `p` is the corresponding probability, `dp` is a 1D array containing
            the derivative of `p` with respect to each parameter, and `hp` is a
            2D array containing the Hessian of `p` with respect to each parameter.
            If `returnPr` if False, then `p` is not included in the tuples.
            If `returnDeriv` if False, then `dp` is not included in the tuples.
        """
        nElements = evalTree.num_final_elements()
        nDerivCols1 = self.Np if (wrtFilter1 is None) \
                           else len(wrtFilter1)
        nDerivCols2 = self.Np if (wrtFilter2 is None) \
                           else len(wrtFilter2)

        vhp = _np.empty( (nElements,nDerivCols1,nDerivCols2),'d')
        vdp1 = _np.empty( (nElements,self.Np), 'd' ) \
            if returnDeriv else None
        vdp2 = vdp1.copy() if (returnDeriv and wrtFilter1!=wrtFilter2) else None
        vp = _np.empty( nElements, 'd' ) if returnPr else None

        self.bulk_fill_hprobs(vhp, evalTree,
                              vp, vdp1, vdp2, clipTo, check, comm,
                              wrtFilter1,wrtFilter1,wrtBlockSize1,wrtBlockSize2)

        ret = _collections.OrderedDict()
        for i, gstr in enumerate(gatestrings):
            elInds = _slct.indices(elIndices[i]) \
                     if isinstance(elIndices[i],slice) else elIndices[i]
            outcomeQtys = _collections.OrderedDict()
            for ei, outLbl in zip(elInds, outcomes[i]):
                if returnDeriv:
                    if vdp2 is None:
                        if returnPr: t = (vhp[ei],vdp1[ei],vp[ei])
                        else:        t = (vhp[ei],vdp1[ei])
                    else:
                        if returnPr: t = (vhp[ei],vdp1[ei],vdp2[ei],vp[ei])
                        else:        t = (vhp[ei],vdp1[ei],vdp2[ei])
                else:
                    if returnPr: t = (vhp[ei],vp[ei])
                    else:        t = vhp[ei]
                outcomeQtys[outLbl] = t
            ret[gstr] = outcomeQtys
            
        return ret


    def bulk_hprobs_by_block(self, evalTree, wrtSlicesList,
                             bReturnDProbs12=False, comm=None):
                             
        """
        Constructs a generator that computes the 2nd derivatives of the
        probabilities generated by a each gate sequence given by evalTree
        column-by-column.

        This routine can be useful when memory constraints make constructing
        the entire Hessian at once impractical, and one is able to compute
        reduce results from a single column of the Hessian at a time.  For
        example, the Hessian of a function of many gate sequence probabilities
        can often be computed column-by-column from the using the columns of
        the gate sequences.


        Parameters
        ----------
        spam_label_rows : dictionary
            a dictionary with keys == spam labels and values which
            are integer row indices into mxToFill, specifying the
            correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the gate strings
            to compute the bulk operation on.  This tree *cannot* be split.

        wrtSlicesList : list
            A list of `(rowSlice,colSlice)` 2-tuples, each of which specify
            a "block" of the Hessian to compute.  Iterating over the output
            of this function iterates over these computed blocks, in the order
            given by `wrtSlicesList`.  `rowSlice` and `colSlice` must by Python
            `slice` objects.

        bReturnDProbs12 : boolean, optional
           If true, the generator computes a 2-tuple: (hessian_col, d12_col),
           where d12_col is a column of the matrix d12 defined by:
           d12[iSpamLabel,iGateStr,p1,p2] = dP/d(p1)*dP/d(p2) where P is is
           the probability generated by the sequence and spam label indexed
           by iGateStr and iSpamLabel.  d12 has the same dimensions as the
           Hessian, and turns out to be useful when computing the Hessian
           of functions of the probabilities.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed as in
           bulk_product, bulk_dproduct, and bulk_hproduct.


        Returns
        -------
        block_generator
          A generator which, when iterated, yields the 3-tuple 
          `(rowSlice, colSlice, hprobs)` or `(rowSlice, colSlice, dprobs12)`
          (the latter if `bReturnDProbs12 == True`).  `rowSlice` and `colSlice`
          are slices directly from `wrtSlicesList`. `hprobs` and `dprobs12` are
          arrays of shape K x S x B x B', where:

          - K is the length of spam_label_rows,
          - S is the number of gate strings (i.e. evalTree.num_final_strings()),
          - B is the number of parameter rows (the length of rowSlice)
          - B' is the number of parameter columns (the length of colSlice)

          If `mx`, `dp1`, and `dp2` are the outputs of :func:`bulk_fill_hprobs`
          (i.e. args `mxToFill`, `deriv1MxToFill`, and `deriv1MxToFill`), then:

          - `hprobs == mx[:,:,rowSlice,colSlice]`
          - `dprobs12 == dp1[:,:,rowSlice,None] * dp2[:,:,None,colSlice]`
        """
        raise NotImplementedError("bulk_hprobs_by_block(...) is not implemented!")
                    

    def frobeniusdist(self, otherCalc, transformMx=None,
                      itemWeights=None, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this
        gateset and otherGateSet.  Differences in each corresponding gate
        matrix and spam vector element are squared, weighted (using 
        `itemWeights` as applicable), then summed.  The value returned is the
        square root of this sum, or the square root of this sum divided by the
        number of summands if normalize == True.

        Parameters
        ----------
        otherCalc : GateCalc
            the other gate calculator to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by
            G => inv(transformMx) * G * transformMx, for each gate matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        itemWeights : dict, optional
           Dictionary of weighting factors for individual gates and spam
           operators. Weights are applied multiplicatively to the squared
           differences, i.e., (*before* the final square root is taken).  Keys
           can be gate, state preparation, POVM effect, or spam labels, as well
           as the two special labels `"gates"` and `"spam"` which apply to all
           of the gate or SPAM elements, respectively (but are overridden by
           specific element values).  Values are floating point numbers.
           By default, all weights are 1.0.

        normalize : bool, optional
           if True (the default), the sum of weighted squared-differences
           is divided by the weighted number of differences before the 
           final square root is taken.  If False, the division is not performed.

        Returns
        -------
        float
        """
        d = 0; T = transformMx
        nSummands = 0.0
        if itemWeights is None: itemWeights = {}
        gateWeight = itemWeights.get('gates',1.0)
        spamWeight = itemWeights.get('spam',1.0)

        if T is not None:
            Ti = _nla.inv(T) #TODO: generalize inverse op (call T.inverse() if T were a "transform" object?)
            for gateLabel, gate in self.gates.items():
                wt = itemWeights.get(gateLabel, gateWeight)
                d += wt * gate.frobeniusdist2(
                    otherCalc.gates[gateLabel], T, Ti)                
                nSummands += wt * (gate.dim)**2

            for lbl,rhoV in self.preps.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * rhoV.frobeniusdist2(otherCalc.preps[lbl],
                                              'prep', T, Ti)
                nSummands += wt * rhoV.dim

            for lbl,Evec in self.effects.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * Evec.frobeniusdist2(otherCalc.effects[lbl],
                                              'effect', T, Ti)
                nSummands += wt * Evec.dim

        else:
            for gateLabel,gate in self.gates.items():
                wt = itemWeights.get(gateLabel, gateWeight)
                d += wt * gate.frobeniusdist2(otherCalc.gates[gateLabel])
                nSummands += wt * (gate.dim)**2

            for lbl,rhoV in self.preps.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * rhoV.frobeniusdist2(otherCalc.preps[lbl],'prep')
                nSummands += wt * rhoV.dim

            for lbl,Evec in self.effects.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * Evec.frobeniusdist2(otherCalc.effects[lbl],'effect')
                nSummands += wt * Evec.dim

        #Temporary: check that this function can be computed by
        # calling residuals - replace with this later.
        resids, chk_nSummands = self.residuals(otherCalc, transformMx, itemWeights)
        assert(_np.isclose( _np.sum(resids**2), d))
        assert(_np.isclose( chk_nSummands, nSummands))
        
        if normalize and nSummands > 0:
            return _np.sqrt( d / nSummands )
        else:
            return _np.sqrt(d)

        
    def residuals(self, otherCalc, transformMx=None, itemWeights=None):
        """
        Compute the weighted residuals between two gate sets (the differences
        in corresponding gate matrix and spam vector elements).

        Parameters
        ----------
        otherCalc : GateCalc
            the other gate calculator to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by
            G => inv(transformMx) * G * transformMx, for each gate matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        itemWeights : dict, optional
           Dictionary of weighting factors for individual gates and spam
           operators. Weights applied such that they act multiplicatively on 
           the *squared* differences, so that the residuals themselves are
           scaled by the square roots of these weights.  Keys can be gate, state
           preparation, POVM effect, or spam labels, as well as the two special
           labels `"gates"` and `"spam"` which apply to all of the gate or SPAM
           elements, respectively (but are overridden by specific element
           values).  Values are floating point numbers.  By default, all weights
           are 1.0.

        Returns
        -------
        residuals : numpy.ndarray
            A 1D array of residuals (differences w.r.t. other)
        nSummands : int
            The (weighted) number of elements accounted for by the residuals.
        """
        resids = []
        T = transformMx
        nSummands = 0.0
        if itemWeights is None: itemWeights = {}
        sqrt_itemWeights = { k:_np.sqrt(v) for k,v in itemWeights.items() }
        gateWeight = sqrt_itemWeights.get('gates',1.0)
        spamWeight = sqrt_itemWeights.get('spam',1.0)

        if T is not None:
            Ti = _nla.inv(T) #TODO: generalize inverse op (call T.inverse() if T were a "transform" object?)
            for gateLabel, gate in self.gates.items():
                wt = sqrt_itemWeights.get(gateLabel, gateWeight)
                resids.append(
                        wt * gate.residuals(
                    otherCalc.gates[gateLabel], T, Ti))
                nSummands += wt**2 * (gate.dim)**2

            for lbl,rhoV in self.preps.items():
                wt = sqrt_itemWeights.get(lbl, spamWeight)
                resids.append(
                    wt * rhoV.residuals(otherCalc.preps[lbl],
                                              'prep', T, Ti))
                nSummands += wt**2 * rhoV.dim

            for lbl,Evec in self.effects.items():
                wt = sqrt_itemWeights.get(lbl, spamWeight)
                resids.append(
                    wt * Evec.residuals(otherCalc.effects[lbl],
                                        'effect', T, Ti))

                nSummands += wt**2 * Evec.dim

        else:
            for gateLabel,gate in self.gates.items():
                wt = sqrt_itemWeights.get(gateLabel, gateWeight)
                resids.append(
                    wt * gate.residuals(otherCalc.gates[gateLabel]))
                nSummands += wt**2 * (gate.dim)**2

            for lbl,rhoV in self.preps.items():
                wt = sqrt_itemWeights.get(lbl, spamWeight)
                resids.append(
                    wt * rhoV.residuals(otherCalc.preps[lbl],'prep'))
                nSummands += wt**2 * rhoV.dim

            for lbl,Evec in self.effects.items():
                wt = sqrt_itemWeights.get(lbl, spamWeight)
                resids.append(
                    wt * Evec.residuals(otherCalc.effects[lbl],'effect'))
                nSummands += wt**2 * Evec.dim

        resids = [r.flatten() for r in resids]
        resids = _np.concatenate(resids)
        return resids, nSummands

    def jtracedist(self, otherCalc, transformMx=None):
        """
        Compute the Jamiolkowski trace distance between two
        gatesets, defined as the maximum of the trace distances
        between each corresponding gate, including spam gates.

        Parameters
        ----------
        otherCalc : GateCalc
            the other gate set to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by
            G => inv(transformMx) * G * transformMx, for each gate matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        Returns
        -------
        float
        """
        T = transformMx
        d = 0 #spam difference
        nSummands = 0 # for spam terms
        
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ gate.jtracedist(otherCalc.gates[lbl], T, Ti)
                      for lbl,gate in self.gates.items() ]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            for lbl,rhoV in self.preps.items():
                d += rhoV.frobeniusdist2(otherCalc.preps[lbl],
                                         'prep', T, Ti)
                nSummands += rhoV.dim

            for lbl,Evec in self.effects.items():
                d += Evec.frobeniusdist2(otherCalc.effects[lbl],
                                         'effect', T, Ti)
                nSummands += Evec.dim
                
        else:
            dists = [ gate.jtracedist(otherCalc.gates[lbl])
                      for lbl,gate in self.gates.items() ]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            for lbl,rhoV in self.preps.items():
                d += rhoV.frobeniusdist2(otherCalc.preps[lbl],
                                         'prep')
                nSummands += rhoV.dim

            for lbl,Evec in self.effects.items():
                d += Evec.frobeniusdist2(otherCalc.effects[lbl],
                                         'effect')
                nSummands += Evec.dim

        spamVal = _np.sqrt(d / nSummands) if (nSummands > 0) else 0
        return max(dists) + spamVal


    def diamonddist(self, otherCalc, transformMx=None):
        """
        Compute the diamond-norm distance between two
        gatesets, defined as the maximum
        of the diamond-norm distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        otherCalc : GateCalc
            the other gate calculator to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by
            G => inv(transformMx) * G * transformMx, for each gate matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        Returns
        -------
        float
        """
        T = transformMx
        d = 0 #spam difference
        nSummands = 0 # for spam terms
        
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ gate.diamonddist(otherCalc.gates[lbl], T, Ti)
                      for lbl,gate in self.gates.items() ]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            for lbl,rhoV in self.preps.items():
                d += rhoV.frobeniusdist2(otherCalc.preps[lbl],
                                         'prep', T, Ti)
                nSummands += rhoV.dim

            for lbl,Evec in self.effects.items():
                d += Evec.frobeniusdist2(otherCalc.effects[lbl],
                                         'effect', T, Ti)
                nSummands += Evec.dim
                
        else:
            dists = [ gate.diamonddist(otherCalc.gates[lbl])
                      for lbl,gate in self.gates.items() ]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            for lbl,rhoV in self.preps.items():
                d += rhoV.frobeniusdist2(otherCalc.preps[lbl],
                                         'prep')
                nSummands += rhoV.dim

            for lbl,Evec in self.effects.items():
                d += Evec.frobeniusdist2(otherCalc.effects[lbl],
                                         'effect')
                nSummands += Evec.dim

        spamVal = _np.sqrt(d / nSummands) if (nSummands > 0) else 0
        return max(dists) + spamVal

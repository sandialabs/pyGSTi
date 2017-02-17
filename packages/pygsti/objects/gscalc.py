from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the GateSetCalculator class"""

import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla
import time as _time
import collections as _collections

from ..tools import gatetools as _gt
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from .profiler import DummyProfiler as _DummyProfiler
_dummy_profiler = _DummyProfiler()

# Smallness tolerances, used internally for conditional scaling required
# to control bulk products, their gradients, and their Hessians.
PSMALL = 1e-100
DSMALL = 1e-100
HSMALL = 1e-100

class GateSetCalculator(object):
    """
    Encapsulates a calculation tool used by gate set objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from GateSet to allow for additional
    gate set classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of gate matrices and SPAM vectors) access to these
    fundamental operations.
    """

    def __init__(self, dim, gates, preps, effects, povm_identity, spamdefs,
                 remainderLabel, identityLabel):
        """
        Construct a new GateSetCalculator object.

        Parameters
        ----------
        dim : int
            The gate-dimension.  All gate matrices should be dim x dim, and all
            SPAM vectors should be dim x 1.

        gates, preps, effects : OrderedDict
            Ordered dictionaries of Gate, SPAMVec, and SPAMVec objects,
            respectively.  Must be *ordered* dictionaries to specify a
            well-defined column ordering when taking derivatives.

        povm_identity : SPAMVec
            Identity vector (shape must be dim x 1) used when spamdefs
            contains the value (<rho_label>,"remainder"), which specifies
            a POVM effect that is the identity minus the sum of all the
            effect vectors in effects.

        spamdefs : OrderedDict
            A dictionary whose keys are the allowed SPAM labels, and whose
            values are 2-tuples comprised of a state preparation label
            followed by a POVM effect label (both of which are strings,
            and keys of preps and effects, respectively, except for the
            special case when eith both or just the effect label is set
            to "remainder").

        remainderLabel : string
            A string that may appear in the values of spamdefs to designate
            special behavior.

        identityLabel : string
            The string used to designate the identity POVM vector.
        """
        self._remainderLabel = remainderLabel
        self._identityLabel = identityLabel
        self.dim = dim
        self.gates = gates
        self.preps = preps
        self.effects = effects
        self.povm_identity = povm_identity
        self.spamdefs = spamdefs
        self.assumeSumToOne = bool( (self._remainderLabel,self._remainderLabel) in list(spamdefs.values()))
          #Whether spamdefs contains the value ("remainder", "remainder"),
          #  which specifies a spam label that generates probabilities such that
          #  all SPAM label probabilities sum exactly to 1.0.

        self.num_rho_params = [v.num_params() for v in list(self.preps.values())]
        self.num_e_params = [v.num_params() for v in list(self.effects.values())]
        self.num_gate_params = [g.num_params() for g in list(self.gates.values())]
        self.rho_offset = [ sum(self.num_rho_params[0:i]) for i in range(len(self.preps)+1) ]
        self.e_offset = [ sum(self.num_e_params[0:i]) for i in range(len(self.effects)+1) ]
        self.tot_rho_params = sum(self.num_rho_params)
        self.tot_e_params = sum(self.num_e_params)
        self.tot_gate_params = sum(self.num_gate_params)
        self.tot_spam_params = self.tot_rho_params + self.tot_e_params
        self.tot_params = self.tot_spam_params + self.tot_gate_params



    def _is_remainder_spamlabel(self, label):
        """
        Returns whether or not the given SPAM label is the
        special "remainder" SPAM label which generates
        probabilities such that all SPAM label probabilities
        sum exactly to 1.0.
        """
        return bool(self.spamdefs[label] == (self._remainderLabel, self._remainderLabel))

    def _get_evec(self, elabel):
        """
        Get a POVM effect vector by label.

        Parameters
        ----------
        elabel : string
            the label of the POVM effect vector to return.

        Returns
        -------
        numpy array
            an effect vector of shape (dim, 1).
        """
        if elabel == self._remainderLabel:
            return self.povm_identity - sum(self.effects.values())
        else:
            return self.effects[elabel]

    def _make_spamgate(self, spamlabel):
        prepLabel,effectLabel = self.spamdefs[spamlabel]
        if prepLabel == self._remainderLabel:
            return None

        rho,E = self.preps[prepLabel], self._get_evec(effectLabel)
        return _np.kron(rho.base, _np.conjugate(_np.transpose(E)))


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
        if bScale:
            scaledGatesAndExps = {};
            for (gateLabel,gatemx) in self.gates.items():
                ng = max(_nla.norm(gatemx),1.0)
                scaledGatesAndExps[gateLabel] = (gatemx / ng, _np.log(ng))

            scale_exp = 0
            G = _np.identity( self.dim )
            for lGate in gatestring:
                gate, ex = scaledGatesAndExps[lGate]
                H = _np.dot(gate,G)   # product of gates, starting with identity
                scale_exp += ex   # scale and keep track of exponent
                if H.max() < PSMALL and H.min() > -PSMALL:
                    nG = max(_nla.norm(G), _np.exp(-scale_exp))
                    G = _np.dot(gate,G/nG); scale_exp += _np.log(nG)
                    #OLD: _np.dot(G/nG,gate); scale_exp += _np.log(nG) LEXICOGRAPHICAL VS MATRIX ORDER
                else: G = H

            old_err = _np.seterr(over='ignore')
            scale = _np.exp(scale_exp)
            _np.seterr(**old_err)

            return G, scale

        else:
            G = _np.identity( self.dim )
            for lGate in gatestring:
                G = _np.dot(self.gates[lGate].base,G) #product of gates
                #OLD: G = _np.dot(G,self[lGate]) LEXICOGRAPHICAL VS MATRIX ORDER
            return G


    #Vectorizing Identities. (Vectorization)
    # Note when vectorizing op uses numpy.flatten rows are kept contiguous, so the first identity below is valid.
    # Below we use E(i,j) to denote the elementary matrix where all entries are zero except the (i,j) entry == 1

    # if vec(.) concatenates rows (which numpy.flatten does)
    # vec( A * E(0,1) * B ) = vec( mx w/ row_i = A[i,0] * B[row1] ) = A tensor B^T * vec( E(0,1) )
    # In general: vec( A * X * B ) = A tensor B^T * vec( X )

    # if vec(.) stacks columns
    # vec( A * E(0,1) * B ) = vec( mx w/ col_i = A[col0] * B[0,1] ) = B^T tensor A * vec( E(0,1) )
    # In general: vec( A * X * B ) = B^T tensor A * vec( X )

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

        # LEXICOGRAPHICAL VS MATRIX ORDER
        revGateLabelList = tuple(reversed(tuple(gatestring))) # we do matrix multiplication in this order (easier to think about)

        #  prod = G1 * G2 * .... * GN , a matrix
        #  dprod/d(gateLabel)_ij   = sum_{L s.t. G(L) == gatelabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)
        #  vec( dprod/d(gateLabel)_ij ) = sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                               = [ sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] * vec( dG(L)/dij) )
        #  if dG(L)/dij = E(i,j)
        #                               = vec(i,j)-col of [ sum_{L s.t. G(L) == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]]
        # So for each gateLabel the matrix [ sum_{L s.t. GL == gatelabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] has columns which
        #  correspond to the vectorized derivatives of each of the product components (i.e. prod_kl) with respect to a given gateLabel_ij
        # This function returns a concatenated form of the above matrices, so that each column corresponds to a (gateLabel,i,j) tuple and
        #  each row corresponds to an element of the product (els of prod.flatten()).
        #
        # Note: if gate G(L) is just a matrix of parameters, then dG(L)/dij = E(i,j), an elementary matrix

        dim = self.dim


        #Create per-gate with-respect-to parameter filters, used to
        # select a subset of all the derivative columns, essentially taking
        # a derivative of only a *subset* of all the gate's parameters
        fltr = {} #keys = gate labels, values = per-gate param indices
        if wrtFilter is not None:
            wrtIndexToGatelableIndexPair = []
            for lbl,g in self.gates.items():
                for k in range(g.num_params()):
                    wrtIndexToGatelableIndexPair.append((lbl,k))

            for gateLabel in list(self.gates.keys()):
                fltr[gateLabel] = []

            for i in wrtFilter:
                lbl,k = wrtIndexToGatelableIndexPair[i]
                fltr[lbl].append(k)

        #Cache partial products
        leftProds = [ ]
        G = _np.identity( dim ); leftProds.append(G)
        for gateLabel in revGateLabelList:
            G = _np.dot(G,self.gates[gateLabel].base)
            leftProds.append(G)

        rightProdsT = [ ]
        G = _np.identity( dim ); rightProdsT.append( _np.transpose(G) )
        for gateLabel in reversed(revGateLabelList):
            G = _np.dot(self.gates[gateLabel].base,G)
            rightProdsT.append( _np.transpose(G) )

        # Initialize storage
        dprod_dgateLabel = { }; dgate_dgateLabel = {}
        for gateLabel,gate in self.gates.items():
            iCols = fltr.get(gateLabel,None)
            nDerivCols = gate.num_params() if (iCols is None) else len(iCols)
            dprod_dgateLabel[gateLabel] = _np.zeros((dim**2, nDerivCols))
            dgate_dgateLabel[gateLabel] = gate.deriv_wrt_params(iCols)
              # (dim**2, nParams[gateLabel])

            #Note: replace gate.num_params()  and .deriv_wrt_params() with something more
            # general like parameterizer.get_num_params(gateLabel), etc, to allow for non-gate-local
            # gatesets params? In this case, would expect output to be of shape (dim**2, nTotParams)
            # and would *add* these together instead of concatenating below.

        #Add contributions for each gate in list
        N = len(revGateLabelList)
        for (i,gateLabel) in enumerate(revGateLabelList):
            dprod_dgate = _np.kron( leftProds[i], rightProdsT[N-1-i] )  # (dim**2, dim**2)
            dprod_dgateLabel[gateLabel] += _np.dot( dprod_dgate, dgate_dgateLabel[gateLabel] ) # (dim**2, nParams[gateLabel])

        #Concatenate per-gateLabel results to get final result
        to_concat = [ dprod_dgateLabel[gateLabel] for gateLabel in self.gates ]
        flattened_dprod = _np.concatenate( to_concat, axis=1 ) # axes = (vectorized_gate_el_index,gateset_parameter)

        if flat:
            return flattened_dprod
        else:
            vec_gs_size = flattened_dprod.shape[1]
            return _np.swapaxes( flattened_dprod, 0, 1 ).reshape( (vec_gs_size, dim, dim) ) # axes = (gate_ij, prod_row, prod_col)


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

        gatesToVectorize1 = list(self.gates.keys()) #which differentiation w.r.t. gates should be done
                                              # (which is all the differentiation done here)
        gatesToVectorize2 = list(self.gates.keys()) # (possibility to later specify different sets of gates
                                              #to differentiate firstly and secondly with)

        # LEXICOGRAPHICAL VS MATRIX ORDER
        revGateLabelList = tuple(reversed(tuple(gatestring))) # we do matrix multiplication in this order (easier to think about)

        #  prod = G1 * G2 * .... * GN , a matrix
        #  dprod/d(gateLabel)_ij   = sum_{L s.t. GL == gatelabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)
        #  d2prod/d(gateLabel1)_kl*d(gateLabel2)_ij = sum_{M s.t. GM == gatelabel1} sum_{L s.t. GL == gatelabel2, M < L}
        #                                                 [ G1 ... G(M-1) dG(M)/dkl G(M+1) ... G(L-1) dG(L)/dij G(L+1) ... GN ] + {similar with L < M} (if L == M ignore)
        #                                                 a matrix for each given (i,j,k,l)
        #  vec( d2prod/d(gateLabel1)_kl*d(gateLabel2)_ij ) = sum{...} [ G1 ...  G(M-1) dG(M)/dkl G(M+1) ... G(L-1) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                                                  = sum{...} [ unvec( G1 ...  G(M-1) tensor (G(M+1) ... G(L-1))^T vec( dG(M)/dkl ) )
        #                                                                tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]
        #                                                  + sum{ L < M} [ G1 ...  G(L-1) tensor
        #                                                       ( unvec( G(L+1) ... G(M-1) tensor (G(M+1) ... GN)^T vec( dG(M)/dkl ) ) )^T vec( dG(L)/dij ) ]
        #
        #  Note: ignoring L == M terms assumes that d^2 G/(dij)^2 == 0, which is true IF each gate matrix element is at most
        #        *linear* in each of the gate parameters.  If this is not the case, need Gate objects to have a 2nd-deriv method in addition of deriv_wrt_params
        #
        #  Note: unvec( X ) can be done efficiently by actually computing X^T ( note (A tensor B)^T = A^T tensor B^T ) and using numpy's reshape

        dim = self.dim

        #Cache partial products
        prods = {}
        ident = _np.identity( dim )
        for (i,gateLabel1) in enumerate(revGateLabelList): #loop over "starting" gate
            prods[ (i,i-1) ] = ident #product of no gates
            G = ident
            for (j,gateLabel2) in enumerate(revGateLabelList[i:],start=i): #loop over "ending" gate (>= starting gate)
                G = _np.dot(G,self.gates[gateLabel2].base)
                prods[ (i,j) ] = G
        prods[ (len(revGateLabelList),len(revGateLabelList)-1) ] = ident #product of no gates

        # Initialize storage
        dgate_dgateLabel = {}; nParams = {}
        for gateLabel in set(gatesToVectorize1).union(gatesToVectorize2):
            dgate_dgateLabel[gateLabel] = self.gates[gateLabel].deriv_wrt_params() # (dim**2, nParams[gateLabel])
            nParams[gateLabel] = dgate_dgateLabel[gateLabel].shape[1]

        d2prod_dgateLabels = { };
        for gateLabel1 in gatesToVectorize1:
            for gateLabel2 in gatesToVectorize2:
                d2prod_dgateLabels[(gateLabel1,gateLabel2)] = _np.zeros( (dim**2, nParams[gateLabel1], nParams[gateLabel2]), 'd')

        #Add contributions for each gate in list
        N = len(revGateLabelList)
        for m,gateLabel1 in enumerate(revGateLabelList):
            #OLD shortcut: if gateLabel1 in gatesToVectorize1: (and indent below)
            for l,gateLabel2 in enumerate(revGateLabelList):
                #OLD shortcut: if gateLabel2 in gatesToVectorize2: (and indent below)
                # FUTURE: we could add logic that accounts for the symmetry of the Hessian, so that
                # if gl1 and gl2 are both in gatesToVectorize1 and gatesToVectorize2 we only compute d2(prod)/d(gl1)d(gl2)
                # and not d2(prod)/d(gl2)d(gl1) ...
                if m < l:
                    x0 = _np.kron(_np.transpose(prods[(0,m-1)]),prods[(m+1,l-1)])  # (dim**2, dim**2)
                    x  = _np.dot( _np.transpose(dgate_dgateLabel[gateLabel1]), x0); xv = x.view() # (nParams[gateLabel1],dim**2)
                    xv.shape = (nParams[gateLabel1], dim, dim) # (reshape without copying - throws error if copy is needed)
                    y = _np.dot( _np.kron(xv, _np.transpose(prods[(l+1,N-1)])), dgate_dgateLabel[gateLabel2] )
                    # above: (nParams1,dim**2,dim**2) * (dim**2,nParams[gateLabel2]) = (nParams1,dim**2,nParams2)
                    d2prod_dgateLabels[(gateLabel1,gateLabel2)] += _np.swapaxes(y,0,1)
                            # above: dim = (dim2, nParams1, nParams2); swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
                elif l < m:
                    x0 = _np.kron(_np.transpose(prods[(l+1,m-1)]),prods[(m+1,N-1)]) # (dim**2, dim**2)
                    x  = _np.dot( _np.transpose(dgate_dgateLabel[gateLabel1]), x0); xv = x.view() # (nParams[gateLabel1],dim**2)
                    xv.shape = (nParams[gateLabel1], dim, dim) # (reshape without copying - throws error if copy is needed)
                    xv = _np.swapaxes(xv,1,2) # transposes each of the now un-vectorized dim x dim mxs corresponding to a single kl
                    y = _np.dot( _np.kron(prods[(0,l-1)], xv), dgate_dgateLabel[gateLabel2] )
# above: (nParams1,dim**2,dim**2) * (dim**2,nParams[gateLabel2]) = (nParams1,dim**2,nParams2)
                    d2prod_dgateLabels[(gateLabel1,gateLabel2)] += _np.swapaxes(y,0,1)
                    # above: dim = (dim2, nParams1, nParams2); swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
               #else l==m, in which case there's no contribution since we assume all gate elements are at most linear in the parameters

        #Concatenate per-gateLabel results to get final result (Note: this is the lengthy step for 2Q hessian calcs)
        to_concat = []
        for gateLabel1 in gatesToVectorize1:
            to_concat.append( _np.concatenate( [ d2prod_dgateLabels[(gateLabel1,gateLabel2)] for gateLabel2 in gatesToVectorize2 ], axis=2 ) ) #concat along ij (nParams2)
        flattened_d2prod = _np.concatenate( to_concat, axis=1 ) # concat along kl (nParams1)

        if wrtFilter1 is not None:
            flattened_d2prod = flattened_d2prod.take(wrtFilter1, axis=1)
              #take subset of 1st derivatives w.r.t. gateset parameter

        if wrtFilter2 is not None:
            flattened_d2prod = flattened_d2prod.take(wrtFilter2, axis=2)
              #take subset of 2nd derivatives w.r.t. gateset parameter

            # Alternate to take() above, but may be slow (no copyList precomputed) and
            # this shouldn't be a mem bottleneck, so just use numpy.take :
            #flattened_d2prod = inplace_take(flattened_d2prod, wrtFilter2, axis=2) 


        if flat:
            return flattened_d2prod # axes = (vectorized_gate_el_index, gateset_parameter1, gateset_parameter2)
        else:
            vec_kl_size, vec_ij_size = flattened_d2prod.shape[1:3]
            return _np.rollaxis( flattened_d2prod, 0, 3 ).reshape( (vec_kl_size, vec_ij_size, dim, dim) )
            # axes = (gateset_parameter1, gateset_parameter2, gateset_element_row, gateset_element_col)


    def pr(self, spamLabel, gatestring, clipTo=None, bUseScaling=True):
        """
        Compute the probability of the given gate sequence, where initialization
        & measurement operations are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        clipTo : 2-tuple, optional
          (min,max) to clip return value if not None.

        bUseScaling : bool, optional
          Whether to use a post-scaled product internally.  If False, this
          routine will run slightly faster, but with a chance that the
          product will overflow and the subsequent trace operation will
          yield nan as the returned probability.

        Returns
        -------
        float
        """

        if self._is_remainder_spamlabel(spamLabel):
            #then compute 1.0 - (all other spam label probabilities)
            otherSpamdefs = list(self.spamdefs.keys())[:]; del otherSpamdefs[ otherSpamdefs.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamdefs]) )
            return 1.0 - sum( [self.pr(sl, gatestring, clipTo, bUseScaling) for sl in otherSpamdefs] )

        (rholabel,elabel) = self.spamdefs[spamLabel]
        rho = self.preps[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        if bUseScaling:
            old_err = _np.seterr(over='ignore')
            G,scale = self.product(gatestring, True)
            p = float(_np.dot(E, _np.dot(G, rho)) * scale) # probability, with scaling applied (may generate overflow, but OK)

            #DEBUG: catch warnings to make sure correct (inf if value is large) evaluation occurs when there's a warning
            #bPrint = False
            #with _warnings.catch_warnings():
            #    _warnings.filterwarnings('error')
            #    try:
            #        test = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * scale
            #    except Warning: bPrint = True
            #if bPrint:  print 'Warning in Gateset.pr : scale=%g, trace=%g, p=%g' % (scale,_np.dot(self.SPAMs[spamLabel],G) ), p)
            _np.seterr(**old_err)

        else: #no scaling -- faster but susceptible to overflow
            G = self.product(gatestring, False)
            p = float(_np.dot(E, _np.dot(G, rho) ))

        if _np.isnan(p):
            if len(gatestring) < 10:
                strToPrint = str(gatestring)
            else:
                strToPrint = str(gatestring[0:10]) + " ... (len %d)" % len(gatestring)
            _warnings.warn("pr(%s) == nan" % strToPrint)
            #DEBUG: print "backtrace" of product leading up to nan

            #G = _np.identity( self.dim ); total_exp = 0.0
            #for i,lGate in enumerate(gateLabelList):
            #    G = _np.dot(G,self[lGate])  # product of gates, starting with G0
            #    nG = norm(G); G /= nG; total_exp += log(nG) # scale and keep track of exponent
            #
            #    p = _mt.trace( _np.dot(self.SPAMs[spamLabel],G) ) * exp(total_exp) # probability
            #    print "%d: p = %g, norm %g, exp %g\n%s" % (i,p,norm(G),total_exp,str(G))
            #    if _np.isnan(p): raise ValueError("STOP")

        if clipTo is not None:
            return _np.clip(p,clipTo[0],clipTo[1])
        else: return p


    def dpr(self, spamLabel, gatestring, returnPr=False,clipTo=None):
        """
        Compute the derivative of a probability generated by a gate string and
        spam label as a 1 x M numpy array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        returnPr : bool, optional
          when set to True, additionally return the probability itself.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        derivative : numpy array
            a 1 x M numpy array of derivatives of the probability w.r.t.
            each gateset parameter (M is the length of the vectorized gateset).

        probability : float
            only returned if returnPr == True.
        """

        if self._is_remainder_spamlabel(spamLabel):
            #then compute Deriv[ 1.0 - (all other spam label probabilities) ]
            otherSpamdefs = list(self.spamdefs.keys())[:]; del otherSpamdefs[ otherSpamdefs.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamdefs]) )
            otherResults = [self.dpr(sl, gatestring, returnPr, clipTo) for sl in otherSpamdefs]
            if returnPr:
                return -1.0 * sum([dpr for dpr,p in otherResults]), 1.0 - sum([p for dpr,p in otherResults])
            else:
                return -1.0 * sum(otherResults)

        #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
        #  dpr/d(gateLabel)_ij = sum E_k [dprod/d(gateLabel)_ij]_kl rho_l
        #  dpr/d(rho)_i = sum E_k prod_ki
        #  dpr/d(E)_i   = sum prod_il rho_l

        (rholabel,elabel) = self.spamdefs[spamLabel]
        rho = self.preps[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        #Derivs wrt Gates
        old_err = _np.seterr(over='ignore')
        prod,scale = self.product(gatestring, True)
        dprod_dGates = self.dproduct(gatestring); vec_gs_size = dprod_dGates.shape[0]
        dpr_dGates = _np.empty( (1, vec_gs_size) )
        for i in range(vec_gs_size):
            dpr_dGates[0,i] = float(_np.dot(E, _np.dot( dprod_dGates[i], rho)))

        if returnPr:
            p = _np.dot(E, _np.dot(prod, rho)) * scale  #may generate overflow, but OK
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )

        #Derivs wrt SPAM
        num_rho_params = [v.num_params() for v in list(self.preps.values())]
        rho_offset = [ sum(num_rho_params[0:i]) for i in range(len(self.preps)+1) ]
        rhoIndex = list(self.preps.keys()).index(rholabel)
        dpr_drhos = _np.zeros( (1, sum(num_rho_params)) )
        derivWrtAnyRhovec = scale * _np.dot(E,prod)
        dpr_drhos[0, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
            _np.dot( derivWrtAnyRhovec, rho.deriv_wrt_params())  #may overflow, but OK

        num_e_params = [v.num_params() for v in list(self.effects.values())]
        e_offset = [ sum(num_e_params[0:i]) for i in range(len(self.effects)+1) ]
        dpr_dEs = _np.zeros( (1, sum(num_e_params)) );
        derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod,rho)) # may overflow, but OK
           # (** doesn't depend on eIndex **) -- TODO: should also conjugate() here if complex?
        if elabel == self._remainderLabel:
            assert(self._remainderLabel not in self.effects) # "remainder" should be a distint *special* label
            for ei,evec in enumerate(self.effects.values()):  #compute Deriv w.r.t. [ 1 - sum_of_other_Effects ]
                dpr_dEs[0, e_offset[ei]:e_offset[ei+1]] = \
                    -1.0 * _np.dot( derivWrtAnyEvec, evec.deriv_wrt_params() )
        else:
            eIndex = list(self.effects.keys()).index(elabel)
            dpr_dEs[0, e_offset[eIndex]:e_offset[eIndex+1]] = \
                _np.dot( derivWrtAnyEvec, self.effects[elabel].deriv_wrt_params() )

        _np.seterr(**old_err)

        if returnPr:
            return _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 ), p
        else: return _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 )


    def hpr(self, spamLabel, gatestring, returnPr=False, returnDeriv=False,clipTo=None):
        """
        Compute the Hessian of a probability generated by a gate string and
        spam label as a 1 x M x M array, where M is the number of gateset
        parameters.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        gatestring : GateString or tuple of gate labels
          The sequence of gate labels specifying the gate string.

        returnPr : bool, optional
          when set to True, additionally return the probability itself.

        returnDeriv : bool, optional
          when set to True, additionally return the derivative of the
          probability.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        hessian : numpy array
            a 1 x M x M array, where M is the number of gateset parameters.
            hessian[0,j,k] is the derivative of the probability w.r.t. the
            k-th then the j-th gateset parameter.

        derivative : numpy array
            only returned if returnDeriv == True. A 1 x M numpy array of
            derivatives of the probability w.r.t. each gateset parameter.

        probability : float
            only returned if returnPr == True.
        """

        if self._is_remainder_spamlabel(spamLabel):
            #then compute Hessian[ 1.0 - (all other spam label probabilities) ]
            otherSpamdefs = list(self.spamdefs.keys())[:]; del otherSpamdefs[ otherSpamdefs.index(spamLabel) ]
            assert( not any([ self._is_remainder_spamlabel(sl) for sl in otherSpamdefs]) )
            otherResults = [self.hpr(sl, gatestring, returnPr, returnDeriv, clipTo) for sl in otherSpamdefs]
            if returnDeriv:
                if returnPr: return ( -1.0 * sum([hpr for hpr,dpr,p in otherResults]),
                                      -1.0 * sum([dpr for hpr,dpr,p in otherResults]),
                                       1.0 - sum([p   for hpr,dpr,p in otherResults])   )
                else:        return ( -1.0 * sum([hpr for hpr,dpr in otherResults]),
                                      -1.0 * sum([dpr for hpr,dpr in otherResults])     )
            else:
                if returnPr: return ( -1.0 * sum([hpr for hpr,p in otherResults]),
                                       1.0 - sum([p   for hpr,p in otherResults])   )
                else:        return   -1.0 * sum(otherResults)


        #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
        #  d2pr/d(gateLabel1)_mn d(gateLabel2)_ij = sum E_k [dprod/d(gateLabel1)_mn d(gateLabel2)_ij]_kl rho_l
        #  d2pr/d(rho)_i d(gateLabel)_mn = sum E_k [dprod/d(gateLabel)_mn]_ki     (and same for other diff order)
        #  d2pr/d(E)_i d(gateLabel)_mn   = sum [dprod/d(gateLabel)_mn]_il rho_l   (and same for other diff order)
        #  d2pr/d(E)_i d(rho)_j          = prod_ij                                (and same for other diff order)
        #  d2pr/d(E)_i d(E)_j            = 0
        #  d2pr/d(rho)_i d(rho)_j        = 0

        (rholabel,elabel) = self.spamdefs[spamLabel]
        rho = self.preps[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))

        d2prod_dGates = self.hproduct(gatestring)
        vec_gs_size = d2prod_dGates.shape[0]
        assert( d2prod_dGates.shape[0] == d2prod_dGates.shape[1] )

        d2pr_dGates2 = _np.empty( (1, vec_gs_size, vec_gs_size) )
        for i in range(vec_gs_size):
            for j in range(vec_gs_size):
                d2pr_dGates2[0,i,j] = float(_np.dot(E, _np.dot( d2prod_dGates[i,j], rho)))

        old_err = _np.seterr(over='ignore')

        prod,scale = self.product(gatestring, True)
        if returnPr:
            p = _np.dot(E, _np.dot(prod, rho)) * scale  #may generate overflow, but OK
            if clipTo is not None:  p = _np.clip( p, clipTo[0], clipTo[1] )

        dprod_dGates  = self.dproduct(gatestring)
        assert( dprod_dGates.shape[0] == vec_gs_size )
        if returnDeriv: # same as in dpr(...)
            dpr_dGates = _np.empty( (1, vec_gs_size) )
            for i in range(vec_gs_size):
                dpr_dGates[0,i] = float(_np.dot(E, _np.dot( dprod_dGates[i], rho)))


        #Derivs wrt SPAM
        num_rho_params = [v.num_params() for v in list(self.preps.values())]
        num_e_params = [v.num_params() for v in list(self.effects.values())]
        rho_offset = [ sum(num_rho_params[0:i]) for i in range(len(self.preps)+1) ]
        e_offset = [ sum(num_e_params[0:i]) for i in range(len(self.effects)+1) ]
        rhoIndex = list(self.preps.keys()).index(rholabel)

        if returnDeriv:  #same as in dpr(...)
            dpr_drhos = _np.zeros( (1, sum(num_rho_params)) )
            derivWrtAnyRhovec = scale * _np.dot(E,prod)
            dpr_drhos[0, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                _np.dot( derivWrtAnyRhovec, rho.deriv_wrt_params())  #may overflow, but OK

            dpr_dEs = _np.zeros( (1, sum(num_e_params)) );
            derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod,rho)) # may overflow, but OK
            if elabel == self._remainderLabel:
                assert(self._remainderLabel not in self.effects)
                for ei,evec in enumerate(self.effects.values()):  #compute Deriv w.r.t. [ 1 - sum_of_other_Effects ]
                    dpr_dEs[0, e_offset[ei]:e_offset[ei+1]] = \
                        -1.0 * _np.dot( derivWrtAnyEvec, evec.deriv_wrt_params() )
            else:
                eIndex = list(self.effects.keys()).index(elabel)
                dpr_dEs[0, e_offset[eIndex]:e_offset[eIndex+1]] = \
                    _np.dot( derivWrtAnyEvec, self.effects[elabel].deriv_wrt_params() )

            dpr = _np.concatenate( (dpr_drhos,dpr_dEs,dpr_dGates), axis=1 )

        d2pr_drhos = _np.zeros( (1, vec_gs_size, sum(num_rho_params)) )
        d2pr_drhos[0, :, sum(num_rho_params[0:rhoIndex]):sum(num_rho_params[0:rhoIndex+1])] \
            = _np.dot( _np.dot(E,dprod_dGates), rho.deriv_wrt_params())[0] # (= [0,:,:])

        d2pr_dEs = _np.zeros( (1, vec_gs_size, sum(num_e_params)) )
        derivWrtAnyEvec = _np.squeeze(_np.dot(dprod_dGates,rho), axis=(2,))
        if elabel == self._remainderLabel:
            assert(self._remainderLabel not in self.effects)
            for ei,evec in enumerate(self.effects.values()): #similar to above, but now after a deriv w.r.t gates
                d2pr_dEs[0, :, e_offset[ei]:e_offset[ei+1]] = \
                    -1.0 * _np.dot( derivWrtAnyEvec, evec.deriv_wrt_params() )
        else:
            eIndex = list(self.effects.keys()).index(elabel)
            d2pr_dEs[0, :, e_offset[eIndex]:e_offset[eIndex+1]] = \
                _np.dot(derivWrtAnyEvec, self.effects[elabel].deriv_wrt_params())

        d2pr_dErhos = _np.zeros( (1, sum(num_e_params), sum(num_rho_params)) )
        derivWrtAnyEvec = scale * _np.dot(prod, rho.deriv_wrt_params()) #may generate overflow, but OK

        if elabel == self._remainderLabel:
            for ei,evec in enumerate(self.effects.values()): #similar to above, but now after also a deriv w.r.t rhos
                d2pr_dErhos[0, e_offset[ei]:e_offset[ei+1], rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                    -1.0 * _np.dot( _np.transpose(evec.deriv_wrt_params()),derivWrtAnyEvec)
                # ET*P*rho -> drhoP -> ET*P*drho/drhoP = ((P*drho/drhoP)^T*E)^T -> dEp ->
                # ((P*drho/drhoP)^T*dE/dEp)^T = dE/dEp^T*(P*drho/drhoP) = (d,eP)^T*(d,rhoP) = (eP,rhoP) OK!
        else:
            eIndex = list(self.effects.keys()).index(elabel)
            d2pr_dErhos[0, e_offset[eIndex]:e_offset[eIndex+1],
                        rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = \
                        _np.dot( _np.transpose(self.effects[elabel].deriv_wrt_params()),derivWrtAnyEvec)

        d2pr_d2rhos = _np.zeros( (1, sum(num_rho_params), sum(num_rho_params)) )
        d2pr_d2Es   = _np.zeros( (1, sum(num_e_params), sum(num_e_params)) )

        ret_row1 = _np.concatenate( ( d2pr_d2rhos, _np.transpose(d2pr_dErhos,(0,2,1)), _np.transpose(d2pr_drhos,(0,2,1)) ), axis=2) # wrt rho
        ret_row2 = _np.concatenate( ( d2pr_dErhos, d2pr_d2Es, _np.transpose(d2pr_dEs,(0,2,1)) ), axis=2 ) # wrt E
        ret_row3 = _np.concatenate( ( d2pr_drhos,d2pr_dEs,d2pr_dGates2), axis=2 ) #wrt gates
        ret = _np.concatenate( (ret_row1, ret_row2, ret_row3), axis=1 )

        _np.seterr(**old_err)

        if returnDeriv:
            if returnPr: return ret, dpr, p
            else:        return ret, dpr
        else:
            if returnPr: return ret, p
            else:        return ret


    def probs(self, gatestring, clipTo=None):
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
        probs = { }
        if not self.assumeSumToOne:
            for spamLabel in self.spamdefs:
                probs[spamLabel] = self.pr(spamLabel, gatestring, clipTo)
        else:
            s = 0; lastLabel = None
            for spamLabel in self.spamdefs:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one "remainder" spam label
                    lastLabel = spamLabel; continue
                probs[spamLabel] = self.pr(spamLabel, gatestring, clipTo)
                s += probs[spamLabel]
            if lastLabel is not None:
                probs[lastLabel] = 1.0 - s  #last spam label is computed so sum == 1
        return probs


    def dprobs(self, gatestring, returnPr=False,clipTo=None):
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
        if not self.assumeSumToOne:
            for spamLabel in self.spamdefs:
                dprobs[spamLabel] = self.dpr(spamLabel, gatestring, returnPr,clipTo)
        else:
            ds = None; s=0; lastLabel = None
            for spamLabel in self.spamdefs:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                dprobs[spamLabel] = self.dpr(spamLabel, gatestring, returnPr,clipTo)
                if returnPr:
                    ds = dprobs[spamLabel][0] if ds is None else ds + dprobs[spamLabel][0]
                    s += dprobs[spamLabel][1]
                else:
                    ds = dprobs[spamLabel] if ds is None else ds + dprobs[spamLabel]
            if lastLabel is not None:
                dprobs[lastLabel] = (-ds,1.0-s) if returnPr else -ds
        return dprobs



    def hprobs(self, gatestring, returnPr=False, returnDeriv=False, clipTo=None):
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
        if not self.assumeSumToOne:
            for spamLabel in self.spamdefs:
                hprobs[spamLabel] = self.hpr(spamLabel, gatestring, returnPr,
                                             returnDeriv,clipTo)
        else:
            hs = None; ds=None; s=0; lastLabel = None
            for spamLabel in self.spamdefs:
                if self._is_remainder_spamlabel(spamLabel):
                    assert(lastLabel is None) # ensure there is at most one dummy spam label
                    lastLabel = spamLabel; continue
                hprobs[spamLabel] = self.hpr(spamLabel, gatestring, returnPr,
                                             returnDeriv,clipTo)
                if returnPr:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                        s += hprobs[spamLabel][2]
                    else:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        s += hprobs[spamLabel][1]
                else:
                    if returnDeriv:
                        hs = hprobs[spamLabel][0] if hs is None else hs + hprobs[spamLabel][0]
                        ds = hprobs[spamLabel][1] if ds is None else ds + hprobs[spamLabel][1]
                    else:
                        hs = hprobs[spamLabel] if hs is None else hs + hprobs[spamLabel]

            if lastLabel is not None:
                if returnPr:
                    hprobs[lastLabel] = (-hs,-ds,1.0-s) if returnDeriv else (-hs,1.0-s)
                else:
                    hprobs[lastLabel] = (-hs,-ds) if returnDeriv else -hs

        return hprobs


    def _compute_product_cache(self, evalTree, comm=None):
        """
        Computes a tree of products in a linear cache space. Will *not*
        parallelize computation, even if given a split tree (since there's
        no good way to reconstruct the parent tree's *non-final* elements from 
        those of the sub-trees).  Note also that there would be no memory savings
        from using a split tree.  In short, parallelization should be done at a
        higher level.
        """

        dim = self.dim

        #Note: previously, we tried to allow for parallelization of
        # _compute_product_cache when the tree was split, but this is was 
        # incorrect (and luckily never used) - so it's been removed.
        
        if comm is not None: #ignorning comm since can't do anything with it!
            #_warnings.warn("More processors than can be used for product computation")
            pass #this is a fairly common occurance, and doesn't merit a warning

        # ------------------------------------------------------------------

        if evalTree.is_split():
            _warnings.warn("Ignoring tree splitting in product cache calc.")

        cacheSize = len(evalTree)
        prodCache = _np.zeros( (cacheSize, dim, dim) )
        scaleCache = _np.zeros( cacheSize, 'd' )

        #First element of cache are given by evalTree's initial single- or zero-gate labels
        for i,gateLabel in zip(evalTree.get_init_indices(), evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                prodCache[i] = _np.identity( dim )
                # Note: scaleCache[i] = 0.0 from initialization
            else:
                gate = self.gates[gateLabel].base
                nG = max(_nla.norm(gate), 1.0)
                prodCache[i] = gate / nG
                scaleCache[i] = _np.log(nG)

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        #cnt = 0
        for i in evalTree.get_evaluation_order():
            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft) = evalTree[i]   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            prodCache[i] = _np.dot(L,R)
            scaleCache[i] = scaleCache[iLeft] + scaleCache[iRight]

            if prodCache[i].max() < PSMALL and prodCache[i].min() > -PSMALL:
                nL,nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]),1e-300)
                sL, sR = L/nL, R/nR
                prodCache[i] = _np.dot(sL,sR); scaleCache[i] += _np.log(nL) + _np.log(nR)

        #print "bulk_product DEBUG: %d rescalings out of %d products" % (cnt, len(evalTree))

        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0]  #may be duplicates (a list, not a set)
        assert( len(nanOrInfCacheIndices) == 0 ) # since all scaled gates start with norm <= 1, products should all have norm <= 1

        return prodCache, scaleCache


    def _compute_dproduct_cache(self, evalTree, prodCache, scaleCache,
                                comm=None, wrtSlice=None, profiler=None):
        """
        Computes a tree of product derivatives in a linear cache space. Will
        use derivative columns and then (and only when needed) a split tree
        to parallelize computation, since there are no memory savings
        from using a split tree.
        """

        if profiler is None: profiler = _dummy_profiler
        dim = self.dim
        nGateDerivCols = self.tot_gate_params if (wrtSlice is None) \
                           else _slct.length(wrtSlice)
        deriv_shape = (nGateDerivCols, dim, dim)
        cacheSize = len(evalTree)

        # ------------------------------------------------------------------

        #print("MPI: _compute_dproduct_cache begin: %d deriv cols" % nGateDerivCols)
        if comm is not None and comm.Get_size() > 1:
            #print("MPI: _compute_dproduct_cache called w/comm size %d" % comm.Get_size())
            # parallelize of deriv cols, then sub-trees (if available and necessary)

            if comm.Get_size() > nGateDerivCols:

                #If there are more processors than deriv cols, give a
                # warning -- note that we *cannot* make use of a tree being
                # split because there's no good way to reconstruct the
                # *non-final* parent-tree elements from those of the sub-trees.
                _warnings.warn("Increased speed could be obtained" +
                               " by giving dproduct cache computation" +
                               " *fewer* processors and *smaller* (sub-)tree" +
                               " (e.g. by splitting tree beforehand), as there"+
                               " are more cpus than derivative columns.")

            # Use comm to distribute columns
            allDerivColSlice = slice(0,nGateDerivCols) if (wrtSlice is None) else wrtSlice
            _, myDerivColSlice, _, mySubComm = \
                _mpit.distribute_slice(allDerivColSlice, comm)
            #print("MPI: _compute_dproduct_cache over %d cols (%s) (rank %d computing %s)" \
            #    % (nGateDerivCols, str(allDerivColIndices), comm.Get_rank(), str(myDerivColIndices)))
            if mySubComm is not None and mySubComm.Get_size() > 1:
                _warnings.warn("Too many processors to make use of in " +
                               " _compute_dproduct_cache.")
                if mySubComm.Get_rank() > 0: myDerivColSlice = slice(0,0)
                  #don't compute anything on "extra", i.e. rank != 0, cpus

            my_results = self._compute_dproduct_cache(
                evalTree, prodCache, scaleCache, None, myDerivColSlice, profiler)
                # pass None as comm, *not* mySubComm, since we can't do any
                #  further parallelization

            tm = _time.time()
            all_results = comm.allgather(my_results)
            profiler.add_time("MPI IPC", tm)
            return _np.concatenate(all_results, axis=1) #TODO: remove this concat w/better gather?

        # ------------------------------------------------------------------
        tSerialStart = _time.time()

        if evalTree.is_split():
            _warnings.warn("Ignoring tree splitting in dproduct cache calc.")

        dProdCache = _np.zeros( (cacheSize,) + deriv_shape )

        # This iteration **must** match that in bulk_evaltree
        #   in order to associate the right single-gate-strings w/indices
        wrtIndices = _slct.indices(wrtSlice) if (wrtSlice is not None) else None
        for i,gateLabel in zip(evalTree.get_init_indices(), evalTree.get_init_labels()):
            if gateLabel == "": #special case of empty label == no gate
                dProdCache[i] = _np.zeros( deriv_shape )
            else:                
                dgate = self.dproduct( (gateLabel,) , wrtFilter=wrtIndices)
                dProdCache[i] = dgate / _np.exp(scaleCache[i])

        #profiler.print_mem("DEBUGMEM: POINT1"); profiler.comm.barrier()

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        for i in evalTree.get_evaluation_order():
            tm = _time.time()
            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft) = evalTree[i]   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            dL,dR = dProdCache[iLeft], dProdCache[iRight]
            dProdCache[i] = _np.dot(dL, R) + \
                _np.swapaxes(_np.dot(L, dR),0,1) #dot(dS, T) + dot(S, dT)
            profiler.add_time("compute_dproduct_cache: dots", tm)
            profiler.add_count("compute_dproduct_cache: dots")

            scale = scaleCache[i] - (scaleCache[iLeft] + scaleCache[iRight])
            if abs(scale) > 1e-8: # _np.isclose(scale,0) is SLOW!
                dProdCache[i] /= _np.exp(scale)
                if dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
            elif _np.count_nonzero(dProdCache[i]) and dProdCache[i].max() < DSMALL and dProdCache[i].min() > -DSMALL:
                _warnings.warn("Would have scaled dProd but now will not alter scaleCache.")

        #profiler.print_mem("DEBUGMEM: POINT2"); profiler.comm.barrier()

        profiler.add_time("compute_dproduct_cache: serial", tSerialStart)
        profiler.add_count("compute_dproduct_cache: num columns", nGateDerivCols)

        return dProdCache


    def _compute_hproduct_cache(self, evalTree, prodCache, dProdCache1,
                                dProdCache2, scaleCache, comm=None,
                                wrtSlice1=None, wrtSlice2=None):
        """
        Computes a tree of product 2nd derivatives in a linear cache space. Will
        use derivative rows and columns and then (as needed) a split tree
        to parallelize computation, since there are no memory savings
        from using a split tree.
        """

        dim = self.dim

        # Note: dProdCache?.shape = (#gatestrings,#params_to_diff_wrt,dim,dim)
        nGateDerivCols1 = dProdCache1.shape[1]
        nGateDerivCols2 = dProdCache2.shape[1]
        assert(wrtSlice1 is None or _slct.length(wrtSlice1) == nGateDerivCols1)
        assert(wrtSlice2 is None or _slct.length(wrtSlice2) == nGateDerivCols2)
        hessn_shape = (nGateDerivCols1, nGateDerivCols2, dim, dim)
        cacheSize = len(evalTree)

        # ------------------------------------------------------------------

        if comm is not None and comm.Get_size() > 1:
            # parallelize of deriv cols, then sub-trees (if available and necessary)

            if comm.Get_size() > nGateDerivCols1*nGateDerivCols2:
                #If there are more processors than deriv cells, give a
                # warning -- note that we *cannot* make use of a tree being
                # split because there's no good way to reconstruct the
                # *non-final* parent-tree elements from those of the sub-trees.
                _warnings.warn("Increased speed could be obtained" +
                               " by giving hproduct cache computation" +
                               " *fewer* processors and *smaller* (sub-)tree" +
                               " (e.g. by splitting tree beforehand), as there"+
                               " are more cpus than hessian elements.")

            # allocate final result memory
            hProdCache = _np.zeros( (cacheSize,) + hessn_shape )            

            # Use comm to distribute columns
            allDeriv1ColSlice = slice(0,nGateDerivCols1)
            allDeriv2ColSlice = slice(0,nGateDerivCols2)
            deriv1Slices, myDeriv1ColSlice, deriv1Owners, mySubComm = \
                _mpit.distribute_slice(allDeriv1ColSlice, comm)

            # Get slice into entire range of gateset params so that
            #  per-gate hessians can be computed properly
            if wrtSlice1 is not None and wrtSlice1.start is not None:
                myHessianSlice1 = _slct.shift(myDeriv1ColSlice, wrtSlice1.start)
            else: myHessianSlice1 = myDeriv1ColSlice

            #print("MPI: _compute_hproduct_cache over %d cols (rank %d computing %s)" \
            #    % (nGateDerivCols2, comm.Get_rank(), str(myDerivColSlice)))

            if mySubComm is not None and mySubComm.Get_size() > 1:
                deriv2Slices, myDeriv2ColSlice, deriv2Owners, mySubSubComm = \
                    _mpit.distribute_slice(allDeriv2ColSlice, mySubComm)

                # Get slice into entire range of gateset params (see above)
                if wrtSlice2 is not None and wrtSlice2.start is not None:
                    myHessianSlice2 = _slct.shift(myDeriv2ColSlice, wrtSlice2.start)
                else: myHessianSlice2 = myDeriv2ColSlice

                if mySubSubComm is not None and mySubSubComm.Get_size() > 1:
                    _warnings.warn("Too many processors to make use of in " +
                                   " _compute_hproduct_cache.")
                    #TODO: remove: not needed now that we track owners
                    #if mySubSubComm.Get_rank() > 0: myDeriv2ColSlice = slice(0,0)
                    #  #don't compute anything on "extra", i.e. rank != 0, cpus

                hProdCache[:,myDeriv1ColSlice,myDeriv2ColSlice] = self._compute_hproduct_cache(
                    evalTree, prodCache, dProdCache1[:,myDeriv1ColSlice], dProdCache2[:,myDeriv2ColSlice],
                    scaleCache, None, myHessianSlice1, myHessianSlice2)
                    # pass None as comm, *not* mySubSubComm, since we can't do any further parallelization

                _mpit.gather_slices(deriv2Slices, deriv2Owners, hProdCache[:,myDeriv1ColSlice],
                                    2, mySubComm) #, gatherMemLimit) #gather over col-distribution (Deriv2)
                  #note: gathering axis 2 of hProdCache[:,myDeriv1ColSlice],
                  #      dim=(cacheSize,nGateDerivCols1,nGateDerivCols2,dim,dim)
            else:
                #compute "Deriv1" row-derivatives distribution only; don't use column distribution
                hProdCache[:,myDeriv1ColSlice] = self._compute_hproduct_cache(
                    evalTree, prodCache, dProdCache1[:,myDeriv1ColSlice], dProdCache2,
                    scaleCache, None, myHessianSlice1, wrtSlice2)
                    # pass None as comm, *not* mySubComm (this is ok, see "if" condition above)

            _mpit.gather_slices(deriv1Slices, deriv1Owners, hProdCache, 1, comm)
                        #, gatherMemLimit) #gather over row-distribution (Deriv1)
              #note: gathering axis 1 of hProdCache,
              #      dim=(cacheSize,nGateDerivCols1,nGateDerivCols2,dim,dim)

            return hProdCache

        # ------------------------------------------------------------------

        if evalTree.is_split():
            _warnings.warn("Ignoring tree splitting in hproduct cache calc.")

        hProdCache = _np.zeros( (cacheSize,) + hessn_shape )

        #First element of cache are given by evalTree's initial single- or zero-gate labels
        for i,_ in zip(evalTree.get_init_indices(), evalTree.get_init_labels()):
            hProdCache[i] = _np.zeros( hessn_shape )
            #assume all gate elements are at most linear in params,
            # all hessiansl for single- or zero-gate strings are zero.

            #OLD (slow and unnecessary unless we relax linear assumption):
            #if gateLabel == "": #special case of empty label == no gate
            #    hProdCache[i] = _np.zeros( hessn_shape )
            #else:
            #    hgate = self.hproduct( (gateLabel,), 
            #                           wrtFilter1=_slct.indices(wrtSlice1),
            #                           wrtFilter2=_slct.indices(wrtSlice2))
            #    hProdCache[i] = hgate / _np.exp(scaleCache[i])

        #evaluate gate strings using tree (skip over the zero and single-gate-strings)
        for i in evalTree.get_evaluation_order():

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from evalTree because
            # (iRight,iLeft,iFinal) = tup implies gatestring[i] = gatestring[iLeft] + gatestring[iRight], but we want:
            (iRight,iLeft) = evalTree[i]   # since then matrixOf(gatestring[i]) = matrixOf(gatestring[iLeft]) * matrixOf(gatestring[iRight])
            L,R = prodCache[iLeft], prodCache[iRight]
            dL1,dR1 = dProdCache1[iLeft], dProdCache1[iRight]
            dL2,dR2 = dProdCache2[iLeft], dProdCache2[iRight]
            hL,hR = hProdCache[iLeft], hProdCache[iRight]
              # Note: L, R = GxG ; dL,dR = vgs x GxG ; hL,hR = vgs x vgs x GxG

            dLdRa = _np.swapaxes(_np.dot(dL1,dR2),1,2)
            dLdRb = _np.swapaxes(_np.dot(dL2,dR1),1,2)
            dLdR_sym = dLdRa + _np.swapaxes(dLdRb,0,1) 

            hProdCache[i] = _np.dot(hL, R) + dLdR_sym + _np.transpose(_np.dot(L,hR),(1,2,0,3))

            scale = scaleCache[i] - (scaleCache[iLeft] + scaleCache[iRight])
            if abs(scale) > 1e-8: # _np.isclose(scale,0) is SLOW!
                hProdCache[i] /= _np.exp(scale)
                if hProdCache[i].max() < HSMALL and hProdCache[i].min() > -HSMALL:
                    _warnings.warn("Scaled hProd small in order to keep prod managable.")
            elif _np.count_nonzero(hProdCache[i]) and hProdCache[i].max() < HSMALL and hProdCache[i].min() > -HSMALL:
                _warnings.warn("hProd is small (oh well!).")

        return hProdCache


## END CACHE FUNCTIONS


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
        prodCache, scaleCache = self._compute_product_cache(evalTree,comm)

        #use cached data to construct return values
        Gs = evalTree.final_view(prodCache, axis=0)
           #shape == ( len(gatestring_list), dim, dim ), Gs[i] is product for i-th gate string
        scaleExps = evalTree.final_view(scaleCache)

        old_err = _np.seterr(over='ignore')
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bScale:
            return Gs, scaleVals
        else:
            old_err = _np.seterr(over='ignore')
            Gs = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
            _np.seterr(**old_err)
            return Gs



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
        nGateStrings = evalTree.num_final_strings()
        nGateDerivCols = self.tot_gate_params
        dim = self.dim

        wrtSlice = _slct.list_to_slice(wrtFilter) if (wrtFilter is not None) else None
          #TODO: just allow slices as argument: wrtFilter -> wrtSlice?
        prodCache, scaleCache = self._compute_product_cache(evalTree, comm)
        dProdCache = self._compute_dproduct_cache(evalTree, prodCache, scaleCache,
                                                  comm, wrtSlice)

        #use cached data to construct return values
        old_err = _np.seterr(over='ignore')
        scaleExps = evalTree.final_view( scaleCache )
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bReturnProds:
            Gs  = evalTree.final_view(prodCache, axis=0)
              #shape == ( len(gatestring_list), dim, dim ), 
              # Gs[i] is product for i-th gate string

            dGs = evalTree.final_view(dProdCache, axis=0) 
              #shape == ( len(gatestring_list), nGateDerivCols, dim, dim ),
              # dGs[i] is dprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                Gs  = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                _np.seterr(**old_err)

            if flat:
                dGs =  _np.swapaxes( _np.swapaxes(dGs,0,1).reshape(
                    (nGateDerivCols, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened everything else

            return (dGs, Gs, scaleVals) if bScale else (dGs, Gs)

        else:
            dGs = evalTree.final_view(dProdCache, axis=0) 
              #shape == ( len(gatestring_list), nGateDerivCols, dim, dim ),
              # dGs[i] is dprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                dGs = _np.swapaxes( _np.swapaxes(dGs,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs[_np.isnan(dGs)] =  0 #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value, and we
                                        # assume the zero deriv value trumps since we've renormed to keep all the products within decent bounds
                #assert( len( (_np.isnan(dGs)).nonzero()[0] ) == 0 )
                #assert( len( (_np.isinf(dGs)).nonzero()[0] ) == 0 )
                #dGs = clip(dGs,-1e300,1e300)
                _np.seterr(**old_err)

            if flat:
                dGs =  _np.swapaxes( _np.swapaxes(dGs,0,1).reshape(
                    (nGateDerivCols, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened everything else
            return (dGs, scaleVals) if bScale else dGs



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
        dim = self.dim
        nGateDerivCols1 = self.tot_gate_params if (wrtFilter1 is None) else _slct.length(wrtFilter1)
        nGateDerivCols2 = self.tot_gate_params if (wrtFilter2 is None) else _slct.length(wrtFilter2)
        nGateStrings = evalTree.num_final_strings() #len(gatestring_list)
        wrtSlice1 = _slct.list_to_slice(wrtFilter1) if (wrtFilter1 is not None) else None
        wrtSlice2 = _slct.list_to_slice(wrtFilter2) if (wrtFilter2 is not None) else None
          #TODO: just allow slices as argument: wrtFilter -> wrtSlice?

        prodCache, scaleCache = self._compute_product_cache(evalTree, comm)
        dProdCache1 = self._compute_dproduct_cache(evalTree, prodCache, scaleCache,
                                                  comm, wrtSlice1)        
        dProdCache2 = dProdCache1 if (wrtSlice1 == wrtSlice2) else \
            self._compute_dproduct_cache(evalTree, prodCache, scaleCache,
                                         comm, wrtSlice2)

        hProdCache = self._compute_hproduct_cache(evalTree, prodCache, dProdCache1, dProdCache2,
                                                  scaleCache, comm, wrtSlice1, wrtSlice2)

        #use cached data to construct return values
        old_err = _np.seterr(over='ignore')
        scaleExps = evalTree.final_view(scaleCache)
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)

        if bReturnDProdsAndProds:
            Gs  = evalTree.final_view( prodCache, axis=0)
              #shape == ( len(gatestring_list), dim, dim ), 
              # Gs[i] is product for i-th gate string

            dGs1 = evalTree.final_view(dProdCache1, axis=0)
            dGs2 = evalTree.final_view(dProdCache2, axis=0)
              #shape == ( len(gatestring_list), nGateDerivColsX, dim, dim ),
              # dGs[i] is dprod_dGates for ith string

            hGs = evalTree.final_view(hProdCache, axis=0)
              #shape == ( len(gatestring_list), nGateDerivCols1, nGateDerivCols2, dim, dim ),
              # hGs[i] is hprod_dGates for ith string

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                Gs  = _np.swapaxes( _np.swapaxes(Gs,0,2) * scaleVals, 0,2)  #may overflow, but ok
                dGs1 = _np.swapaxes( _np.swapaxes(dGs1,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                dGs2 = _np.swapaxes( _np.swapaxes(dGs2,0,3) * scaleVals, 0,3) #may overflow or get nans (invalid), but ok
                hGs = _np.swapaxes( _np.swapaxes(hGs,0,4) * scaleVals, 0,4) #may overflow or get nans (invalid), but ok
                dGs1[_np.isnan(dGs1)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                dGs2[_np.isnan(dGs2)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value (see below)
                hGs[_np.isnan(hGs)] = 0  #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero hessian value (see below)
                _np.seterr(**old_err)

            if flat:
                dGs1 = _np.swapaxes( _np.swapaxes(dGs1,0,1).reshape( (nGateDerivCols1, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened all else
                dGs2 = _np.swapaxes( _np.swapaxes(dGs2,0,1).reshape( (nGateDerivCols2, nGateStrings*dim**2) ), 0,1 ) # cols = deriv cols, rows = flattened all else
                hGs = _np.rollaxis( _np.rollaxis(hGs,0,3).reshape( (nGateDerivCols1, nGateDerivCols2, nGateStrings*dim**2) ), 2) # cols = deriv cols, rows = all else

            return (hGs, dGs1, dGs2, Gs, scaleVals) if bScale else (hGs, dGs1, dGs2, Gs)

        else:
            hGs = evalTree.final_view(hProdCache, axis=0) 
              #shape == ( len(gatestring_list), nGateDerivCols, nGateDerivCols, dim, dim )

            if not bScale:
                old_err = _np.seterr(over='ignore', invalid='ignore')
                hGs = _np.swapaxes( _np.swapaxes(hGs,0,4) * scaleVals, 0,4) #may overflow or get nans (invalid), but ok
                hGs[_np.isnan(hGs)] =  0 #convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero hessian value, and we
                                         # assume the zero hessian value trumps since we've renormed to keep all the products within decent bounds
                #assert( len( (_np.isnan(hGs)).nonzero()[0] ) == 0 )
                #assert( len( (_np.isinf(hGs)).nonzero()[0] ) == 0 )
                #hGs = clip(hGs,-1e300,1e300)
                _np.seterr(**old_err)

            if flat: hGs = _np.rollaxis( _np.rollaxis(hGs,0,3).reshape( (nGateDerivCols1, nGateDerivCols2, nGateStrings*dim**2) ), 2) # as above

            return (hGs, scaleVals) if bScale else hGs


    def _scaleExp(self, scaleExps):
        old_err = _np.seterr(over='ignore')
        scaleVals = _np.exp(scaleExps) #may overflow, but OK if infs occur here
        _np.seterr(**old_err)
        return scaleVals


    def _rhoE_from_spamLabel(self, spamLabel):
        (rholabel,elabel) = self.spamdefs[spamLabel]
        rho = self.preps[rholabel]
        E   = _np.conjugate(_np.transpose(self._get_evec(elabel)))
        return rho,E

    def _probs_from_rhoE(self, spamLabel, rho, E, Gs, scaleVals):
        #Compute probability and save in return array
        # want vp[iFinal] = float(dot(E, dot(G, rho)))  ##OLD, slightly slower version: p = trace(dot(self.SPAMs[spamLabel], G))
        #  vp[i] = sum_k,l E[0,k] Gs[i,k,l] rho[l,0] * scaleVals[i]
        #  vp[i] = sum_k E[0,k] dot(Gs, rho)[i,k,0]  * scaleVals[i]
        #  vp[i] = dot( E, dot(Gs, rho))[0,i,0]      * scaleVals[i]
        #  vp    = squeeze( dot( E, dot(Gs, rho)), axis=(0,2) ) * scaleVals
        return _np.squeeze( _np.dot(E, _np.dot(Gs, rho)), axis=(0,2) ) * scaleVals
          # shape == (len(gatestring_list),) ; may overflow but OK

    def _dprobs_from_rhoE(self, spamLabel, rho, E, Gs, dGs, scaleVals, wrtSlices=None):
        (rholabel,elabel) = self.spamdefs[spamLabel]
        nGateStrings = Gs.shape[0]

        #Compute d(probability)/dGates and save in return list (now have G,dG => product, dprod_dGates)
        #  prod, dprod_dGates = G,dG
        # dp_dGates[i,j] = sum_k,l E[0,k] dGs[i,j,k,l] rho[l,0]
        # dp_dGates[i,j] = sum_k E[0,k] dot( dGs, rho )[i,j,k,0]
        # dp_dGates[i,j] = dot( E, dot( dGs, rho ) )[0,i,j,0]
        # dp_dGates      = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,3))
        old_err2 = _np.seterr(invalid='ignore', over='ignore')
        dp_dGates = _np.squeeze( _np.dot( E, _np.dot( dGs, rho ) ), axis=(0,3) ) * scaleVals[:,None]
        _np.seterr(**old_err2)
           # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols)
           # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
           #  this case set to zero since we can't tell whether it's + or - inf anyway...
        dp_dGates[ _np.isnan(dp_dGates) ] = 0

        #DEBUG
        #assert( len( (_np.isnan(scaleVals)).nonzero()[0] ) == 0 )
        #xxx = _np.dot( E, _np.dot( dGs, rho ) )
        #assert( len( (_np.isnan(xxx)).nonzero()[0] ) == 0 )
        #if len( (_np.isnan(dp_dGates)).nonzero()[0] ) != 0:
        #    print "scaleVals = ",_np.min(scaleVals),", ",_np.max(scaleVals)
        #    print "xxx = ",_np.min(xxx),", ",_np.max(xxx)
        #    print len( (_np.isinf(xxx)).nonzero()[0] )
        #    print len( (_np.isinf(scaleVals)).nonzero()[0] )
        #    assert( len( (_np.isnan(dp_dGates)).nonzero()[0] ) == 0 )

        #SPAM -------------

        # Get: dp_drhos[i, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = dot(E,Gs[i],drho/drhoP)
        # dp_drhos[i,J0+J] = sum_kl E[0,k] Gs[i,k,l] drhoP[l,J]
        # dp_drhos[i,J0+J] = dot(E, Gs, drhoP)[0,i,J]
        # dp_drhos[:,J0+J] = squeeze(dot(E, Gs, drhoP),axis=(0,))[:,J]
        rhoIndex = list(self.preps.keys()).index(rholabel)
        dp_drhos = _np.zeros( (nGateStrings, self.tot_rho_params ) )
        dp_drhos[: , self.rho_offset[rhoIndex]:self.rho_offset[rhoIndex+1] ] = \
            _np.squeeze(_np.dot(_np.dot(E, Gs), rho.deriv_wrt_params()),axis=(0,)) \
            * scaleVals[:,None] # may overflow, but OK

        # Get: dp_dEs[i, e_offset[eIndex]:e_offset[eIndex+1]] = dot(transpose(dE/dEP),Gs[i],rho))
        # dp_dEs[i,J0+J] = sum_lj dEPT[J,j] Gs[i,j,l] rho[l,0]
        # dp_dEs[i,J0+J] = sum_j dEP[j,J] dot(Gs, rho)[i,j]
        # dp_dEs[i,J0+J] = sum_j dot(Gs, rho)[i,j,0] dEP[j,J]
        # dp_dEs[i,J0+J] = dot(squeeze(dot(Gs, rho),2), dEP)[i,J]
        # dp_dEs[:,J0+J] = dot(squeeze(dot(Gs, rho),axis=(2,)), dEP)[:,J]
        dp_dEs = _np.zeros( (nGateStrings, self.tot_e_params) )
        dp_dAnyE = _np.squeeze(_np.dot(Gs, rho),axis=(2,)) * scaleVals[:,None] #may overflow, but OK (deriv w.r.t any of self.effects - independent of which)
        if elabel == self._remainderLabel:
            for ei,evec in enumerate(self.effects.values()): #compute Deriv w.r.t. [ 1 - sum_of_other_Effects ]
                dp_dEs[:,self.e_offset[ei]:self.e_offset[ei+1]] = -1.0 * _np.dot(dp_dAnyE, evec.deriv_wrt_params())
        else:
            eIndex = list(self.effects.keys()).index(elabel)
            dp_dEs[:,self.e_offset[eIndex]:self.e_offset[eIndex+1]] = \
                _np.dot(dp_dAnyE, self.effects[elabel].deriv_wrt_params())

        if wrtSlices is None:
            sub_vdp = _np.concatenate( (dp_drhos,dp_dEs,dp_dGates), axis=1 )
        else:
            sub_vdp = _np.concatenate((dp_drhos[:,wrtSlices['preps']],
                                       dp_dEs[:,wrtSlices['effects']],
                                       dp_dGates), axis=1 )
        return sub_vdp


    def _get_filter_info(self, wrtSlices):
        """ 
        Returns a "filter" object containing info about the mapping
        of prep and effect parameters onto a final "filtered" set.
        """
        PrepEffectFilter = _collections.namedtuple(
            'PrepEffectFilter', 'rho_local_slices rho_global_slices ' +
            'e_local_slices e_global_slices num_rho_params num_e_params')
      
        if wrtSlices is not None:
            loc_rho_slices = [ 
                _slct.shift(_slct.intersect(
                        wrtSlices['preps'],
                        slice(self.rho_offset[i],self.rho_offset[i+1])),
                            -self.rho_offset[i]) for i in range(len(self.preps))]
            tmp_num_params = [_slct.length(s) for s in loc_rho_slices]
            tmp_offsets = [ sum(tmp_num_params[0:i]) for i in range(len(self.preps)+1) ]
            global_rho_slices = [ slice(tmp_offsets[i],tmp_offsets[i+1]) 
                                  for i in range(len(self.preps)) ]

            loc_e_slices = [ 
                _slct.shift(_slct.intersect(
                        wrtSlices['effects'],
                        slice(self.e_offset[i],self.e_offset[i+1])),
                            -self.e_offset[i]) for i in range(len(self.effects))]
            tmp_num_params = [_slct.length(s) for s in loc_e_slices]
            tmp_offsets = [ sum(tmp_num_params[0:i]) for i in range(len(self.effects)+1) ]
            global_e_slices = [ slice(tmp_offsets[i],tmp_offsets[i+1]) 
                                  for i in range(len(self.effects)) ]

            return PrepEffectFilter(rho_local_slices=loc_rho_slices,
                                    rho_global_slices=global_rho_slices,
                                    e_local_slices=loc_e_slices,
                                    e_global_slices=global_e_slices,
                                    num_rho_params=_slct.length(wrtSlices['preps']),
                                    num_e_params=_slct.length(wrtSlices['effects']))
        else:
            loc_rho_slices = [slice(None,None)]*len(self.preps)
            loc_e_slices = [slice(None,None)]*len(self.effects)
            global_rho_slices = [slice(self.rho_offset[i],self.rho_offset[i+1]) for i in range(len(self.preps)) ]
            global_e_slices = [slice(self.e_offset[i],self.e_offset[i+1]) for i in range(len(self.effects)) ]
            return PrepEffectFilter(rho_local_slices=loc_rho_slices,
                                    rho_global_slices=global_rho_slices,
                                    e_local_slices=loc_e_slices,
                                    e_global_slices=global_e_slices,
                                    num_rho_params=self.tot_rho_params,
                                    num_e_params=self.tot_e_params)
                               


    def _hprobs_from_rhoE(self, spamLabel, rho, E, Gs, dGs1, dGs2, hGs, scaleVals,
                          wrtSlices1=None, wrtSlices2=None):
        (rholabel,elabel) = self.spamdefs[spamLabel]
        nGateStrings = Gs.shape[0]
        flt1 = self._get_filter_info(wrtSlices1)
        flt2 = self._get_filter_info(wrtSlices2)

        # GATE DERIVS (assume hGs is already sized/filtered) -------------------

        #Compute d2(probability)/dGates2 and save in return list
        # d2pr_dGates2[i,j,k] = sum_l,m E[0,l] hGs[i,j,k,l,m] rho[m,0]
        # d2pr_dGates2[i,j,k] = sum_l E[0,l] dot( dGs, rho )[i,j,k,l,0]
        # d2pr_dGates2[i,j,k] = dot( E, dot( dGs, rho ) )[0,i,j,k,0]
        # d2pr_dGates2        = squeeze( dot( E, dot( dGs, rho ) ), axis=(0,4))
        old_err2 = _np.seterr(invalid='ignore', over='ignore')
        d2pr_dGates2 = _np.squeeze( _np.dot( E, _np.dot( hGs, rho ) ), axis=(0,4) ) * scaleVals[:,None,None]
        _np.seterr(**old_err2)

        # may overflow, but OK ; shape == (len(gatestring_list), nGateDerivCols, nGateDerivCols)
        # may also give invalid value due to scaleVals being inf and dot-prod being 0. In
        #  this case set to zero since we can't tell whether it's + or - inf anyway...
        d2pr_dGates2[ _np.isnan(d2pr_dGates2) ] = 0


        # SPAM DERIVS (assume dGs1 and dGs2 are already sized/filtered) --------
        
        vec_gs_size1 = dGs1.shape[1]
        vec_gs_size2 = dGs2.shape[1]

        # Get: d2pr_drhos[i, j, rho_offset[rhoIndex]:rho_offset[rhoIndex+1]] = dot(E,dGs[i,j],drho/drhoP))
        # d2pr_drhos[i,j,J0+J] = sum_kl E[0,k] dGs[i,j,k,l] drhoP[l,J]
        # d2pr_drhos[i,j,J0+J] = dot(E, dGs, drhoP)[0,i,j,J]
        # d2pr_drhos[:,:,J0+J] = squeeze(dot(E, dGs, drhoP),axis=(0,))[:,:,J]
        rhoIndex = list(self.preps.keys()).index(rholabel)
        drho = rho.deriv_wrt_params()[:,flt2.rho_local_slices[rhoIndex]]
        d2pr_drhos1 = _np.zeros( (nGateStrings, vec_gs_size1, flt2.num_rho_params) )
        d2pr_drhos1[:, :, flt2.rho_global_slices[rhoIndex]] = \
            _np.squeeze( _np.dot(_np.dot(E,dGs1),drho), axis=(0,)) \
            * scaleVals[:,None,None] #overflow OK

        # get d2pr_drhos where gate derivatives are wrt the 2nd set of gate parameters
        if dGs1 is dGs2 and wrtSlices1 == wrtSlices2: #TODO: better check for equivalence: maybe let dGs2 be None?
            assert(vec_gs_size1 == vec_gs_size2)
            d2pr_drhos2 = _np.transpose(d2pr_drhos1,(0,2,1))
        else:
            drho = rho.deriv_wrt_params()[:,flt1.rho_local_slices[rhoIndex]]
            d2pr_drhos2 = _np.zeros( (nGateStrings, vec_gs_size2, flt1.num_rho_params) )
            d2pr_drhos2[:, :, flt1.rho_global_slices[rhoIndex]] = \
                _np.squeeze( _np.dot(_np.dot(E,dGs2),drho), axis=(0,)) \
                * scaleVals[:,None,None] #overflow OK
            d2pr_drhos2 = _np.transpose(d2pr_drhos2,(0,2,1))


        # Get: d2pr_dEs[i, j, e_offset[eIndex]:e_offset[eIndex+1]] = dot(transpose(dE/dEP),dGs[i,j],rho)
        # d2pr_dEs[i,j,J0+J] = sum_kl dEPT[J,k] dGs[i,j,k,l] rho[l,0]
        # d2pr_dEs[i,j,J0+J] = sum_k dEP[k,J] dot(dGs, rho)[i,j,k,0]
        # d2pr_dEs[i,j,J0+J] = dot( squeeze(dot(dGs, rho),axis=(3,)), dEP)[i,j,J]
        # d2pr_dEs[:,:,J0+J] = dot( squeeze(dot(dGs, rho),axis=(3,)), dEP)[:,:,J]
        d2pr_dEs1 = _np.zeros( (nGateStrings, vec_gs_size1, flt2.num_e_params) )
        dp_dAnyE = _np.squeeze(_np.dot(dGs1,rho), axis=(3,)) * scaleVals[:,None,None] #overflow OK
        if elabel == self._remainderLabel:
            for ei,evec in enumerate(self.effects.values()):
                devec = evec.deriv_wrt_params()[:,flt2.e_local_slices[ei]]
                d2pr_dEs1[:, :, flt2.e_global_slices[ei]] = -1.0 * _np.dot(dp_dAnyE, devec)
        else:
            eIndex = list(self.effects.keys()).index(elabel)
            devec = self.effects[elabel].deriv_wrt_params()[:,flt2.e_local_slices[eIndex]]
            d2pr_dEs1[:, :, flt2.e_global_slices[eIndex]] = \
                _np.dot(dp_dAnyE, devec)

        # get d2pr_dEs where gate derivatives are wrt the 2nd set of gate parameters
        if dGs1 is dGs2 and wrtSlices1 == wrtSlices2: #TODO: better check for equivalence: maybe let dGs2 be None?
            assert(vec_gs_size1 == vec_gs_size2)
            d2pr_dEs2 = _np.transpose(d2pr_dEs1,(0,2,1))
        else:
            d2pr_dEs2 = _np.zeros( (nGateStrings, vec_gs_size2, flt1.num_e_params) )
            dp_dAnyE = _np.squeeze(_np.dot(dGs2,rho), axis=(3,)) * scaleVals[:,None,None] #overflow OK
            if elabel == self._remainderLabel:
                for ei,evec in enumerate(self.effects.values()):
                    devec = evec.deriv_wrt_params()[:,flt1.e_local_slices[ei]]
                    d2pr_dEs2[:, :, flt1.e_global_slices[ei]] = -1.0 * _np.dot(dp_dAnyE, devec)
            else:
                eIndex = list(self.effects.keys()).index(elabel)
                devec = self.effects[elabel].deriv_wrt_params()[:,flt1.e_local_slices[eIndex]]
                d2pr_dEs2[:, :, flt1.e_global_slices[eIndex]] = \
                    _np.dot(dp_dAnyE, devec)
            d2pr_dEs2 = _np.transpose(d2pr_dEs2,(0,2,1))


        # Get: d2pr_dErhos[i, e_offset[eIndex]:e_offset[eIndex+1], e_offset[rhoIndex]:e_offset[rhoIndex+1]] =
        #    dEP^T * prod[i,:,:] * drhoP
        # d2pr_dErhos[i,J0+J,K0+K] = sum jk dEPT[J,j] prod[i,j,k] drhoP[k,K]
        # d2pr_dErhos[i,J0+J,K0+K] = sum j dEPT[J,j] dot(prod,drhoP)[i,j,K]
        # d2pr_dErhos[i,J0+J,K0+K] = dot(dEPT,prod,drhoP)[J,i,K]
        # d2pr_dErhos[i,J0+J,K0+K] = swapaxes(dot(dEPT,prod,drhoP),0,1)[i,J,K]
        # d2pr_dErhos[:,J0+J,K0+K] = swapaxes(dot(dEPT,prod,drhoP),0,1)[:,J,K]
        d2pr_dErhos1 = _np.zeros( (nGateStrings, flt1.num_e_params, flt2.num_rho_params) )
        drho = rho.deriv_wrt_params()[:,flt2.rho_local_slices[rhoIndex]]
        dp_dAnyE = _np.dot(Gs, drho) * scaleVals[:,None,None] #overflow OK
        if elabel == self._remainderLabel:
            for ei,evec in enumerate(self.effects.values()):
                devec = evec.deriv_wrt_params()[:, flt1.e_local_slices[ei]]
                d2pr_dErhos1[:, flt1.e_global_slices[ei], flt2.rho_global_slices[rhoIndex]] = \
                    -1.0 * _np.swapaxes( _np.dot(_np.transpose(devec), dp_dAnyE ), 0,1)
        else:
            eIndex = list(self.effects.keys()).index(elabel)
            devec = self.effects[elabel].deriv_wrt_params()[:, flt1.e_local_slices[eIndex]]
            d2pr_dErhos1[:, flt1.e_global_slices[eIndex], flt2.rho_global_slices[rhoIndex]] = \
                _np.swapaxes( _np.dot(_np.transpose(devec), dp_dAnyE ), 0,1)

        # get d2pr_dEs where E derivatives are wrt the 2nd set of gate parameters
        if wrtSlices1 == wrtSlices2: #Note: this doesn't involve gate derivatives
            d2pr_dErhos2 = _np.transpose(d2pr_dErhos1,(0,2,1))
        else:
            d2pr_dErhos2 = _np.zeros( (nGateStrings, flt2.num_e_params, flt1.num_rho_params) )
            drho = rho.deriv_wrt_params()[:,flt1.rho_local_slices[rhoIndex]]
            dp_dAnyE = _np.dot(Gs, drho) * scaleVals[:,None,None] #overflow OK
            if elabel == self._remainderLabel:
                for ei,evec in enumerate(self.effects.values()):
                    devec = evec.deriv_wrt_params()[:, flt2.e_local_slices[ei]]
                    d2pr_dErhos2[:, flt2.e_global_slices[ei], flt1.rho_global_slices[rhoIndex]] = \
                        -1.0 * _np.swapaxes( _np.dot(_np.transpose(devec), dp_dAnyE ), 0,1)
            else:
                eIndex = list(self.effects.keys()).index(elabel)
                devec = self.effects[elabel].deriv_wrt_params()[:, flt2.e_local_slices[eIndex]]
                d2pr_dErhos2[:, flt2.e_global_slices[eIndex], flt1.rho_global_slices[rhoIndex]] = \
                    _np.swapaxes( _np.dot(_np.transpose(devec), dp_dAnyE ), 0,1)
            d2pr_dErhos2 = _np.transpose(d2pr_dErhos2,(0,2,1))

                
        #Note: these 2nd derivatives would need to be modified from being all-zeros if 
        # the spam vectors were allowed to be more than linear in their parameters.
        d2pr_d2rhos = _np.zeros( (nGateStrings, flt1.num_rho_params, flt2.num_rho_params) )
        d2pr_d2Es   = _np.zeros( (nGateStrings, flt1.num_e_params, flt2.num_e_params) )

        # END SPAM DERIVS -----------------------
        
        ret_row1 = _np.concatenate( ( d2pr_d2rhos,  d2pr_dErhos2, d2pr_drhos2  ), axis=2 ) # wrt rho
        ret_row2 = _np.concatenate( ( d2pr_dErhos1, d2pr_d2Es,    d2pr_dEs2    ), axis=2 ) # wrt E
        ret_row3 = _np.concatenate( ( d2pr_drhos1,  d2pr_dEs1,    d2pr_dGates2 ), axis=2 ) # wrt gates
           
        sub_vhp = _np.concatenate( (ret_row1, ret_row2, ret_row3), axis=1 )
        return sub_vhp


    def _check(self, evalTree, spam_label_rows, prMxToFill=None, dprMxToFill=None, hprMxToFill=None, clipTo=None):
        # compare with older slower version that should do the same thing (for debugging)
        for spamLabel,rowIndex in spam_label_rows.items():
            gatestring_list = evalTree.generate_gatestring_list(permute=False)

            if prMxToFill is not None:
                check_vp = _np.array( [ self.pr(spamLabel, gateString, clipTo) for gateString in gatestring_list ] )
                if _nla.norm(prMxToFill[rowIndex] - check_vp) > 1e-6:
                    _warnings.warn("norm(vp-check_vp) = %g - %g = %g" % \
                               (_nla.norm(prMxToFill[rowIndex]),
                                _nla.norm(check_vp),
                                _nla.norm(prMxToFill[rowIndex] - check_vp)))
                    #for i,gs in enumerate(gatestring_list):
                    #    if abs(vp[i] - check_vp[i]) > 1e-7:
                    #        print "   %s => p=%g, check_p=%g, diff=%g" % (str(gs),vp[i],check_vp[i],abs(vp[i]-check_vp[i]))

            if dprMxToFill is not None:
                check_vdp = _np.concatenate(
                    [ self.dpr(spamLabel, gateString, False,clipTo)
                      for gateString in gatestring_list ], axis=0 )
                if _nla.norm(dprMxToFill[rowIndex] - check_vdp) > 1e-6:
                    _warnings.warn("norm(vdp-check_vdp) = %g - %g = %g" %
                          (_nla.norm(dprMxToFill[rowIndex]),
                           _nla.norm(check_vdp),
                           _nla.norm(dprMxToFill[rowIndex] - check_vdp)))

            if hprMxToFill is not None:
                check_vhp = _np.concatenate(
                    [ self.hpr(spamLabel, gateString, False,False,clipTo)
                      for gateString in gatestring_list ], axis=0 )
                if _nla.norm(hprMxToFill[rowIndex] - check_vhp) > 1e-6:
                    _warnings.warn("norm(vhp-check_vhp) = %g - %g = %g" %
                             (_nla.norm(hprMxToFill[rowIndex]),
                              _nla.norm(check_vhp),
                              _nla.norm(hprMxToFill[rowIndex] - check_vhp)))

    #def _compute_sub_result(self, spam_label_rows, calc_from_spamlabel_fn):
    #
    #    remainder_label = None
    #    sub_results = {}
    #    for spamLabel in spam_label_rows.keys():
    #        if self._is_remainder_spamlabel(spamLabel):
    #            remainder_label = spamLabel
    #            continue
    #        sub_results[spamLabel] = calc_from_spamlabel_fn(spamLabel)
    #
    #    #compute remainder label
    #    if remainder_label is not None:
    #        sums = None
    #        for spamLabel in self.spamdefs: #loop over ALL spam labels
    #            if spamLabel == remainder_label: continue # except "remainder"
    #            sub = sub_results.get(spamLabel, calc_from_spamlabel_fn(spamLabel))
    #
    #            if sums is None: sums = [None]*len(sub)
    #            for i,s in enumerate(sums):
    #                sums[i] = sub[i] if (s is None) else (s + sub[i])
    #
    #        csums = [ 1.0-sums[0] ] if (sums[0] is not None) else [ None ]
    #        csums.extend( [ -sums[i] if (sums[i] is not None) else None \
    #                             for i in range(1,len(sums)) ] )
    #        sub_results[remainder_label] = tuple(csums)
    #    return sub_results


    def _fill_result_tuple(self, result_tup, spam_label_rows, tree_slice,
                           param_slice1, param_slice2, calc_and_fill_fn):
        fslc = tree_slice
        pslc1 = param_slice1
        pslc2 = param_slice2
        remainder_index = None
        for spamLabel,rowIndex in spam_label_rows.items():
            if self._is_remainder_spamlabel(spamLabel):
                remainder_index = rowIndex; continue
            calc_and_fill_fn(spamLabel,rowIndex,fslc,pslc1,pslc2,False)

        #compute remainder label
        if remainder_index is not None:
            # nps[k] == num of param slices in result_tup[k] index (assume
            #           the first two dims are spamLabel and a gatestring indx.
            nps = { k: (el.ndim-2) 
                    for k,el in enumerate(result_tup) if el is not None }
                
            def mkindx(iSpam,k): 
                """ Constructs multi-index appropriate for result_tup[k]
                    (Note that pslc1,pslc2 alwsys act on *final* dimension2)   """
                if nps[k] > 1: addl = [slice(None)]*(nps[k]-2)+[pslc1,pslc2]
                elif nps[k] == 1: addl = [pslc1]
                else: addl = []
                return [ iSpam,fslc ] + addl

            non_none_result_indices = [ i for i in range(len(result_tup)) \
                                           if result_tup[i] is not None ]

            for i in non_none_result_indices: #zero out for ensuing sum
                result_tup[i][mkindx(remainder_index,i)] = 0

            for spamLabel in self.spamdefs: #loop over ALL spam labels
                if self._is_remainder_spamlabel(spamLabel): 
                    continue # ...except remainder label

                rowIndex = spam_label_rows.get(spamLabel,None)
                if rowIndex is not None:
                    for i in non_none_result_indices:                        
                        result_tup[i][mkindx(remainder_index,i)] += \
                            result_tup[i][mkindx(rowIndex,i)]
                else:
                    calc_and_fill_fn(spamLabel,remainder_index,fslc,
                                     pslc1,pslc2,sumInto=True)

            #At this point, result_tup[i][remainder_index,fslc,...] contains the 
            # sum of the results from all other spam labels.
            for i in non_none_result_indices:
                mi = mkindx(remainder_index,i)
                result_tup[i][mi] *= -1.0
                if nps[i] == 0: # special case: when there are no param slices,
                    result_tup[i][mi] += 1.0 # result == 1.0-sum (not just -sum)
                    
        return


    #def _fill_sub_result(self, result_tup, spam_label_rows, calc_from_spamlabel_fn):
    #
    #    remainder_index = None
    #    for spamLabel,rowIndex in spam_label_rows.items():
    #        if self._is_remainder_spamlabel(spamLabel):
    #            remainder_index = rowIndex
    #            continue
    #        sub = calc_from_spamlabel_fn(spamLabel)
    #        for i,val in enumerate(sub):
    #            if result_tup[i] is not None:
    #                result_tup[i][rowIndex] = val
    #            else: assert(val is None)
    #
    #    #compute remainder label
    #    if remainder_index is not None:
    #        sums = None
    #        for spamLabel in self.spamdefs: #loop over ALL spam labels
    #            if self._is_remainder_spamlabel(spamLabel): continue
    #
    #            rowIndex = spam_label_rows.get(spamLabel,None)
    #            if rowIndex is not None:
    #                sub = [ ]
    #                for i in range(len(result_tup)):
    #                    if result_tup[i] is not None:
    #                        sub.append( result_tup[i][rowIndex] )
    #                    else: sub.append(None)
    #            else:
    #                sub = calc_from_spamlabel_fn(spamLabel)
    #
    #            if sums is None: sums = [None]*len(sub)
    #            for i,s in enumerate(sums):
    #                sums[i] = sub[i] if (s is None) else (s + sub[i])
    #
    #        csums = [ 1.0-sums[0] ] if (sums[0] is not None) else [ None ]
    #        csums.extend( [ -sums[i] if (sums[i] is not None) else None \
    #                             for i in range(1,len(sums)) ] )
    #        for i,val in enumerate(csums):
    #            if val is not None:
    #                result_tup[i][remainder_index] = val
    #    return




    def bulk_fill_probs(self, mxToFill, spam_label_rows,
                        evalTree, clipTo=None, check=False, comm=None):

        """
        Identical to bulk_probs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probabilities for all gate strings
        and a given SPAM label are placed into the row of
        mxToFill specified by spam_label_rows[spamLabel].

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated KxS numpy array, where K is larger
          than the maximum value in spam_label_rows and S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings())

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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

        remainder_row_index = None
        for spamLabel,rowIndex in spam_label_rows.items():
            if self._is_remainder_spamlabel(spamLabel):
                assert(self.assumeSumToOne) # ensure the remainder label is allowed
                assert(remainder_row_index is None) # ensure there is at most one dummy spam label
                remainder_row_index = rowIndex

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            fslc = evalSubTree.final_slice(evalTree)

            #Free memory from previous subtree iteration before computing caches
            scaleVals = Gs = prodCache = scaleCache = None

            #Fill cache info
            prodCache, scaleCache = self._compute_product_cache(evalSubTree, mySubComm)

            #use cached data to final values
            scaleVals = self._scaleExp( evalSubTree.final_view(scaleCache) )
            Gs  = evalSubTree.final_view( prodCache, axis=0)
              # ( nGateStrings, dim, dim )

            def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                tm = _time.time()
                old_err = _np.seterr(over='ignore')
                rho,E = self._rhoE_from_spamLabel(spamLabel)
                if sumInto:
                    mxToFill[isp,fslc] += self._probs_from_rhoE(spamLabel, rho,
                                                              E, Gs, scaleVals)
                else:
                    mxToFill[isp,fslc] =  self._probs_from_rhoE(spamLabel, rho,
                                                              E, Gs, scaleVals)
                _np.seterr(**old_err)

            self._fill_result_tuple( (mxToFill,), spam_label_rows,
                                     fslc, slice(None), slice(None), calc_and_fill )

        #collect/gather results
        subtreeFinalSlices = [ t.final_slice(evalTree) for t in subtrees]
        _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, mxToFill,
                            1, comm) 
        #note: pass mxToFill, dim=(K,S), so gather mxToFill[:,fslc] (axis=1)

        if clipTo is not None:
            _np.clip( mxToFill, clipTo[0], clipTo[1], out=mxToFill ) # in-place clip

        if check:
            self._check(evalTree, spam_label_rows, mxToFill, clipTo=clipTo)


    def bulk_fill_dprobs(self, mxToFill, spam_label_rows, evalTree,
                         prMxToFill=None,clipTo=None,check=False,
                         comm=None, wrtFilter=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):

        """
        Identical to bulk_dprobs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability derivatives for all gate
        strings and a given SPAM label are placed into
        mxToFill[ spam_label_rows[spamLabel] ].
        Optionally, probabilities can be placed into
        prMxToFill[ spam_label_rows[spamLabel] ]

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        clipTo : 2-tuple, optional
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

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

        tStart = _time.time()
        if profiler is None: profiler = _dummy_profiler
        
        remainder_row_index = None
        for spamLabel,rowIndex in spam_label_rows.items():
            if self._is_remainder_spamlabel(spamLabel):
                assert(self.assumeSumToOne) # ensure the remainder label is allowed
                assert(remainder_row_index is None) # ensure there is at most one dummy spam label
                remainder_row_index = rowIndex

        if wrtFilter is not None:
            assert(wrtBlockSize is None) #Cannot specify both wrtFilter and wrtBlockSize
            tot_rho = self.tot_rho_params
            tot_spam = self.tot_rho_params + self.tot_e_params
            wrtSlices = {
                'preps':    [ x for x in wrtFilter if x < tot_rho ],
                'effects' : [ (x-tot_rho) for x in wrtFilter if tot_rho <= x < tot_spam ],
                'gates' :   [ (x-tot_spam) for x in wrtFilter if x >= tot_spam ] }
        
            wrtSlices['preps'] = _slct.list_to_slice(wrtSlices['preps'])
            wrtSlices['effects'] = _slct.list_to_slice(wrtSlices['effects'])
            wrtSlices['gates'] = _slct.list_to_slice(wrtSlices['gates'])
        else:
            wrtSlices = None

        profiler.mem_check("bulk_fill_dprobs: begin (expect ~ %.2fGB)" 
                           % (mxToFill.nbytes/(1024.0**3)) )

        ## memory profiling of python objects (never seemed very useful
        ##  since numpy does all the major allocation/deallocation).
        #if comm is None or comm.Get_rank() == 0:
        #    import objgraph 
        #    objgraph.show_growth(limit=50) 

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)
        #if comm is not None: 
        #    print("MPI DEBUG: Rank%d subtee sizes = %s" % 
        #          (comm.Get_rank(),",".join([str(len(subtrees[i]))
        #                                     for i in mySubTreeIndices])))

        #eval on each local subtree
        #my_results = []
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            fslc = evalSubTree.final_slice(evalTree)

            #Free memory from previous subtree iteration before computing caches
            scaleVals = Gs = dGs = None
            prodCache = scaleCache = dProdCache = None

            #Fill cache info (not requiring column distribution)
            tm = _time.time()
            prodCache, scaleCache = self._compute_product_cache(evalSubTree, mySubComm)
            profiler.add_time("bulk_fill_dprobs: compute_product_cache", tm)

            #use cached data to final values
            scaleVals = self._scaleExp( evalSubTree.final_view( scaleCache ))
            Gs  = evalSubTree.final_view( prodCache, axis=0 )
              #( nGateStrings, dim, dim )
            profiler.mem_check("bulk_fill_dprobs: post compute product")

            def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                tm = _time.time()
                old_err = _np.seterr(over='ignore')
                rho,E = self._rhoE_from_spamLabel(spamLabel)
                
                if sumInto:
                    if prMxToFill is not None:
                        prMxToFill[isp,fslc] += \
                            self._probs_from_rhoE(spamLabel, rho, E, Gs, scaleVals)
                    mxToFill[isp,fslc,pslc1] += self._dprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs, scaleVals, wrtSlices)
                else:
                    if prMxToFill is not None:
                        prMxToFill[isp,fslc] = \
                            self._probs_from_rhoE(spamLabel, rho, E, Gs, scaleVals)
                    mxToFill[isp,fslc,pslc1] = self._dprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs, scaleVals, wrtSlices)

                _np.seterr(**old_err)
                profiler.add_time("bulk_fill_dprobs: calc_and_fill", tm)

            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter is None:
                blkSize = wrtBlockSize #could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.tot_gate_params / mySubComm.Get_size()
                    blkSize = comm_blkSize if (blkSize is None) \
                        else min(comm_blkSize, blkSize) #override with smaller comm_blkSize
            else:
                blkSize = None # wrtFilter dictates block


            if blkSize is None:
                #Fill derivative cache info
                tm = _time.time()
                gatesSlice = wrtSlices['gates'] if (wrtSlices is not None) else None
                dProdCache = self._compute_dproduct_cache(evalSubTree, prodCache, scaleCache,
                                                          mySubComm, gatesSlice, profiler)
                dGs = evalSubTree.final_view(dProdCache, axis=0)
                  #( nGateStrings, nDerivCols, dim, dim )
                profiler.add_time("bulk_fill_dprobs: compute_dproduct_cache", tm)
                profiler.mem_check("bulk_fill_dprobs: post compute dproduct")

                #Compute all requested derivative columns at once
                self._fill_result_tuple( (prMxToFill, mxToFill), spam_label_rows,
                                         fslc, slice(None), slice(None), calc_and_fill )
                profiler.mem_check("bulk_fill_dprobs: post fill")
                dProdCache = dGs = None #free mem

            else: # Divide columns into blocks of at most blkSize
                assert(wrtFilter is None) #cannot specify both wrtFilter and blkSize
                nBlks = int(_np.ceil(self.tot_gate_params / blkSize))
                  # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(self.tot_gate_params, nBlks,
                                              start=self.tot_spam_params)

                # Create placeholder dGs for *no* gate params to compute
                #  derivatives wrt all spam parameters
                dGs = _np.empty( (Gs.shape[0],0,self.dim,self.dim), 'd')

                #Compute spam derivative columns and possibly probs
                # (computation that is *not* divided into blocks)
                self._fill_result_tuple( 
                    (prMxToFill, mxToFill), spam_label_rows, fslc,
                    slice(0,self.tot_spam_params), slice(None), calc_and_fill )
                profiler.mem_check("bulk_fill_dprobs: post fill spam")

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                       +" than derivative columns(%d)!" % self.tot_gate_params 
                       +" [blkSize = %.1f, nBlks=%d]" % (blkSize,nBlks))

                def calc_and_fill_blk(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                    tm = _time.time()
                    old_err = _np.seterr(over='ignore')
                    rho,E = self._rhoE_from_spamLabel(spamLabel)
                    wrtNoSpam = {'preps':slice(0,0),'effects':slice(0,0)}
                    
                    if sumInto:
                        mxToFill[isp,fslc,pslc1] += self._dprobs_from_rhoE(
                            spamLabel, rho, E, Gs, dGs, scaleVals, wrtNoSpam)
                            
                    else:
                        mxToFill[isp,fslc,pslc1] = self._dprobs_from_rhoE(
                            spamLabel, rho, E, Gs, dGs, scaleVals, wrtNoSpam)
                    _np.seterr(**old_err)
                    profiler.add_time("bulk_fill_dprobs: calc_and_fill_blk", tm)

                for iBlk in myBlkIndices:
                    tm = _time.time()
                    gateSlice = _slct.shift(blocks[iBlk],-self.tot_spam_params)
                    dProdCache = self._compute_dproduct_cache(evalSubTree, prodCache, scaleCache,
                                                              blkComm, gateSlice, profiler)
                    profiler.add_time("bulk_fill_dprobs: compute_dproduct_cache", tm)
                    profiler.mem_check(
                        "bulk_fill_dprobs: post compute dproduct blk (expect "+
                        " +%.2fGB, shape=%s)" % (dProdCache.nbytes/(1024.0**3),
                                                 str(dProdCache.shape)) )

                    dGs = evalSubTree.final_view(dProdCache, axis=0)
                      #( nGateStrings, nDerivCols, dim, dim )
                    self._fill_result_tuple( 
                        (mxToFill,), spam_label_rows, fslc, 
                        blocks[iBlk], slice(None), calc_and_fill_blk )                    

                    profiler.mem_check("bulk_fill_dprobs: post fill blk")
                    dProdCache = dGs = None #free mem

                #gather results
                tm = _time.time()
                _mpit.gather_slices(blocks, blkOwners, mxToFill[:,fslc],
                                    2, mySubComm, gatherMemLimit)
                #note: gathering axis 2 of mxToFill[:,fslc], dim=(K,s,M)
                profiler.add_time("MPI IPC", tm)
                profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        tm = _time.time()
        subtreeFinalSlices = [ t.final_slice(evalTree) for t in subtrees]
        _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, mxToFill,
                            1, comm, gatherMemLimit) 
        #note: pass mxToFill, dim=(K,S,M), so gather mxToFill[:,fslc] (axis=1)

        if prMxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, prMxToFill,
                                1, comm) 
        #note: pass prMxToFill, dim=(K,S), so gather prMxToFill[:,fslc] (axis=1)

        profiler.add_time("MPI IPC", tm)
        profiler.mem_check("bulk_fill_dprobs: post gather subtrees")

        if clipTo is not None and prMxToFill is not None:
            _np.clip( prMxToFill, clipTo[0], clipTo[1], out=prMxToFill ) # in-place clip

        if check:
            self._check(evalTree, spam_label_rows, prMxToFill, mxToFill,
                        clipTo=clipTo)
        profiler.add_time("bulk_fill_dprobs: total", tStart)
        profiler.add_count("bulk_fill_dprobs count")
        profiler.mem_check("bulk_fill_dprobs: end")



    def bulk_fill_hprobs(self, mxToFill, spam_label_rows, evalTree,
                         prMxToFill=None, deriv1MxToFill=None, deriv2MxToFill=None, 
                         clipTo=None, check=False,comm=None, wrtFilter1=None, wrtFilter2=None,
                         wrtBlockSize1=None, wrtBlockSize2=None, gatherMemLimit=None):

        """
        Identical to bulk_hprobs(...) except results are
        placed into rows of a pre-allocated array instead
        of being returned in a dictionary.

        Specifically, the probability hessians for all gate
        strings and a given SPAM label are placed into
        mxToFill[ spam_label_rows[spamLabel] ].
        Optionally, probabilities and/or derivatives can be placed into
        prMxToFill[ spam_label_rows[spamLabel] ] and
        derivMxToFill[ spam_label_rows[spamLabel] ] respectively.

        Parameters
        ----------
        mxToFill : numpy array
          an already-allocated KxSxMxM numpy array, where K is larger
          than the maximum value in spam_label_rows, S is equal
          to the number of gate strings (i.e. evalTree.num_final_strings()),
          and M is the length of the vectorized gateset.

        spam_label_rows : dictionary
          a dictionary with keys == spam labels and values which
          are integer row indices into mxToFill, specifying the
          correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated KxS numpy array that is filled
          with the probabilities as per spam_label_rows, similar to
          bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated KxSxM numpy array that is filled
          with the probability derivatives as per spam_label_rows, similar to
          bulk_fill_dprobs(...), but where M is the number of gateset parameters
          selected for the 1st and 2nd differentiation, respectively (i.e. by
          wrtFilter1 and wrtFilter2).

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

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

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.


        Returns
        -------
        None
        """
        remainder_row_index = None
        for spamLabel,rowIndex in spam_label_rows.items():
            if self._is_remainder_spamlabel(spamLabel):
                assert(self.assumeSumToOne) # ensure the remainder label is allowed
                assert(remainder_row_index is None) # ensure there is at most one dummy spam label
                remainder_row_index = rowIndex

        if wrtFilter1 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None) #Cannot specify both wrtFilter and wrtBlockSize
            tot_rho = self.tot_rho_params
            tot_spam = self.tot_rho_params + self.tot_e_params
            wrtSlices1 = {
                'preps':    [ x for x in wrtFilter1 if x < tot_rho ],
                'effects' : [ (x-tot_rho) for x in wrtFilter1 if tot_rho <= x < tot_spam ],
                'gates' :   [ (x-tot_spam) for x in wrtFilter1 if x >= tot_spam ] }

            wrtSlices1['preps'] = _slct.list_to_slice(wrtSlices1['preps'])
            wrtSlices1['effects'] = _slct.list_to_slice(wrtSlices1['effects'])
            wrtSlices1['gates'] = _slct.list_to_slice(wrtSlices1['gates'])
        else:
            wrtSlices1 = None


        if wrtFilter2 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None) #Cannot specify both wrtFilter and wrtBlockSize
            tot_rho = self.tot_rho_params
            tot_spam = self.tot_rho_params + self.tot_e_params
            wrtSlices2 = {
                'preps':    [ x for x in wrtFilter2 if x < tot_rho ],
                'effects' : [ (x-tot_rho) for x in wrtFilter2 if tot_rho <= x < tot_spam ],
                'gates' :   [ (x-tot_spam) for x in wrtFilter2 if x >= tot_spam ] }

            wrtSlices2['preps'] = _slct.list_to_slice(wrtSlices2['preps'])
            wrtSlices2['effects'] = _slct.list_to_slice(wrtSlices2['effects'])
            wrtSlices2['gates'] = _slct.list_to_slice(wrtSlices2['gates'])
        else:
            wrtSlices2 = None

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        my_results = []
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            fslc = evalSubTree.final_slice(evalTree)

            #Free memory from previous subtree iteration before computing caches
            scaleVals = Gs = dGs1 = dGs2 = hGs = None
            prodCache = scaleCache = dProdCache = None

            #Fill product cache info (not requiring row or column distribution)
            prodCache, scaleCache = self._compute_product_cache(evalSubTree, mySubComm)
            scaleVals = self._scaleExp( evalSubTree.final_view(scaleCache))
            Gs  = evalSubTree.final_view(prodCache, axis=0)
              #( nGateStrings, dim, dim )

            def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
                tm = _time.time()
                old_err = _np.seterr(over='ignore')
                rho,E = self._rhoE_from_spamLabel(spamLabel)
                
                if sumInto:
                    if prMxToFill is not None:
                        prMxToFill[isp,fslc] += \
                            self._probs_from_rhoE(spamLabel, rho, E, Gs, scaleVals)
                    if deriv1MxToFill is not None:
                        deriv1MxToFill[isp,fslc,pslc1] += self._dprobs_from_rhoE( 
                            spamLabel, rho, E, Gs, dGs1, scaleVals, wrtSlices1)
                    if deriv2MxToFill is not None:
                        deriv2MxToFill[isp,fslc,pslc2] += self._dprobs_from_rhoE( 
                            spamLabel, rho, E, Gs, dGs2, scaleVals, wrtSlices2)

                    mxToFill[isp,fslc,pslc1,pslc2] += self._hprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs1, dGs2, hGs, scaleVals, wrtSlices1, wrtSlices2)

                else:
                    if prMxToFill is not None:
                        prMxToFill[isp,fslc] = \
                            self._probs_from_rhoE(spamLabel, rho, E, Gs, scaleVals)
                    if deriv1MxToFill is not None:
                        deriv1MxToFill[isp,fslc,pslc1] = self._dprobs_from_rhoE( 
                            spamLabel, rho, E, Gs, dGs1, scaleVals, wrtSlices1)
                    if deriv2MxToFill is not None:
                        deriv2MxToFill[isp,fslc,pslc2] = self._dprobs_from_rhoE( 
                            spamLabel, rho, E, Gs, dGs2, scaleVals, wrtSlices2)

                    mxToFill[isp,fslc,pslc1,pslc2] = self._hprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs1, dGs2, hGs, scaleVals, wrtSlices1, wrtSlices2)

                _np.seterr(**old_err)

            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter1 is None and wrtFilter2 is None:
                blkSize1 = wrtBlockSize1 #could be None
                blkSize2 = wrtBlockSize2 #could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.tot_gate_params / mySubComm.Get_size()
                    blkSize1 = comm_blkSize if (blkSize1 is None) \
                        else min(comm_blkSize, blkSize1) #override with smaller comm_blkSize
                    blkSize2 = comm_blkSize if (blkSize2 is None) \
                        else min(comm_blkSize, blkSize2) #override with smaller comm_blkSize
            else:
                blkSize1 = blkSize2 = None # wrtFilter1 & wrtFilter2 dictates block


            if blkSize1 is None and blkSize2 is None:
                #Fill hessian cache info
                gatesSlice1 = wrtSlices1['gates'] if (wrtSlices1 is not None) else None
                gatesSlice2 = wrtSlices2['gates'] if (wrtSlices2 is not None) else None

                dProdCache1 = self._compute_dproduct_cache(
                    evalSubTree, prodCache, scaleCache, mySubComm, gatesSlice1)
                dProdCache2 = dProdCache1 if (gatesSlice1 == gatesSlice2) else \
                    self._compute_dproduct_cache(evalSubTree, prodCache,
                                                 scaleCache, mySubComm, gatesSlice2)

                dGs1 = evalSubTree.final_view(dProdCache1, axis=0) 
                dGs2 = evalSubTree.final_view(dProdCache2, axis=0) 
                  #( nGateStrings, nGateDerivColsX, dim, dim )

                hProdCache = self._compute_hproduct_cache(evalSubTree, prodCache, dProdCache1,
                                                          dProdCache2, scaleCache, mySubComm,
                                                          gatesSlice1, gatesSlice2)
                hGs = evalSubTree.final_view(hProdCache, axis=0)
                   #( nGateStrings, len(wrtFilter1), len(wrtFilter2), dim, dim )

                #Compute all requested derivative columns at once
                self._fill_result_tuple((prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                                        spam_label_rows, fslc, slice(None),
                                        slice(None), calc_and_fill)

            else: # Divide columns into blocks of at most blkSize
                assert(wrtFilter1 is None and wrtFilter2 is None) #cannot specify both wrtFilter and blkSize
                nBlks1 = int(_np.ceil(self.tot_params / blkSize1))
                nBlks2 = int(_np.ceil(self.tot_params / blkSize2))
                  # num blocks required to achieve desired average size == blkSize1 or blkSize2
                blocks1 = _mpit.slice_up_range(self.tot_params, nBlks1)
                blocks2 = _mpit.slice_up_range(self.tot_params, nBlks2)

                #distribute derivative computation across blocks
                myBlk1Indices, blk1Owners, blk1Comm = \
                    _mpit.distribute_indices(list(range(nBlks1)), mySubComm)

                myBlk2Indices, blk2Owners, blk2Comm = \
                    _mpit.distribute_indices(list(range(nBlks2)), blk1Comm)

                if blk2Comm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                       +" than hessian elements(%d)!" % (self.tot_gate_params**2)
                       +" [blkSize = {%.1f,%.1f}, nBlks={%d,%d}]" % (blkSize1,blkSize2,nBlks1,nBlks2))

                for iBlk1 in myBlk1Indices:
                    prepSlice1 = _slct.intersect(blocks1[iBlk1],slice(0,self.tot_rho_params))
                    effectSlice1 = _slct.shift( _slct.intersect(blocks1[iBlk1],slice(self.tot_rho_params,self.tot_spam_params)), -self.tot_rho_params)
                    gateSlice1 = _slct.shift( _slct.intersect(blocks1[iBlk1],slice(self.tot_spam_params,None)), -self.tot_spam_params)
                    dProdCache1 = self._compute_dproduct_cache(
                        evalSubTree, prodCache, scaleCache, blk1Comm, gateSlice1)
                    dGs1 = evalSubTree.final_view(dProdCache1, axis=0) 

                    for iBlk2 in myBlk2Indices:
                        prepSlice2 = _slct.intersect(blocks2[iBlk2],slice(0,self.tot_rho_params))
                        effectSlice2 = _slct.shift( _slct.intersect(blocks2[iBlk2],slice(self.tot_rho_params,self.tot_spam_params)), -self.tot_rho_params)
                        gateSlice2 = _slct.shift( _slct.intersect(blocks2[iBlk2],slice(self.tot_spam_params,None)), -self.tot_spam_params)

                        if (gateSlice1 == gateSlice2):
                            dProdCache2 = dProdCache1 ; dGs2 = dGs1
                        else:
                            dProdCache2 =self._compute_dproduct_cache(
                                evalSubTree, prodCache, scaleCache, blk2Comm, gateSlice2)
                            dGs2 = evalSubTree.final_view(dProdCache2, axis=0) 
                        rank = comm.Get_rank()

                        hProdCache = self._compute_hproduct_cache(
                            evalSubTree, prodCache, dProdCache1, dProdCache2,
                            scaleCache, blk2Comm, gateSlice1, gateSlice2)
                        hGs = evalSubTree.final_view(hProdCache, axis=0)

                        #Set spam filtering for calc_and_fill
                        wrtSlices1 = {'preps': prepSlice1, 'effects': effectSlice1 }
                        wrtSlices2 = {'preps': prepSlice2, 'effects': effectSlice2 }

                        self._fill_result_tuple((prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                                                spam_label_rows, fslc, blocks1[iBlk1], blocks2[iBlk2],
                                                calc_and_fill)
    
                        hProdCache = hGs = dProdCache2 = dGs2 =  None # free mem
                    dProdCache1 = dGs1 = None #free mem

                    #gather column results: gather axis 3 of mxToFill[:,fslc,blocks1[iBlk1]], dim=(K,s,blk1,M)
                    _mpit.gather_slices(blocks2, blk2Owners, mxToFill[:,fslc,blocks1[iBlk1]],
                                        3, blk1Comm, gatherMemLimit)

                #gather row results; gather axis 2 of mxToFill[:,fslc], dim=(K,s,M,M)
                _mpit.gather_slices(blocks1, blk1Owners, mxToFill[:,fslc],
                                    2, mySubComm, gatherMemLimit)
                if deriv1MxToFill is not None:
                    _mpit.gather_slices(blocks1, blk1Owners, deriv1MxToFill[:,fslc],
                                        2, mySubComm, gatherMemLimit)
                if deriv2MxToFill is not None:
                    _mpit.gather_slices(blocks2, blk2Owners, deriv2MxToFill[:,fslc],
                                        2, blk1Comm, gatherMemLimit) 
                   #Note: deriv2MxToFill gets computed on every inner loop completion
                   # (to save mem) but isn't gathered until now (but using blk1Comm).
                   # (just as prMxToFill is computed fully on each inner loop *iteration*!)
            
        #collect/gather results
        subtreeFinalSlices = [ t.final_slice(evalTree) for t in subtrees]
        _mpit.gather_slices(subtreeFinalSlices, subTreeOwners, 
                            mxToFill, 1, comm, gatherMemLimit) 
        if deriv1MxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners,
                                deriv1MxToFill, 1, comm, gatherMemLimit) 
        if deriv2MxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners,
                                deriv2MxToFill, 1, comm, gatherMemLimit) 
        if prMxToFill is not None:
            _mpit.gather_slices(subtreeFinalSlices, subTreeOwners,
                                prMxToFill, 1, comm) 


        if clipTo is not None and prMxToFill is not None:
            _np.clip( prMxToFill, clipTo[0], clipTo[1], out=prMxToFill ) # in-place clip

        if check:
            self._check(evalTree, spam_label_rows,
                        prMxToFill, deriv1MxToFill, mxToFill, clipTo)



    def bulk_pr(self, spamLabel, evalTree, clipTo=None,
                check=False, comm=None):
        """
        Compute the probabilities of the gate sequences given by evalTree,
        where initialization & measurement operations are always the same
        and are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
        numpy array
          An array of length equal to the number of gate strings containing
          the (float) probabilities.
        """
        vp = _np.empty( (1,evalTree.num_final_strings()), 'd' )
        self.bulk_fill_probs(vp, { spamLabel: 0}, evalTree,
                             clipTo, check, comm)
        return vp[0]


    def bulk_probs(self, evalTree, clipTo=None, check=False, comm=None):
        """
        Construct a dictionary containing the bulk-probabilities
        for every spam label (each possible initialization &
        measurement pair) for each gate sequence given by
        evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
            A dictionary such that
            probs[SL] = bulk_pr(SL,evalTree,clipTo,check)
            for each spam label (string) SL.
        """
        spam_label_rows = \
            { spamLabel: i for (i,spamLabel) in enumerate(self.spamdefs) }
        vp = _np.empty((len(self.spamdefs),evalTree.num_final_strings()),'d')
        self.bulk_fill_probs(vp, spam_label_rows, evalTree,
                             clipTo, check, comm)
        return { spamLabel: vp[i] \
                     for (i,spamLabel) in enumerate(self.spamdefs) }


    def bulk_dpr(self, spamLabel, evalTree,
                 returnPr=False,clipTo=None,check=False,
                 comm=None, wrtFilter=None, wrtBlockSize=None):
        """
        Compute the derivatives of the probabilities generated by a each gate
        sequence given by evalTree, where initialization
        & measurement operations are always the same and are
        together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
           the label specifying the state prep and measure operations

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
        dprobs : numpy array
            An array of shape S x M, where

            - S == the number of gate strings
            - M == the length of the vectorized gateset

            and dprobs[i,j] holds the derivative of the i-th probability w.r.t.
            the j-th gateset parameter.

        probs : numpy array
            Only returned when returnPr == True. An array of shape S containing
            the probabilities of each gate string.
        """
        nGateStrings = evalTree.num_final_strings()
        nDerivCols = self.tot_params

        vdp = _np.empty( (1,nGateStrings,nDerivCols), 'd' )
        vp = _np.empty( (1,nGateStrings), 'd' ) if returnPr else None

        self.bulk_fill_dprobs(vdp, {spamLabel: 0}, evalTree,
                              vp, clipTo, check, comm,
                              wrtFilter, wrtBlockSize)
        return (vdp[0], vp[0]) if returnPr else vdp[0]


    def bulk_dprobs(self, evalTree,
                    returnPr=False,clipTo=None,
                    check=False,comm=None,
                    wrtFilter=None, wrtBlockSize=None):

        """
        Construct a dictionary containing the bulk-probability-
        derivatives for every spam label (each possible
        initialization & measurement pair) for each gate
        sequence given by evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
            A dictionary such that
            ``dprobs[SL] = bulk_dpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,clipTo,check)``
            for each spam label (string) SL.
        """
        spam_label_rows = \
            { spamLabel: i for (i,spamLabel) in enumerate(self.spamdefs) }
        nGateStrings = evalTree.num_final_strings()
        nDerivCols = self.tot_params
        nSpamLabels = len(self.spamdefs)

        vdp = _np.empty( (nSpamLabels,nGateStrings,nDerivCols), 'd' )
        vp = _np.empty( (nSpamLabels,nGateStrings), 'd' ) if returnPr else None

        self.bulk_fill_dprobs(vdp, spam_label_rows, evalTree,
                              vp, clipTo, check, comm,
                              wrtFilter, wrtBlockSize)

        if returnPr:
            return { spamLabel: (vdp[i],vp[i]) \
                     for (i,spamLabel) in enumerate(self.spamdefs) }
        else:
            return { spamLabel: vdp[i] \
                     for (i,spamLabel) in enumerate(self.spamdefs) }





    def bulk_hpr(self, spamLabel, evalTree,
                 returnPr=False,returnDeriv=False,
                 clipTo=None,check=False,comm=None,
                 wrtFilter1=None, wrtFilter2=None,
                 wrtBlockSize1=None, wrtBlockSize2=None):

        """
        Compute the 2nd derivatives of the probabilities generated by a each gate
        sequence given by evalTree, where initialization & measurement
        operations are always the same and are together specified by spamLabel.

        Parameters
        ----------
        spamLabel : string
          the label specifying the state prep and measure operations

        evalTree : EvalTree
          given by a prior call to bulk_evaltree.  Specifies the gate strings
          to compute the bulk operation on.

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
        hessians : numpy array
            a S x M x M array, where

            - S == the number of gate strings
            - M == the length of the vectorized gateset

            and hessians[i,j,k] is the derivative of the i-th probability
            w.r.t. the k-th then the j-th gateset parameter.

        derivs1, derivs2 : numpy array
            only returned if returnDeriv == True. Two S x M arrays where
            derivsX[i,j] holds the derivative of the i-th probability
            w.r.t. the j-th gateset parameter, where j is taken from the
            first and second sets of filtered parameters (i.e. by
            wrtFilter1 and wrtFilter2).  If `wrtFilter1 == wrtFilter2`,
            then derivs2 is not returned (to save memory, since it's the
            same as derivs1).

        probabilities : numpy array
            only returned if returnPr == True.  A length-S array
            containing the probabilities for each gate string.
        """
        nGateStrings = evalTree.num_final_strings()
        nDerivCols1 = self.tot_params if (wrtFilter1 is None) \
                           else len(wrtFilter1)
        nDerivCols2 = self.tot_params if (wrtFilter2 is None) \
                           else len(wrtFilter2)

        vhp = _np.empty( (1,nGateStrings,nDerivCols1,nDerivCols2), 'd' )
        vdp1 = _np.empty( (1,nGateStrings,self.tot_params), 'd' ) \
            if returnDeriv else None
        vdp2 = vdp1.copy() if (returnDeriv and wrtFilter1!=wrtFilter2) else None
        vp = _np.empty( (1,nGateStrings), 'd' ) if returnPr else None

        self.bulk_fill_hprobs(vhp, {spamLabel: 0}, evalTree,
                              vp, vdp1, vdp2, clipTo, check, comm,
                              wrtFilter1,wrtFilter2,wrtBlockSize1,wrtBlockSize2)
        if returnDeriv:
            if vdp2 is None:
                return (vhp[0], vdp1[0], vp[0]) if returnPr else (vhp[0],vdp1[0])
            else:
                return (vhp[0], vdp1[0], vdp2[0], vp[0]) if returnPr else (vhp[0],vdp1[0],vdp2[0])
        else:
            return (vhp[0], vp[0]) if returnPr else vhp[0]



    def bulk_hprobs(self, evalTree,
                    returnPr=False,returnDeriv=False,clipTo=None,
                    check=False,comm=None,
                    wrtFilter1=None, wrtFilter2=None,
                    wrtBlockSize1=None, wrtBlockSize2=None):

        """
        Construct a dictionary containing the bulk-probability-
        Hessians for every spam label (each possible
        initialization & measurement pair) for each gate
        sequence given by evalTree.

        Parameters
        ----------
        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the gate strings
           to compute the bulk operation on.

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
        dict
            A dictionary such that
            ``hprobs[SL] = bulk_hpr(SL,evalTree,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo,check)``
            for each spam label (string) SL.
        """
        spam_label_rows = \
            { spamLabel: i for (i,spamLabel) in enumerate(self.spamdefs) }
        nGateStrings = evalTree.num_final_strings()
        nDerivCols1 = self.tot_params if (wrtFilter1 is None) \
                           else len(wrtFilter1)
        nDerivCols2 = self.tot_params if (wrtFilter2 is None) \
                           else len(wrtFilter2)
        nSpamLabels = len(self.spamdefs)

        vhp = _np.empty( (nSpamLabels,nGateStrings,nDerivCols1,nDerivCols2),'d')
        vdp1 = _np.empty( (nSpamLabels,nGateStrings,self.tot_params), 'd' ) \
            if returnDeriv else None
        vdp2 = vdp1.copy() if (returnDeriv and wrtFilter1!=wrtFilter2) else None
        vp = _np.empty( (nSpamLabels,nGateStrings), 'd' ) if returnPr else None

        self.bulk_fill_hprobs(vhp, spam_label_rows, evalTree,
                              vp, vdp1, vdp2, clipTo, check, comm,
                              wrtFilter1,wrtFilter1,wrtBlockSize1,wrtBlockSize2)
        if returnDeriv:
            if vdp2 is None:
                if returnPr:
                    return { spamLabel: (vhp[i],vdp1[i],vp[i]) \
                             for (i,spamLabel) in enumerate(self.spamdefs) }
                else:
                    return { spamLabel: (vhp[i],vdp1[i]) \
                             for (i,spamLabel) in enumerate(self.spamdefs) }
            else:
                if returnPr:
                    return { spamLabel: (vhp[i],vdp1[i],vdp2[i],vp[i]) \
                             for (i,spamLabel) in enumerate(self.spamdefs) }
                else:
                    return { spamLabel: (vhp[i],vdp1[i],vdp2[i]) \
                             for (i,spamLabel) in enumerate(self.spamdefs) }
        else:
            if returnPr:
                return { spamLabel: (vhp[i],vp[i]) \
                         for (i,spamLabel) in enumerate(self.spamdefs) }
            else:
                return { spamLabel: vhp[i] \
                         for (i,spamLabel) in enumerate(self.spamdefs) }


    def bulk_hprobs_by_block(self, spam_label_rows, evalTree, wrtSlicesList,
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
        assert(not evalTree.is_split())
        nGateStrings = evalTree.num_final_strings()

        #Fill product cache info (not distributed)
        prodCache, scaleCache = self._compute_product_cache(evalTree, comm)
        scaleVals = self._scaleExp( evalTree.final_view(scaleCache))
        Gs  = evalTree.final_view(prodCache, axis=0)
          #( nGateStrings, dim, dim )

        #Same as in bulk_fill_hprobs (TODO consolidate?)
        def calc_and_fill(spamLabel, isp, fslc, pslc1, pslc2, sumInto):
            tm = _time.time()
            old_err = _np.seterr(over='ignore')
            rho,E = self._rhoE_from_spamLabel(spamLabel)
            
            if sumInto:
                if prMxToFill is not None:
                    prMxToFill[isp,fslc] += \
                        self._probs_from_rhoE(spamLabel, rho, E, Gs, scaleVals)
                if deriv1MxToFill is not None:
                    deriv1MxToFill[isp,fslc,pslc1] += self._dprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs1, scaleVals, wrtSPAMSlices1)
                if deriv2MxToFill is not None:
                    deriv2MxToFill[isp,fslc,pslc2] += self._dprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs2, scaleVals, wrtSPAMSlices2)

                mxToFill[isp,fslc,pslc1,pslc2] += self._hprobs_from_rhoE( 
                    spamLabel, rho, E, Gs, dGs1, dGs2, hGs, scaleVals, wrtSPAMSlices1, wrtSPAMSlices2)

            else:
                if prMxToFill is not None:
                    prMxToFill[isp,fslc] = \
                        self._probs_from_rhoE(spamLabel, rho, E, Gs, scaleVals)
                if deriv1MxToFill is not None:
                    deriv1MxToFill[isp,fslc,pslc1] = self._dprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs1, scaleVals, wrtSPAMSlices1)
                if deriv2MxToFill is not None:
                    deriv2MxToFill[isp,fslc,pslc2] = self._dprobs_from_rhoE( 
                        spamLabel, rho, E, Gs, dGs2, scaleVals, wrtSPAMSlices2)

                mxToFill[isp,fslc,pslc1,pslc2] = self._hprobs_from_rhoE( 
                    spamLabel, rho, E, Gs, dGs1, dGs2, hGs, scaleVals, wrtSPAMSlices1, wrtSPAMSlices2)

            _np.seterr(**old_err)


        #NOTE: don't distribute wrtSlicesList across comm procs,
        # as we assume the user has already done any such distribution
        # and has given each processor a list appropriate for it.
        # Use comm only for speeding up the calcs of the given 
        # wrtSlicesList

        last_gateSlice1 = None #keep last dProdCache1

        for wrtSlice1,wrtSlice2 in wrtSlicesList:
            
            prepSlice1 = _slct.intersect(wrtSlice1,slice(0,self.tot_rho_params))
            effectSlice1 = _slct.shift( _slct.intersect(wrtSlice1,slice(self.tot_rho_params,self.tot_spam_params)), -self.tot_rho_params)
            gateSlice1 = _slct.shift( _slct.intersect(wrtSlice1,slice(self.tot_spam_params,None)), -self.tot_spam_params)
            
            if gateSlice1 != last_gateSlice1:
                dProdCache1 = dGs1 = None #free Mem
                dProdCache1 = self._compute_dproduct_cache(
                    evalTree, prodCache, scaleCache, comm, gateSlice1)
                dGs1 = evalTree.final_view(dProdCache1, axis=0) 
                last_gateSlice1 = gateSlice1
    
            prepSlice2 = _slct.intersect(wrtSlice2,slice(0,self.tot_rho_params))
            effectSlice2 = _slct.shift( _slct.intersect(wrtSlice2,slice(self.tot_rho_params,self.tot_spam_params)), -self.tot_rho_params)
            gateSlice2 = _slct.shift( _slct.intersect(wrtSlice2,slice(self.tot_spam_params,None)), -self.tot_spam_params)
        
            if (gateSlice1 == gateSlice2):
                dProdCache2 = dProdCache1 ; dGs2 = dGs1
            else:
                dProdCache2 =self._compute_dproduct_cache(
                    evalTree, prodCache, scaleCache, comm, gateSlice2)
                dGs2 = evalTree.final_view(dProdCache2, axis=0) 
            
            hProdCache = self._compute_hproduct_cache(
                evalTree, prodCache, dProdCache1, dProdCache2,
                scaleCache, comm, gateSlice1, gateSlice2)
            hGs = evalTree.final_view(hProdCache, axis=0)
                
            if bReturnDProbs12:
                dprobs1 = _np.zeros( (len(spam_label_rows),nGateStrings,_slct.length(wrtSlice1)), 'd' )
                dprobs2 = _np.zeros( (len(spam_label_rows),nGateStrings,_slct.length(wrtSlice2)), 'd' )
            else:
                dprobs1 = dprobs2 = None            
            hprobs = _np.zeros( (len(spam_label_rows),nGateStrings,
                                 _slct.length(wrtSlice1),_slct.length(wrtSlice2)), 'd' )

            #Set spam filtering and params for calc_and_fill
            wrtSPAMSlices1 = {'preps': prepSlice1, 'effects': effectSlice1 }
            wrtSPAMSlices2 = {'preps': prepSlice2, 'effects': effectSlice2 }
            prMxToFill = None
            deriv1MxToFill = dprobs1
            deriv2MxToFill = dprobs2
            mxToFill = hprobs

            #Fill arrays
            self._fill_result_tuple((None, dprobs1, dprobs2, hprobs), spam_label_rows,
                                    slice(None), slice(None), slice(None), calc_and_fill)
            hProdCache = hGs = dProdCache2 = dGs2 =  None # free mem
            if bReturnDProbs12:
                dprobs12 = dprobs1[:,:,:,None] * dprobs2[:,:,None,:] # (K,M,N,1) * (K,M,1,N') = (K,M,N,N')
                yield wrtSlice1, wrtSlice2, hprobs, dprobs12
            else:
                yield wrtSlice1, wrtSlice2, hprobs

        dProdCache1 = dGs1 = None #free mem
                    

    def frobeniusdist(self, otherCalc, transformMx=None,
                      gateWeight=1.0, spamWeight=1.0, itemWeights=None,
                      normalize=True):
        """
        Compute the weighted frobenius norm of the difference between two
        gatesets.  Differences in each corresponding gate matrix and spam
        vector element are squared, weighted (using gateWeight
        or spamWeight as applicable), then summed.  The value returned is the
        square root of this sum, or the square root of this sum divided by the
        number of summands if normalize == True.

        Parameters
        ----------
        otherCalc : GateSetCalculator
            the other gate set calculator to difference against.

        transformMx : numpy array, optional
            if not None, transform this gateset by
            G => inv(transformMx) * G * transformMx, for each gate matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this gateset.

        gateWeight : float, optional
           weighting factor for differences between gate elements.

        spamWeight : float, optional
           weighting factor for differences between elements of spam vectors.

        itemWeights : dict, optional
           Dictionary of weighting factors for individual gates and spam
           operators. Weights are applied multiplicatively to the squared
           differences, i.e., (*before* the final square root is taken).  Keys
           can be gate, state preparation, POVM effect, or spam labels.  Values
           are floating point numbers.  By default, weights are set by
           gateWeight and spamWeight.

        normalize : bool, optional
           if True (the default), the frobenius difference is defined by the
           sum of weighted squared-differences divided by the number of
           differences.  If False, this final division is not performed.

        Returns
        -------
        float
        """
        d = 0; T = transformMx
        nSummands = 0.0
        if itemWeights is None: itemWeights = {}

        if T is not None:
            Ti = _nla.inv(T)
            for gateLabel in self.gates:
                wt = itemWeights.get(gateLabel, gateWeight)
                d += wt * _gt.frobeniusdist2(_np.dot(
                    Ti,_np.dot(self.gates[gateLabel],T)),
                    otherCalc.gates[gateLabel] )
                nSummands += wt * _np.size(self.gates[gateLabel])

            for (lbl,rhoV) in self.preps.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * _gt.frobeniusdist2(_np.dot(Ti,rhoV),
                                             otherCalc.preps[lbl])
                nSummands += wt * _np.size(rhoV)

            for (lbl,Evec) in self.effects.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * _gt.frobeniusdist2(_np.dot(
                    _np.transpose(T),Evec),otherCalc.effects[lbl])
                nSummands += wt * _np.size(Evec)

            if self.povm_identity is not None:
                wt = itemWeights.get(self._identityLabel, spamWeight)
                d += wt * _gt.frobeniusdist2(_np.dot(
                    _np.transpose(T),self.povm_identity),otherCalc.povm_identity)
                nSummands += wt * _np.size(self.povm_identity)

        else:
            for gateLabel in self.gates:
                wt = itemWeights.get(gateLabel, gateWeight)
                d += wt * _gt.frobeniusdist2(self.gates[gateLabel],
                                             otherCalc.gates[gateLabel])
                nSummands += wt * _np.size(self.gates[gateLabel])

            for (lbl,rhoV) in self.preps.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * _gt.frobeniusdist2(rhoV,
                                             otherCalc.preps[lbl])
                nSummands += wt *  _np.size(rhoV)

            for (lbl,Evec) in self.effects.items():
                wt = itemWeights.get(lbl, spamWeight)
                d += wt * _gt.frobeniusdist2(Evec,otherCalc.effects[lbl])
                nSummands += wt * _np.size(Evec)

            if self.povm_identity is not None and \
               otherCalc.povm_identity is not None:
                wt = itemWeights.get(self._identityLabel, spamWeight)
                d += wt * _gt.frobeniusdist2(self.povm_identity,
                                             otherCalc.povm_identity)
                nSummands += wt * _np.size(self.povm_identity)

        if normalize and nSummands > 0:
            return _np.sqrt( d / nSummands )
        else:
            return _np.sqrt(d)


    def jtracedist(self, otherCalc, transformMx=None):
        """
        Compute the Jamiolkowski trace distance between two
        gatesets, defined as the maximum of the trace distances
        between each corresponding gate, including spam gates.

        Parameters
        ----------
        otherCalc : GateSetCalculator
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
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ _gt.jtracedist( _np.dot(Ti,_np.dot(self.gates[gateLabel],
                                      T)), otherCalc.gates[gateLabel] )
                       for gateLabel in self.gates ]

            for spamLabel in self.spamdefs:
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.jtracedist( _np.dot(Ti,
                                  _np.dot(spamGate,T)), spamGate2 ) )
        else:
            dists = [ _gt.jtracedist(self.gates[gateLabel], otherCalc.gates[gateLabel])
                      for gateLabel in self.gates ]

            for spamLabel in self.spamdefs:
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.jtracedist(spamGate, spamGate2 ) )

        return max(dists)


    def diamonddist(self, otherCalc, transformMx=None):
        """
        Compute the diamond-norm distance between two
        gatesets, defined as the maximum
        of the diamond-norm distances between each
        corresponding gate, including spam gates.

        Parameters
        ----------
        otherCalc : GateSetCalculator
            the other gate set calculator to difference against.

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
        if T is not None:
            Ti = _nla.inv(T)
            dists = [ _gt.diamonddist(
                    _np.dot(Ti,_np.dot(self.gates[gateLabel],T)),
                    otherCalc.gates[gateLabel] )
                      for gateLabel in self.gates ]

            for spamLabel in self.spamdefs:
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.diamonddist(
                            _np.dot(Ti,_np.dot(spamGate,T)),spamGate2 ) )
        else:
            dists = [ _gt.diamonddist(self.gates[gateLabel],
                                      otherCalc.gates[gateLabel])
                      for gateLabel in self.gates ]

            for spamLabel in self.spamdefs:
                spamGate = self._make_spamgate(spamLabel)
                spamGate2 = otherCalc._make_spamgate(spamLabel)
                if spamGate is not None and spamGate2 is not None:
                    dists.append( _gt.diamonddist(spamGate, spamGate2) )

        return max(dists)

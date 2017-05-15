from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines the GateCalc calculator class"""

import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla
import time as _time
import collections as _collections

from ..tools import gatetools as _gt
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from .profiler import DummyProfiler as _DummyProfiler
_dummy_profiler = _DummyProfiler()

class GateCalc(object):
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
        if not _compat.isstr(label): return False #b/c label could be a custom (rho,E) pair
        return bool(self.spamdefs[label] == (self._remainderLabel, self._remainderLabel))

    def _get_remainder_row_index(self, spam_label_rows):
        """ 
        Returns the index within the spam_label_rows dictionary
        of the remainder label, or None if the remainder label
        is not present.
        """
        remainder_row_index = None
        for spamLabel,rowIndex in spam_label_rows.items():
            if self._is_remainder_spamlabel(spamLabel):
                assert(self.assumeSumToOne) # ensure the remainder label is allowed
                assert(remainder_row_index is None) # ensure there is at most one dummy spam label
                remainder_row_index = rowIndex
        return remainder_row_index


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
            return 1.0 - sum( [self._pr_nr(sl, gatestring, clipTo, bUseScaling) for sl in otherSpamdefs] )
        else:
            return self._pr_nr(spamLabel, gatestring, clipTo, bUseScaling)

    def _pr_nr(self, spamLabel, gatestring, clipTo, bUseScaling):
        """ non-remainder version of pr(...) overridden by derived clases """
        raise NotImplementedError("_pr_nr must be implemented by the derived class") 


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
            otherResults = [self._dpr_nr(sl, gatestring, returnPr, clipTo) for sl in otherSpamdefs]
            if returnPr:
                return -1.0 * sum([dpr for dpr,p in otherResults]), 1.0 - sum([p for dpr,p in otherResults])
            else:
                return -1.0 * sum(otherResults)
        else:
            return self._dpr_nr(spamLabel, gatestring, returnPr, clipTo)

        
    def _dpr_nr(self, spamLabel, gatestring, returnPr, clipTo):
        """ non-remainder version of dpr(...) overridden by derived clases """
        raise NotImplementedError("_dpr_nr must be implemented by the derived class") 


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
            otherResults = [self._hpr_nr(sl, gatestring, returnPr, returnDeriv, clipTo) for sl in otherSpamdefs]
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
        else:
            return self._hpr_nr(spamLabel, gatestring, returnPr, returnDeriv, clipTo)

            
    def _hpr_nr(self, spamLabel, gatestring, returnPr, returnDeriv, clipTo):
        """ non-remainder version of hpr(...) overridden by derived clases """
        raise NotImplementedError("_hpr_nr must be implemented by the derived class") 


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
    

    def _fill_result_tuple(self, result_tup, spam_label_rows, tree_slice,
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
        raise NotImplementedError("bulk_fill_probs(...) is not implemented!")


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
        raise NotImplementedError("bulk_fill_dprobs(...) is not implemented!")


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
        raise NotImplementedError("bulk_fill_hprobs(...) is not implemented!")        


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
        raise NotImplementedError("bulk_hprobs_by_block(...) is not implemented!")
                    

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
        otherCalc : GateCalc
            the other gate calculator to difference against.

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
            Ti = _nla.inv(T) #TODO: generalize inverse op (call T.inverse() if T were a "transform" object?)
            for gateLabel,gate in self.gates.items():
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

            if self.povm_identity is not None:
                wt = itemWeights.get(self._identityLabel, spamWeight)
                d += wt * self.povm_identity.frobeniusdist2(
                    otherCalc.povm_identity, 'effect', T, Ti)
                nSummands += wt * self.povm_identity.dim

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

            if self.povm_identity is not None and \
               otherCalc.povm_identity is not None:
                wt = itemWeights.get(self._identityLabel, spamWeight)
                d += wt * self.povm_identity.frobeniusdist2(
                    otherCalc.povm_identity, 'effect')
                nSummands += wt * self.povm_identity.dim

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

            if self.povm_identity is not None:
                d += self.povm_identity.frobeniusdist2(
                    otherCalc.povm_identity, 'effect', T, Ti)
                nSummands += self.povm_identity.dim
                
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

            if self.povm_identity is not None:
                d += self.povm_identity.frobeniusdist2(
                    otherCalc.povm_identity, 'effect')
                nSummands += self.povm_identity.dim

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

            if self.povm_identity is not None:
                d += self.povm_identity.frobeniusdist2(
                    otherCalc.povm_identity, 'effect', T, Ti)
                nSummands += self.povm_identity.dim
                
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

            if self.povm_identity is not None:
                d += self.povm_identity.frobeniusdist2(
                    otherCalc.povm_identity, 'effect')
                nSummands += self.povm_identity.dim

        spamVal = _np.sqrt(d / nSummands) if (nSummands > 0) else 0
        return max(dists) + spamVal

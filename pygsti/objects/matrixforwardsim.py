"""
Defines the MatrixForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings
import numpy as _np
import numpy.linalg as _nla
import time as _time
import itertools as _itertools
import collections as _collections

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools.matrixtools import _fas
from .profiler import DummyProfiler as _DummyProfiler
from .label import Label as _Label
from .matrixlayout import MatrixCOPALayout as _MatrixCOPALayout
from .matrixlayout import MatrixTimeDepCOPALayout as _MatrixTimeDepCOPALayout
from .forwardsim import ForwardSimulator as _ForwardSimulator
from .forwardsim import _bytes_for_array_types
from .distforwardsim import DistributableForwardSimulator as _DistributableForwardSimulator
from .resourceallocation import ResourceAllocation as _ResourceAllocation
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .evaltree import EvalTree as _EvalTree
from .objectivefns import RawChi2Function as _RawChi2Function
from .objectivefns import RawPoissonPicDeltaLogLFunction as _RawPoissonPicDeltaLogLFunction
_dummy_profiler = _DummyProfiler()

# Smallness tolerances, used internally for conditional scaling required
# to control bulk products, their gradients, and their Hessians.
_PSMALL = 1e-100
_DSMALL = 1e-100
_HSMALL = 1e-100


class SimpleMatrixForwardSimulator(_ForwardSimulator):

    def product(self, circuit, scale=False):
        """
        Compute the product of a specified sequence of operation labels.

        Note: LinearOperator matrices are multiplied in the reversed order of the tuple. That is,
        the first element of circuit can be thought of as the first gate operation
        performed, which is on the far right of the product of matrices.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels.

        scale : bool, optional
            When True, return a scaling factor (see below).

        Returns
        -------
        product : numpy array
            The product or scaled product of the operation matrices.
        scale : float
            Only returned when scale == True, in which case the
            actual product == product * scale.  The purpose of this
            is to allow a trace or other linear operation to be done
            prior to the scaling.
        """
        if scale:
            scaledGatesAndExps = {}
            scale_exp = 0
            G = _np.identity(self.model.dim)
            for lOp in circuit:
                if lOp not in scaledGatesAndExps:
                    opmx = self.model.circuit_layer_operator(lOp, 'op').to_dense()
                    ng = max(_nla.norm(opmx), 1.0)
                    scaledGatesAndExps[lOp] = (opmx / ng, _np.log(ng))

                gate, ex = scaledGatesAndExps[lOp]
                H = _np.dot(gate, G)   # product of gates, starting with identity
                scale_exp += ex   # scale and keep track of exponent
                if H.max() < _PSMALL and H.min() > -_PSMALL:
                    nG = max(_nla.norm(G), _np.exp(-scale_exp))
                    G = _np.dot(gate, G / nG); scale_exp += _np.log(nG)  # LEXICOGRAPHICAL VS MATRIX ORDER
                else: G = H

            old_err = _np.seterr(over='ignore')
            scale = _np.exp(scale_exp)
            _np.seterr(**old_err)

            return G, scale

        else:
            G = _np.identity(self.model.dim)
            for lOp in circuit:
                G = _np.dot(self.model.circuit_layer_operator(lOp, 'op').to_dense(), G)  # LEXI VS MATRIX ORDER
            return G

    def _rho_es_from_spam_tuples(self, rholabel, elabels):
        # This calculator uses the convention that rho has shape (N,1)
        rho = self.model.circuit_layer_operator(rholabel, 'prep').to_dense()[:, None]
        Es = [_np.conjugate(_np.transpose(self.model.circuit_layer_operator(elabel, 'povm').to_dense()[:, None]))
              for elabel in elabels]  # [:, None] becuse of convention: E has shape (1,N)
        return rho, Es

    def _process_wrt_filter(self, wrt_filter, obj):
        """ Helper function for doperation and hoperation below: pulls out pieces of
            a wrt_filter argument relevant for a single object (gate or spam vec) """

        #Create per-gate with-respect-to parameter filters, used to
        # select a subset of all the derivative columns, essentially taking
        # a derivative of only a *subset* of all the gate's parameters

        if isinstance(wrt_filter, slice):
            wrt_filter = _slct.indices(wrt_filter)

        if wrt_filter is not None:
            obj_wrtFilter = []  # values = object-local param indices
            relevant_gpindices = []  # indices into original wrt_filter'd indices

            gpindices = obj.gpindices_as_array()

            for ii, i in enumerate(wrt_filter):
                if i in gpindices:
                    relevant_gpindices.append(ii)
                    obj_wrtFilter.append(list(gpindices).index(i))
            relevant_gpindices = _np.array(relevant_gpindices, _np.int64)
            if len(relevant_gpindices) == 1:
                #Don't return a length-1 list, as this doesn't index numpy arrays
                # like length>1 lists do... ugh.
                relevant_gpindices = slice(relevant_gpindices[0],
                                           relevant_gpindices[0] + 1)
            elif len(relevant_gpindices) == 0:
                #Don't return a length-0 list, as this doesn't index numpy arrays
                # like length>1 lists do... ugh.
                relevant_gpindices = slice(0, 0)  # slice that results in a zero dimension

        else:
            obj_wrtFilter = None
            relevant_gpindices = obj.gpindices

        return obj_wrtFilter, relevant_gpindices

    #Vectorizing Identities. (Vectorization)
    # Note when vectorizing op uses numpy.flatten rows are kept contiguous, so the first identity below is valid.
    # Below we use E(i,j) to denote the elementary matrix where all entries are zero except the (i,j) entry == 1

    # if vec(.) concatenates rows (which numpy.flatten does)
    # vec( A * E(0,1) * B ) = vec( mx w/ row_i = A[i,0] * B[row1] ) = A tensor B^T * vec( E(0,1) )
    # In general: vec( A * X * B ) = A tensor B^T * vec( X )

    # if vec(.) stacks columns
    # vec( A * E(0,1) * B ) = vec( mx w/ col_i = A[col0] * B[0,1] ) = B^T tensor A * vec( E(0,1) )
    # In general: vec( A * X * B ) = B^T tensor A * vec( X )

    def _doperation(self, op_label, flat=False, wrt_filter=None):
        """
        Return the derivative of a length-1 (single-gate) sequence
        """
        dim = self.model.dim
        gate = self.model.circuit_layer_operator(op_label, 'op')
        op_wrtFilter, gpindices = self._process_wrt_filter(wrt_filter, gate)

        # Allocate memory for the final result
        num_deriv_cols = self.model.num_params if (wrt_filter is None) else len(wrt_filter)
        flattened_dprod = _np.zeros((dim**2, num_deriv_cols), 'd')

        _fas(flattened_dprod, [None, gpindices],
             gate.deriv_wrt_params(op_wrtFilter))  # (dim**2, n_params[op_label])

        if _slct.length(gpindices) > 0:  # works for arrays too
            # Compute the derivative of the entire circuit with respect to the
            # gate's parameters and fill appropriate columns of flattened_dprod.
            #gate = self.model.operation[op_label] UNNEEDED (I think)
            _fas(flattened_dprod, [None, gpindices],
                 gate.deriv_wrt_params(op_wrtFilter))  # (dim**2, n_params in wrt_filter for op_label)

        if flat:
            return flattened_dprod
        else:
            # axes = (gate_ij, prod_row, prod_col)
            return _np.swapaxes(flattened_dprod, 0, 1).reshape((num_deriv_cols, dim, dim))

    def _hoperation(self, op_label, flat=False, wrt_filter1=None, wrt_filter2=None):
        """
        Return the hessian of a length-1 (single-gate) sequence
        """
        dim = self.model.dim

        gate = self.model.circuit_layer_operator(op_label, 'op')
        op_wrtFilter1, gpindices1 = self._process_wrt_filter(wrt_filter1, gate)
        op_wrtFilter2, gpindices2 = self._process_wrt_filter(wrt_filter2, gate)

        # Allocate memory for the final result
        num_deriv_cols1 = self.model.num_params if (wrt_filter1 is None) else len(wrt_filter1)
        num_deriv_cols2 = self.model.num_params if (wrt_filter2 is None) else len(wrt_filter2)
        flattened_hprod = _np.zeros((dim**2, num_deriv_cols1, num_deriv_cols2), 'd')

        if _slct.length(gpindices1) > 0 and _slct.length(gpindices2) > 0:  # works for arrays too
            # Compute the derivative of the entire circuit with respect to the
            # gate's parameters and fill appropriate columns of flattened_dprod.
            _fas(flattened_hprod, [None, gpindices1, gpindices2],
                 gate.hessian_wrt_params(op_wrtFilter1, op_wrtFilter2))

        if flat:
            return flattened_hprod
        else:
            return _np.transpose(flattened_hprod, (1, 2, 0)).reshape(
                (num_deriv_cols1, num_deriv_cols2, dim, dim))  # axes = (gate_ij1, gateij2, prod_row, prod_col)

    def dproduct(self, circuit, flat=False, wrt_filter=None):
        """
        Compute the derivative of a specified sequence of operation labels.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels.

        flat : bool, optional
            Affects the shape of the returned derivative array (see below).

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which gate parameters
            to include in the derivative.  Each element is an index into an
            array of gate parameters ordered by concatenating each gate's
            parameters (in the order specified by the model).  This argument
            is used internally for distributing derivative calculations across
            multiple processors.

        Returns
        -------
        deriv : numpy array
            * if flat == False, a M x G x G array, where:

              - M == length of the vectorized model (number of model parameters)
              - G == the linear dimension of a operation matrix (G x G operation matrices).

              and deriv[i,j,k] holds the derivative of the (j,k)-th entry of the product
              with respect to the i-th model parameter.

            * if flat == True, a N x M array, where:

              - N == the number of entries in a single flattened gate (ordering as numpy.flatten)
              - M == length of the vectorized model (number of model parameters)

              and deriv[i,j] holds the derivative of the i-th entry of the flattened
              product with respect to the j-th model parameter.
        """

        # LEXICOGRAPHICAL VS MATRIX ORDER
        # we do matrix multiplication in this order (easier to think about)
        revOpLabelList = tuple(reversed(tuple(circuit)))
        N = len(revOpLabelList)  # length of circuit

        #  prod = G1 * G2 * .... * GN , a matrix                                                                                                                # noqa
        #  dprod/d(opLabel)_ij   = sum_{L s.t. G(L) == oplabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)                       # noqa
        #  vec( dprod/d(opLabel)_ij ) = sum_{L s.t. G(L) == oplabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]                              # noqa
        #                               = [ sum_{L s.t. G(L) == oplabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] * vec( dG(L)/dij) )                      # noqa
        #  if dG(L)/dij = E(i,j)                                                                                                                                # noqa
        #                               = vec(i,j)-col of [ sum_{L s.t. G(L) == oplabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]]                          # noqa
        #
        # So for each opLabel the matrix [ sum_{L s.t. GL == oplabel} [ (G1 ... G(L-1)) tensor (G(L+1) ... GN)^T ]] has
        # columns which correspond to the vectorized derivatives of each of the product components (i.e. prod_kl) with
        # respect to a given gateLabel_ij.  This function returns a concatenated form of the above matrices, so that
        # each column corresponds to a (opLabel,i,j) tuple and each row corresponds to an element of the product (els of
        # prod.flatten()).
        #
        # Note: if gate G(L) is just a matrix of parameters, then dG(L)/dij = E(i,j), an elementary matrix

        dim = self.model.dim

        #Cache partial products (relatively little mem required)
        leftProds = []
        G = _np.identity(dim); leftProds.append(G)
        for opLabel in revOpLabelList:
            G = _np.dot(G, self.model.circuit_layer_operator(opLabel, 'op').to_dense())
            leftProds.append(G)

        rightProdsT = []
        G = _np.identity(dim); rightProdsT.append(_np.transpose(G))
        for opLabel in reversed(revOpLabelList):
            G = _np.dot(self.model.circuit_layer_operator(opLabel, 'op').to_dense(), G)
            rightProdsT.append(_np.transpose(G))

        # Allocate memory for the final result
        num_deriv_cols = self.model.num_params if (wrt_filter is None) else len(wrt_filter)
        flattened_dprod = _np.zeros((dim**2, num_deriv_cols), 'd')

        # For each operation label, compute the derivative of the entire circuit
        #  with respect to only that gate's parameters and fill the appropriate
        #  columns of flattened_dprod.
        uniqueOpLabels = sorted(list(set(revOpLabelList)))
        for opLabel in uniqueOpLabels:
            gate = self.model.circuit_layer_operator(opLabel, 'op')
            op_wrtFilter, gpindices = self._process_wrt_filter(wrt_filter, gate)
            dop_dopLabel = gate.deriv_wrt_params(op_wrtFilter)

            for (i, gl) in enumerate(revOpLabelList):
                if gl != opLabel: continue  # loop over locations of opLabel
                LRproduct = _np.kron(leftProds[i], rightProdsT[N - 1 - i])  # (dim**2, dim**2)
                _fas(flattened_dprod, [None, gpindices],
                     _np.dot(LRproduct, dop_dopLabel), add=True)  # (dim**2, n_params[opLabel])

        if flat:
            return flattened_dprod
        else:
            # axes = (gate_ij, prod_row, prod_col)
            return _np.swapaxes(flattened_dprod, 0, 1).reshape((num_deriv_cols, dim, dim))

    def hproduct(self, circuit, flat=False, wrt_filter1=None, wrt_filter2=None):
        """
        Compute the hessian of a specified sequence of operation labels.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels.

        flat : bool, optional
            Affects the shape of the returned derivative array (see below).

        wrt_filter1 : list of ints, optional
            If not None, a list of integers specifying which parameters
            to differentiate with respect to in the first (row)
            derivative operations.  Each element is an model-parameter index.
            This argument is used internally for distributing derivative calculations
            across multiple processors.

        wrt_filter2 : list of ints, optional
            If not None, a list of integers specifying which parameters
            to differentiate with respect to in the second (col)
            derivative operations.  Each element is an model-parameter index.
            This argument is used internally for distributing derivative calculations
            across multiple processors.

        Returns
        -------
        hessian : numpy array
            * if flat == False, a  M x M x G x G numpy array, where:

              - M == length of the vectorized model (number of model parameters)
              - G == the linear dimension of a operation matrix (G x G operation matrices).

              and hessian[i,j,k,l] holds the derivative of the (k,l)-th entry of the product
              with respect to the j-th then i-th model parameters.

            * if flat == True, a  N x M x M numpy array, where:

              - N == the number of entries in a single flattened gate (ordered as numpy.flatten)
              - M == length of the vectorized model (number of model parameters)

              and hessian[i,j,k] holds the derivative of the i-th entry of the flattened
              product with respect to the k-th then k-th model parameters.
        """

        # LEXICOGRAPHICAL VS MATRIX ORDER
        # we do matrix multiplication in this order (easier to think about)
        revOpLabelList = tuple(reversed(tuple(circuit)))

        #  prod = G1 * G2 * .... * GN , a matrix                                                                                                                # noqa
        #  dprod/d(opLabel)_ij   = sum_{L s.t. GL == oplabel} [ G1 ... G(L-1) dG(L)/dij G(L+1) ... GN ] , a matrix for each given (i,j)                         # noqa
        #  d2prod/d(opLabel1)_kl*d(opLabel2)_ij = sum_{M s.t. GM == gatelabel1} sum_{L s.t. GL == gatelabel2, M < L}                                            # noqa
        #                                                 [ G1 ... G(M-1) dG(M)/dkl G(M+1) ... G(L-1) dG(L)/dij G(L+1) ... GN ] + {similar with L < M}          # noqa
        #                                                 + sum{M==L} [ G1 ... G(M-1) d2G(M)/(dkl*dij) G(M+1) ... GN ]                                          # noqa
        #                                                 a matrix for each given (i,j,k,l)                                                                     # noqa
        #  vec( d2prod/d(opLabel1)_kl*d(opLabel2)_ij ) = sum{...} [ G1 ...  G(M-1) dG(M)/dkl G(M+1) ... G(L-1) tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]      # noqa
        #                                                  = sum{...} [ unvec( G1 ...  G(M-1) tensor (G(M+1) ... G(L-1))^T vec( dG(M)/dkl ) )                   # noqa
        #                                                                tensor (G(L+1) ... GN)^T vec( dG(L)/dij ) ]                                            # noqa
        #                                                  + sum{ L < M} [ G1 ...  G(L-1) tensor                                                                # noqa
        #                                                       ( unvec( G(L+1) ... G(M-1) tensor (G(M+1) ... GN)^T vec( dG(M)/dkl ) ) )^T vec( dG(L)/dij ) ]   # noqa
        #                                                  + sum{ L == M} [ G1 ...  G(M-1) tensor (G(M+1) ... GN)^T vec( d2G(M)/dkl*dji )                       # noqa
        #
        #  Note: ignoring L == M terms assumes that d^2 G/(dij)^2 == 0, which is true IF each operation matrix element
        #  is at most *linear* in each of the gate parameters.  If this is not the case, need LinearOperator objects to
        #  have a 2nd-deriv method in addition of deriv_wrt_params
        #
        #  Note: unvec( X ) can be done efficiently by actually computing X^T ( note (A tensor B)^T = A^T tensor B^T )
        #  and using numpy's reshape

        dim = self.model.dim

        uniqueOpLabels = sorted(list(set(revOpLabelList)))
        used_operations = _collections.OrderedDict()

        #Cache processed parameter filters for multiple uses below
        gpindices1 = {}; gate_wrtFilters1 = {}
        gpindices2 = {}; gate_wrtFilters2 = {}
        for l in uniqueOpLabels:
            used_operations[l] = self.model.circuit_layer_operator(l, 'op')
            gate_wrtFilters1[l], gpindices1[l] = self._process_wrt_filter(wrt_filter1, used_operations[l])
            gate_wrtFilters2[l], gpindices2[l] = self._process_wrt_filter(wrt_filter2, used_operations[l])

        #Cache partial products (relatively little mem required)
        prods = {}
        ident = _np.identity(dim)
        for (i, opLabel1) in enumerate(revOpLabelList):  # loop over "starting" gate
            prods[(i, i - 1)] = ident  # product of no gates
            G = ident
            for (j, opLabel2) in enumerate(revOpLabelList[i:], start=i):  # loop over "ending" gate (>= starting gate)
                G = _np.dot(G, self.model.circuit_layer_operator(opLabel2, 'op').to_dense())
                prods[(i, j)] = G
        prods[(len(revOpLabelList), len(revOpLabelList) - 1)] = ident  # product of no gates

        #Also Cache gate jacobians (still relatively little mem required)
        dop_dopLabel1 = {
            opLabel: gate.deriv_wrt_params(gate_wrtFilters1[opLabel])
            for opLabel, gate in used_operations.items()}

        if wrt_filter1 == wrt_filter2:
            dop_dopLabel2 = dop_dopLabel1
        else:
            dop_dopLabel2 = {
                opLabel: gate.deriv_wrt_params(gate_wrtFilters2[opLabel])
                for opLabel, gate in used_operations.items()}

        #Finally, cache any nonzero gate hessians (memory?)
        hop_dopLabels = {}
        for opLabel, gate in used_operations.items():
            if gate.has_nonzero_hessian():
                hop_dopLabels[opLabel] = gate.hessian_wrt_params(
                    gate_wrtFilters1[opLabel], gate_wrtFilters2[opLabel])

        # Allocate memory for the final result
        num_deriv_cols1 = self.model.num_params if (wrt_filter1 is None) else len(wrt_filter1)
        num_deriv_cols2 = self.model.num_params if (wrt_filter2 is None) else len(wrt_filter2)
        flattened_d2prod = _np.zeros((dim**2, num_deriv_cols1, num_deriv_cols2), 'd')

        # For each pair of gates in the string, compute the hessian of the entire
        #  circuit with respect to only those two gates' parameters and fill
        #  add the result to the appropriate block of flattened_d2prod.

        #NOTE: if we needed to perform a hessian calculation (i.e. for l==m) then
        # it could make sense to iterate through the self.operations.keys() as in
        # dproduct(...) and find the labels in the string which match the current
        # gate (so we only need to compute this gate hessian once).  But since we're
        # assuming that the gates are at most linear in their parameters, this
        # isn't currently needed.

        N = len(revOpLabelList)
        for m, opLabel1 in enumerate(revOpLabelList):
            inds1 = gpindices1[opLabel1]
            nDerivCols1 = dop_dopLabel1[opLabel1].shape[1]
            if nDerivCols1 == 0: continue

            for l, opLabel2 in enumerate(revOpLabelList):
                inds2 = gpindices1[opLabel2]
                #nDerivCols2 = dop_dopLabel2[opLabel2].shape[1]

                # FUTURE: we could add logic that accounts for the symmetry of the Hessian, so that
                # if gl1 and gl2 are both in opsToVectorize1 and opsToVectorize2 we only compute d2(prod)/d(gl1)d(gl2)
                # and not d2(prod)/d(gl2)d(gl1) ...

                if m < l:
                    x0 = _np.kron(_np.transpose(prods[(0, m - 1)]), prods[(m + 1, l - 1)])  # (dim**2, dim**2)
                    x = _np.dot(_np.transpose(dop_dopLabel1[opLabel1]), x0); xv = x.view()  # (nDerivCols1,dim**2)
                    xv.shape = (nDerivCols1, dim, dim)  # (reshape without copying - throws error if copy is needed)
                    y = _np.dot(_np.kron(xv, _np.transpose(prods[(l + 1, N - 1)])), dop_dopLabel2[opLabel2])
                    # above: (nDerivCols1,dim**2,dim**2) * (dim**2,nDerivCols2) = (nDerivCols1,dim**2,nDerivCols2)
                    flattened_d2prod[:, inds1, inds2] += _np.swapaxes(y, 0, 1)
                    # above: dim = (dim2, nDerivCols1, nDerivCols2);
                    # swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)
                elif l < m:
                    x0 = _np.kron(_np.transpose(prods[(l + 1, m - 1)]), prods[(m + 1, N - 1)])  # (dim**2, dim**2)
                    x = _np.dot(_np.transpose(dop_dopLabel1[opLabel1]), x0); xv = x.view()  # (nDerivCols1,dim**2)
                    xv.shape = (nDerivCols1, dim, dim)  # (reshape without copying - throws error if copy is needed)
                    # transposes each of the now un-vectorized dim x dim mxs corresponding to a single kl
                    xv = _np.swapaxes(xv, 1, 2)
                    y = _np.dot(_np.kron(prods[(0, l - 1)], xv), dop_dopLabel2[opLabel2])
                    # above: (nDerivCols1,dim**2,dim**2) * (dim**2,nDerivCols2) = (nDerivCols1,dim**2,nDerivCols2)

                    flattened_d2prod[:, inds1, inds2] += _np.swapaxes(y, 0, 1)
                    # above: dim = (dim2, nDerivCols1, nDerivCols2);
                    # swapaxes takes (kl,vec_prod_indx,ij) => (vec_prod_indx,kl,ij)

                else:
                    # l==m, which we *used* to assume gave no contribution since we assume all gate elements are at most
                    # linear in the parameters
                    assert(opLabel1 == opLabel2)
                    if opLabel1 in hop_dopLabels:  # indicates a non-zero hessian
                        x0 = _np.kron(_np.transpose(prods[(0, m - 1)]), prods[(m + 1, N - 1)])  # (dim**2, dim**2)
                        # (nDerivCols1,nDerivCols2,dim**2)
                        x = _np.dot(_np.transpose(hop_dopLabels[opLabel1], axes=(1, 2, 0)), x0); xv = x.view()
                        xv = _np.transpose(xv, axes=(2, 0, 1))  # (dim2, nDerivCols1, nDerivCols2)
                        flattened_d2prod[:, inds1, inds2] += xv

        if flat:
            return flattened_d2prod  # axes = (vectorized_op_el_index, model_parameter1, model_parameter2)
        else:
            vec_kl_size, vec_ij_size = flattened_d2prod.shape[1:3]  # == num_deriv_cols1, num_deriv_cols2
            return _np.rollaxis(flattened_d2prod, 0, 3).reshape((vec_kl_size, vec_ij_size, dim, dim))
            # axes = (model_parameter1, model_parameter2, model_element_row, model_element_col)

    def _compute_circuit_outcome_probabilities(self, array_to_fill, circuit, outcomes, resource_alloc, time=None):
        """
        Compute probabilities of a multiple "outcomes" for a single circuit.

        The outcomes correspond to `circuit` sandwiched between `rholabel` (a state preparation)
        and the multiple effect labels in `elabels`.

        Parameters
        ----------
        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        use_scaling : bool, optional
            Whether to use a post-scaled product internally.  If False, this
            routine will run slightly faster, but with a chance that the
            product will overflow and the subsequent trace operation will
            yield nan as the returned probability.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        numpy.ndarray
            An array of floating-point probabilities, corresponding to
            the elements of `elabels`.
        """
        use_scaling = False  # Hardcoded for now
        assert(time is None), "MatrixForwardSimulator cannot be used to simulate time-dependent circuits"

        expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        for spc, spc_outcomes in expanded_circuit_outcomes.items():  # spc is a SeparatePOVMCircuit
            indices = [outcome_to_index[o] for o in spc_outcomes]
            rholabel = spc.circuit_without_povm[0]
            circuit_ops = spc.circuit_without_povm[1:]
            rho, Es = self._rho_es_from_spam_tuples(rholabel, spc.full_effect_labels)
            #shapes: rho = (N,1), Es = (len(elabels),N)

            if use_scaling:
                old_err = _np.seterr(over='ignore')
                G, scale = self.product(circuit_ops, True)
                if self.model.evotype == "statevec":
                    ps = _np.real(_np.abs(_np.dot(Es, _np.dot(G, rho)) * scale)**2)
                else:  # evotype == "densitymx"
                    # probability, with scaling applied (may generate overflow, but OK)
                    ps = _np.real(_np.dot(Es, _np.dot(G, rho)) * scale)
                _np.seterr(**old_err)

            else:  # no scaling -- faster but susceptible to overflow
                G = self.product(circuit_ops, False)
                if self.model.evotype == "statevec":
                    ps = _np.real(_np.abs(_np.dot(Es, _np.dot(G, rho)))**2)
                else:  # evotype == "densitymx"
                    ps = _np.real(_np.dot(Es, _np.dot(G, rho)))
            array_to_fill[indices] = ps.flat

    #TODO: move _dpr and _hpr into a unit test?  They're not really needed here, as
    # using the default finite differencing should be fine, and it's rare that we want
    # the derivative of a single circuit anyway, so this is rarely used.
    #def _dpr(self, spam_tuple, circuit, return_pr, clip_to):
    #    """
    #    Compute the derivative of the probability corresponding to `circuit` and `spam_tuple`.
    #
    #    Parameters
    #    ----------
    #    spam_tuple : (rho_label, simplified_effect_label)
    #        Specifies the prep and POVM effect used to compute the probability.
    #
    #    circuit : Circuit or tuple
    #        A tuple-like object of *simplified* gates (e.g. may include
    #        instrument elements like 'Imyinst_0')
    #
    #    return_pr : bool
    #        when set to True, additionally return the probability itself.
    #
    #    clip_to : 2-tuple
    #        (min,max) to clip returned probability to if not None.
    #        Only relevant when pr_mx_to_fill is not None.
    #
    #    Returns
    #    -------
    #    derivative : numpy array
    #        a 1 x M numpy array of derivatives of the probability w.r.t.
    #        each model parameter (M is the number of model parameters).
    #
    #    probability : float
    #        only returned if return_pr == True.
    #    """
    #    if self.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")
    #    # To support unitary evolution we need to:
    #    # - alter product, dproduct, etc. to allow for *complex* derivatives, since matrices can be complex
    #    # - update probability-deriv. computations: dpr/dx -> d|pr|^2/dx = d(pr*pr.C)/dx = dpr/dx*pr.C + pr*dpr/dx.C
    #    #    = 2 Re(dpr/dx*pr.C) , where dpr/dx is the usual density-matrix-mode probability
    #    # (TODO in FUTURE)
    #
    #    #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
    #    #  dpr/d(op_label)_ij = sum E_k [dprod/d(op_label)_ij]_kl rho_l
    #    #  dpr/d(rho)_i = sum E_k prod_ki
    #    #  dpr/d(E)_i   = sum prod_il rho_l
    #
    #    rholabel, elabel = spam_tuple  # can't deal w/"custom" spam label...
    #    rho, E = self._rho_e_from_spam_tuple(spam_tuple)
    #    rhoVec = self.model.prep(rholabel)  # distinct from rho,E b/c rho,E are
    #    EVec = self.model.effect(elabel)   # arrays, these are SPAMVecs
    #
    #    #Derivs wrt Gates
    #    old_err = _np.seterr(over='ignore')
    #    prod, scale = self.product(circuit, True)
    #    dprod_dOps = self.dproduct(circuit)
    #    dpr_dOps = _np.empty((1, self.model.num_params))
    #    for i in range(self.model.num_params):
    #        dpr_dOps[0, i] = float(_np.dot(E, _np.dot(dprod_dOps[i], rho)))
    #
    #    if return_pr:
    #        p = _np.dot(E, _np.dot(prod, rho)) * scale  # may generate overflow, but OK
    #        if clip_to is not None: p = _np.clip(p, clip_to[0], clip_to[1])
    #
    #    #Derivs wrt SPAM
    #    derivWrtAnyRhovec = scale * _np.dot(E, prod)
    #    dpr_drhos = _np.zeros((1, self.model.num_params))
    #    _fas(dpr_drhos, [0, self.model.prep(rholabel).gpindices],
    #         _np.dot(derivWrtAnyRhovec, rhoVec.deriv_wrt_params()))  # may overflow, but OK
    #
    #    dpr_dEs = _np.zeros((1, self.model.num_params))
    #    derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod, rho))  # may overflow, but OK
    #    # (** doesn't depend on eIndex **) -- TODO: should also conjugate() here if complex?
    #    _fas(dpr_dEs, [0, EVec.gpindices],
    #         _np.dot(derivWrtAnyEvec, EVec.deriv_wrt_params()))
    #
    #    _np.seterr(**old_err)
    #
    #    if return_pr:
    #        return dpr_drhos + dpr_dEs + dpr_dOps, p
    #    else: return dpr_drhos + dpr_dEs + dpr_dOps
    #
    #def _hpr(self, spam_tuple, circuit, return_pr, return_deriv, clip_to):
    #    """
    #    Compute the Hessian of the probability given by `circuit` and `spam_tuple`.
    #
    #    Parameters
    #    ----------
    #    spam_tuple : (rho_label, simplified_effect_label)
    #        Specifies the prep and POVM effect used to compute the probability.
    #
    #    circuit : Circuit or tuple
    #        A tuple-like object of *simplified* gates (e.g. may include
    #        instrument elements like 'Imyinst_0')
    #
    #    return_pr : bool
    #        when set to True, additionally return the probability itself.
    #
    #    return_deriv : bool
    #        when set to True, additionally return the derivative of the
    #        probability.
    #
    #    clip_to : 2-tuple
    #        (min,max) to clip returned probability to if not None.
    #        Only relevant when pr_mx_to_fill is not None.
    #
    #    Returns
    #    -------
    #    hessian : numpy array
    #        a 1 x M x M array, where M is the number of model parameters.
    #        hessian[0,j,k] is the derivative of the probability w.r.t. the
    #        k-th then the j-th model parameter.
    #
    #    derivative : numpy array
    #        only returned if return_deriv == True. A 1 x M numpy array of
    #        derivatives of the probability w.r.t. each model parameter.
    #
    #    probability : float
    #        only returned if return_pr == True.
    #    """
    #    if self.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")
    #
    #    #  pr = Tr( |rho><E| * prod ) = sum E_k prod_kl rho_l
    #    #  d2pr/d(opLabel1)_mn d(opLabel2)_ij = sum E_k [dprod/d(opLabel1)_mn d(opLabel2)_ij]_kl rho_l
    #    #  d2pr/d(rho)_i d(op_label)_mn = sum E_k [dprod/d(op_label)_mn]_ki     (and same for other diff order)
    #    #  d2pr/d(E)_i d(op_label)_mn   = sum [dprod/d(op_label)_mn]_il rho_l   (and same for other diff order)
    #    #  d2pr/d(E)_i d(rho)_j          = prod_ij                                (and same for other diff order)
    #    #  d2pr/d(E)_i d(E)_j            = 0
    #    #  d2pr/d(rho)_i d(rho)_j        = 0
    #
    #    rholabel, elabel = spam_tuple
    #    rho, E = self._rho_e_from_spam_tuple(spam_tuple)
    #    rhoVec = self.model.prep(rholabel)  # distinct from rho,E b/c rho,E are
    #    EVec = self.model.effect(elabel)   # arrays, these are SPAMVecs
    #
    #    d2prod_dGates = self.hproduct(circuit)
    #    assert(d2prod_dGates.shape[0] == d2prod_dGates.shape[1])
    #
    #    d2pr_dOps2 = _np.empty((1, self.model.num_params, self.model.num_params))
    #    for i in range(self.model.num_params):
    #        for j in range(self.model.num_params):
    #            d2pr_dOps2[0, i, j] = float(_np.dot(E, _np.dot(d2prod_dGates[i, j], rho)))
    #
    #    old_err = _np.seterr(over='ignore')
    #
    #    prod, scale = self.product(circuit, True)
    #    if return_pr:
    #        p = _np.dot(E, _np.dot(prod, rho)) * scale  # may generate overflow, but OK
    #        if clip_to is not None: p = _np.clip(p, clip_to[0], clip_to[1])
    #
    #    dprod_dOps = self.dproduct(circuit)
    #    assert(dprod_dOps.shape[0] == self.model.num_params)
    #    if return_deriv:  # same as in dpr(...)
    #        dpr_dOps = _np.empty((1, self.model.num_params))
    #        for i in range(self.model.num_params):
    #            dpr_dOps[0, i] = float(_np.dot(E, _np.dot(dprod_dOps[i], rho)))
    #
    #    #Derivs wrt SPAM
    #    if return_deriv:  # same as in dpr(...)
    #        dpr_drhos = _np.zeros((1, self.model.num_params))
    #        derivWrtAnyRhovec = scale * _np.dot(E, prod)
    #        _fas(dpr_drhos, [0, self.model.prep(rholabel).gpindices],
    #             _np.dot(derivWrtAnyRhovec, rhoVec.deriv_wrt_params()))  # may overflow, but OK
    #
    #        dpr_dEs = _np.zeros((1, self.model.num_params))
    #        derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod, rho))  # may overflow, but OK
    #        _fas(dpr_dEs, [0, EVec.gpindices],
    #             _np.dot(derivWrtAnyEvec, EVec.deriv_wrt_params()))
    #
    #        dpr = dpr_drhos + dpr_dEs + dpr_dOps
    #
    #    d2pr_drhos = _np.zeros((1, self.model.num_params, self.model.num_params))
    #    _fas(d2pr_drhos, [0, None, self.model.prep(rholabel).gpindices],
    #         _np.dot(_np.dot(E, dprod_dOps), rhoVec.deriv_wrt_params())[0])  # (= [0,:,:])
    #
    #    d2pr_dEs = _np.zeros((1, self.model.num_params, self.model.num_params))
    #    derivWrtAnyEvec = _np.squeeze(_np.dot(dprod_dOps, rho), axis=(2,))
    #    _fas(d2pr_dEs, [0, None, EVec.gpindices],
    #         _np.dot(derivWrtAnyEvec, EVec.deriv_wrt_params()))
    #
    #    d2pr_dErhos = _np.zeros((1, self.model.num_params, self.model.num_params))
    #    derivWrtAnyEvec = scale * _np.dot(prod, rhoVec.deriv_wrt_params())  # may generate overflow, but OK
    #    _fas(d2pr_dErhos, [0, EVec.gpindices, self.model.prep(rholabel).gpindices],
    #         _np.dot(_np.transpose(EVec.deriv_wrt_params()), derivWrtAnyEvec))
    #
    #    #Note: these 2nd derivatives are non-zero when the spam vectors have
    #    # a more than linear dependence on their parameters.
    #    if self.model.prep(rholabel).has_nonzero_hessian():
    #        derivWrtAnyRhovec = scale * _np.dot(E, prod)  # may overflow, but OK
    #        d2pr_d2rhos = _np.zeros((1, self.model.num_params, self.model.num_params))
    #        _fas(d2pr_d2rhos, [0, self.model.prep(rholabel).gpindices, self.model.prep(rholabel).gpindices],
    #             _np.tensordot(derivWrtAnyRhovec, self.model.prep(rholabel).hessian_wrt_params(), (1, 0)))
    #        # _np.einsum('ij,jkl->ikl', derivWrtAnyRhovec, self.model.prep(rholabel).hessian_wrt_params())
    #    else:
    #        d2pr_d2rhos = 0
    #
    #    if self.model.effect(elabel).has_nonzero_hessian():
    #        derivWrtAnyEvec = scale * _np.transpose(_np.dot(prod, rho))  # may overflow, but OK
    #        d2pr_d2Es = _np.zeros((1, self.model.num_params, self.model.num_params))
    #        _fas(d2pr_d2Es, [0, self.model.effect(elabel).gpindices, self.model.effect(elabel).gpindices],
    #             _np.tensordot(derivWrtAnyEvec, self.model.effect(elabel).hessian_wrt_params(), (1, 0)))
    #        # _np.einsum('ij,jkl->ikl',derivWrtAnyEvec,self.model.effect(elabel).hessian_wrt_params())
    #    else:
    #        d2pr_d2Es = 0
    #
    #    ret = d2pr_dErhos + _np.transpose(d2pr_dErhos, (0, 2, 1)) + \
    #        d2pr_drhos + _np.transpose(d2pr_drhos, (0, 2, 1)) + \
    #        d2pr_dEs + _np.transpose(d2pr_dEs, (0, 2, 1)) + \
    #        d2pr_d2rhos + d2pr_d2Es + d2pr_dOps2
    #    # Note: add transposes b/c spam terms only compute one triangle of hessian
    #    # Note: d2pr_d2rhos and d2pr_d2Es terms are always zero
    #
    #    _np.seterr(**old_err)
    #
    #    if return_deriv:
    #        if return_pr: return ret, dpr, p
    #        else: return ret, dpr
    #    else:
    #        if return_pr: return ret, p
    #        else: return ret
    #def _check(self, eval_tree, pr_mx_to_fill=None, d_pr_mx_to_fill=None, h_pr_mx_to_fill=None, clip_to=None):
    #    # compare with older slower version that should do the same thing (for debugging)
    #    master_circuit_list = eval_tree.compute_circuits(permute=False)  # raw circuits
    #
    #    for spamTuple, (fInds, gInds) in eval_tree.spamtuple_indices.items():
    #        circuit_list = master_circuit_list[gInds]
    #
    #        if pr_mx_to_fill is not None:
    #            check_vp = _np.array([self._prs(spamTuple[0], [spamTuple[1]], circuit, clip_to, False)[0]
    #                                  for circuit in circuit_list])
    #            if _nla.norm(pr_mx_to_fill[fInds] - check_vp) > 1e-6:
    #                _warnings.warn("norm(vp-check_vp) = %g - %g = %g" %
    #                               (_nla.norm(pr_mx_to_fill[fInds]),
    #                                _nla.norm(check_vp),
    #                                _nla.norm(pr_mx_to_fill[fInds] - check_vp)))  # pragma: no cover
    #
    #        if d_pr_mx_to_fill is not None:
    #            check_vdp = _np.concatenate(
    #                [self._dpr(spamTuple, circuit, False, clip_to)
    #                 for circuit in circuit_list], axis=0)
    #            if _nla.norm(d_pr_mx_to_fill[fInds] - check_vdp) > 1e-6:
    #                _warnings.warn("norm(vdp-check_vdp) = %g - %g = %g" %
    #                               (_nla.norm(d_pr_mx_to_fill[fInds]),
    #                                _nla.norm(check_vdp),
    #                                _nla.norm(d_pr_mx_to_fill[fInds] - check_vdp)))  # pragma: no cover
    #
    #        if h_pr_mx_to_fill is not None:
    #            check_vhp = _np.concatenate(
    #                [self._hpr(spamTuple, circuit, False, False, clip_to)
    #                 for circuit in circuit_list], axis=0)
    #            if _nla.norm(h_pr_mx_to_fill[fInds][0] - check_vhp[0]) > 1e-6:
    #                _warnings.warn("norm(vhp-check_vhp) = %g - %g = %g" %
    #                               (_nla.norm(h_pr_mx_to_fill[fInds]),
    #                                _nla.norm(check_vhp),
    #                                _nla.norm(h_pr_mx_to_fill[fInds] - check_vhp)))  # pragma: no cover


class MatrixForwardSimulator(_DistributableForwardSimulator, SimpleMatrixForwardSimulator):

    @classmethod
    def _array_types_for_method(cls, method_name):
        # The array types of *intermediate* or *returned* values within various class methods (for memory estimates)
        if method_name == 'bulk_fill_probs': return ('zdd', 'z', 'z')  # cache of gates, scales, and scaleVals
        if method_name == 'bulk_fill_dprobs': return ('zddp',)  # cache x dim x dim x distributed_nparams
        if method_name == 'bulk_fill_hprobs': return ('zddpp',)  # cache x dim x dim x dist_nparams1 x dist_nparams2
        return super()._array_types_for_method(method_name)

    def __init__(self, model=None, distribute_by_timestamp=False):
        super().__init__(model)
        self._mode = "distribute_by_timestamp" if distribute_by_timestamp else "time_independent"

    def copy(self):
        """
        Return a shallow copy of this MatrixForwardSimulator

        Returns
        -------
        MatrixForwardSimulator
        """
        return MatrixForwardSimulator(self.model)

    def _compute_product_cache(self, layout_atom_tree, comm=None):
        """
        Computes an array of operation sequence products (process matrices).

        Note: will *not* parallelize computation:  parallelization should be
        done at a higher level.
        """
        dim = self.model.dim

        #Note: previously, we tried to allow for parallelization of
        # _compute_product_cache when the tree was split, but this is was
        # incorrect (and luckily never used) - so it's been removed.

        if comm is not None:  # ignoring comm since can't do anything with it!
            #_warnings.warn("More processors than can be used for product computation")
            pass  # this is a fairly common occurrence, and doesn't merit a warning

        # ------------------------------------------------------------------

        eval_tree = layout_atom_tree
        cacheSize = len(eval_tree)
        prodCache = _np.zeros((cacheSize, dim, dim), 'd')
        scaleCache = _np.zeros(cacheSize, 'd')

        for iDest, iRight, iLeft in eval_tree:

            #Special case of an "initial operation" that can be filled directly
            if iRight is None:  # then iLeft gives operation:
                opLabel = iLeft
                if opLabel is None:
                    prodCache[iDest] = _np.identity(dim)
                    # Note: scaleCache[i] = 0.0 from initialization
                else:
                    gate = self.model.circuit_layer_operator(opLabel, 'op').to_dense()
                    nG = max(_nla.norm(gate), 1.0)
                    prodCache[iDest] = gate / nG
                    scaleCache[iDest] = _np.log(nG)
                continue

            # combine iLeft + iRight => iDest
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from eval_tree because
            # (iRight,iLeft,iFinal) = tup implies circuit[i] = circuit[iLeft] + circuit[iRight], but we want:
            # since then matrixOf(circuit[i]) = matrixOf(circuit[iLeft]) * matrixOf(circuit[iRight])
            L, R = prodCache[iLeft], prodCache[iRight]
            prodCache[iDest] = _np.dot(L, R)
            scaleCache[iDest] = scaleCache[iLeft] + scaleCache[iRight]

            if prodCache[iDest].max() < _PSMALL and prodCache[iDest].min() > -_PSMALL:
                nL, nR = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),
                             1e-300), max(_nla.norm(R), _np.exp(-scaleCache[iRight]), 1e-300)
                sL, sR = L / nL, R / nR
                prodCache[iDest] = _np.dot(sL, sR); scaleCache[iDest] += _np.log(nL) + _np.log(nR)

        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0]  # may be duplicates (a list, not a set)
        # since all scaled gates start with norm <= 1, products should all have norm <= 1
        assert(len(nanOrInfCacheIndices) == 0)

        return prodCache, scaleCache

    def _compute_dproduct_cache(self, layout_atom_tree, prod_cache, scale_cache,
                                comm=None, wrt_slice=None, profiler=None):
        """
        Computes a tree of product derivatives in a linear cache space. Will
        use derivative columns to parallelize computation.
        """

        if profiler is None: profiler = _dummy_profiler
        dim = self.model.dim
        nDerivCols = self.model.num_params if (wrt_slice is None) \
            else _slct.length(wrt_slice)
        deriv_shape = (nDerivCols, dim, dim)
        eval_tree = layout_atom_tree
        cacheSize = len(eval_tree)

        # ------------------------------------------------------------------

        #print("MPI: _compute_dproduct_cache begin: %d deriv cols" % nDerivCols)
        if comm is not None and comm.Get_size() > 1:
            #print("MPI: _compute_dproduct_cache called w/comm size %d" % comm.Get_size())
            # parallelize of deriv cols, then sub-trees (if available and necessary)

            if comm.Get_size() > nDerivCols:

                #If there are more processors than deriv cols, give a
                # warning -- note that we *cannot* make use of a tree being
                # split because there's no good way to reconstruct the
                # *non-final* parent-tree elements from those of the sub-trees.
                _warnings.warn("Increased speed could be obtained by giving dproduct cache computation"
                               " *fewer* processors, as there are more cpus than derivative columns.")

            # Use comm to distribute columns
            allDerivColSlice = slice(0, nDerivCols) if (wrt_slice is None) else wrt_slice
            _, myDerivColSlice, _, mySubComm = \
                _mpit.distribute_slice(allDerivColSlice, comm)
            #print("MPI: _compute_dproduct_cache over %d cols (%s) (rank %d computing %s)" \
            #    % (nDerivCols, str(allDerivColIndices), comm.Get_rank(), str(myDerivColIndices)))
            if mySubComm is not None and mySubComm.Get_size() > 1:
                _warnings.warn("Too many processors to make use of in "
                               " _compute_dproduct_cache.")
                if mySubComm.Get_rank() > 0: myDerivColSlice = slice(0, 0)
                #don't compute anything on "extra", i.e. rank != 0, cpus

            my_results = self._compute_dproduct_cache(
                layout_atom_tree, prod_cache, scale_cache, None, myDerivColSlice, profiler)
            # pass None as comm, *not* mySubComm, since we can't do any
            #  further parallelization

            tm = _time.time()
            all_results = comm.allgather(my_results)
            profiler.add_time("MPI IPC", tm)
            return _np.concatenate(all_results, axis=1)  # TODO: remove this concat w/better gather?

        # ------------------------------------------------------------------
        tSerialStart = _time.time()
        dProdCache = _np.zeros((cacheSize,) + deriv_shape)
        wrtIndices = _slct.indices(wrt_slice) if (wrt_slice is not None) else None

        for iDest, iRight, iLeft in eval_tree:

            #Special case of an "initial operation" that can be filled directly
            if iRight is None:  # then iLeft gives operation:
                opLabel = iLeft
                if opLabel is None:
                    dProdCache[iDest] = _np.zeros(deriv_shape)
                else:
                    #doperation = self.dproduct( (opLabel,) , wrt_filter=wrtIndices)
                    doperation = self._doperation(opLabel, wrt_filter=wrtIndices)
                    dProdCache[iDest] = doperation / _np.exp(scale_cache[iDest])
                continue

            tm = _time.time()

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from eval_tree because
            # (iRight,iLeft,iFinal) = tup implies circuit[i] = circuit[iLeft] + circuit[iRight], but we want:
            # since then matrixOf(circuit[i]) = matrixOf(circuit[iLeft]) * matrixOf(circuit[iRight])
            L, R = prod_cache[iLeft], prod_cache[iRight]
            dL, dR = dProdCache[iLeft], dProdCache[iRight]
            dProdCache[iDest] = _np.dot(dL, R) + \
                _np.swapaxes(_np.dot(L, dR), 0, 1)  # dot(dS, T) + dot(S, dT)
            profiler.add_time("compute_dproduct_cache: dots", tm)
            profiler.add_count("compute_dproduct_cache: dots")

            scale = scale_cache[iDest] - (scale_cache[iLeft] + scale_cache[iRight])
            if abs(scale) > 1e-8:  # _np.isclose(scale,0) is SLOW!
                dProdCache[iDest] /= _np.exp(scale)
                if dProdCache[iDest].max() < _DSMALL and dProdCache[iDest].min() > -_DSMALL:
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
            elif (_np.count_nonzero(dProdCache[iDest]) and dProdCache[iDest].max() < _DSMALL
                  and dProdCache[iDest].min() > -_DSMALL):
                _warnings.warn("Would have scaled dProd but now will not alter scale_cache.")

        #profiler.print_mem("DEBUGMEM: POINT2"); profiler.comm.barrier()

        profiler.add_time("compute_dproduct_cache: serial", tSerialStart)
        profiler.add_count("compute_dproduct_cache: num columns", nDerivCols)

        return dProdCache

    def _compute_hproduct_cache(self, layout_atom_tree, prod_cache, d_prod_cache1,
                                d_prod_cache2, scale_cache, comm=None,
                                wrt_slice1=None, wrt_slice2=None):
        """
        Computes a tree of product 2nd derivatives in a linear cache space. Will
        use derivative rows and columns to parallelize computation.
        """

        dim = self.model.dim

        # Note: dProdCache?.shape = (#circuits,#params_to_diff_wrt,dim,dim)
        nDerivCols1 = d_prod_cache1.shape[1]
        nDerivCols2 = d_prod_cache2.shape[1]
        assert(wrt_slice1 is None or _slct.length(wrt_slice1) == nDerivCols1)
        assert(wrt_slice2 is None or _slct.length(wrt_slice2) == nDerivCols2)
        hessn_shape = (nDerivCols1, nDerivCols2, dim, dim)
        eval_tree = layout_atom_tree
        cacheSize = len(eval_tree)

        # ------------------------------------------------------------------

        if comm is not None and comm.Get_size() > 1:
            # parallelize of deriv cols, then sub-trees (if available and necessary)

            if comm.Get_size() > nDerivCols1 * nDerivCols2:
                #If there are more processors than deriv cells, give a
                # warning -- note that we *cannot* make use of a tree being
                # split because there's no good way to reconstruct the
                # *non-final* parent-tree elements from those of the sub-trees.
                _warnings.warn("Increased speed could be obtained"
                               " by giving hproduct cache computation"
                               " *fewer* processors and *smaller* (sub-)tree"
                               " (e.g. by splitting tree beforehand), as there"
                               " are more cpus than hessian elements.")  # pragma: no cover

            # allocate final result memory
            hProdCache = _np.zeros((cacheSize,) + hessn_shape)

            # Use comm to distribute columns
            allDeriv1ColSlice = slice(0, nDerivCols1)
            allDeriv2ColSlice = slice(0, nDerivCols2)
            deriv1Slices, myDeriv1ColSlice, deriv1Owners, mySubComm = \
                _mpit.distribute_slice(allDeriv1ColSlice, comm)

            # Get slice into entire range of model params so that
            #  per-gate hessians can be computed properly
            if wrt_slice1 is not None and wrt_slice1.start is not None:
                myHessianSlice1 = _slct.shift(myDeriv1ColSlice, wrt_slice1.start)
            else: myHessianSlice1 = myDeriv1ColSlice

            #print("MPI: _compute_hproduct_cache over %d cols (rank %d computing %s)" \
            #    % (nDerivCols2, comm.Get_rank(), str(myDerivColSlice)))

            if mySubComm is not None and mySubComm.Get_size() > 1:
                deriv2Slices, myDeriv2ColSlice, deriv2Owners, mySubSubComm = \
                    _mpit.distribute_slice(allDeriv2ColSlice, mySubComm)

                # Get slice into entire range of model params (see above)
                if wrt_slice2 is not None and wrt_slice2.start is not None:
                    myHessianSlice2 = _slct.shift(myDeriv2ColSlice, wrt_slice2.start)
                else: myHessianSlice2 = myDeriv2ColSlice

                if mySubSubComm is not None and mySubSubComm.Get_size() > 1:
                    _warnings.warn("Too many processors to make use of in "
                                   " _compute_hproduct_cache.")
                    #TODO: remove: not needed now that we track owners
                    #if mySubSubComm.Get_rank() > 0: myDeriv2ColSlice = slice(0,0)
                    #  #don't compute anything on "extra", i.e. rank != 0, cpus

                hProdCache[:, myDeriv1ColSlice, myDeriv2ColSlice] = self._compute_hproduct_cache(
                    layout_atom_tree, prod_cache, d_prod_cache1[:, myDeriv1ColSlice],
                    d_prod_cache2[:, myDeriv2ColSlice], scale_cache, None, myHessianSlice1, myHessianSlice2)
                # pass None as comm, *not* mySubSubComm, since we can't do any further parallelization

                _mpit.gather_slices(deriv2Slices, deriv2Owners, hProdCache, [None, myDeriv1ColSlice],
                                    2, mySubComm)  # , gather_mem_limit) #gather over col-distribution (Deriv2)
                #note: gathering axis 2 of hProdCache[:,myDeriv1ColSlice],
                #      dim=(cacheSize,nDerivCols1,nDerivCols2,dim,dim)
            else:
                #compute "Deriv1" row-derivatives distribution only; don't use column distribution
                hProdCache[:, myDeriv1ColSlice] = self._compute_hproduct_cache(
                    layout_atom_tree, prod_cache, d_prod_cache1[:, myDeriv1ColSlice], d_prod_cache2,
                    scale_cache, None, myHessianSlice1, wrt_slice2)
                # pass None as comm, *not* mySubComm (this is ok, see "if" condition above)

            _mpit.gather_slices(deriv1Slices, deriv1Owners, hProdCache, [], 1, comm)
            #, gather_mem_limit) #gather over row-distribution (Deriv1)
            #note: gathering axis 1 of hProdCache,
            #      dim=(cacheSize,nDerivCols1,nDerivCols2,dim,dim)

            return hProdCache

        # ------------------------------------------------------------------

        hProdCache = _np.zeros((cacheSize,) + hessn_shape)
        wrtIndices1 = _slct.indices(wrt_slice1) if (wrt_slice1 is not None) else None
        wrtIndices2 = _slct.indices(wrt_slice2) if (wrt_slice2 is not None) else None

        for iDest, iRight, iLeft in eval_tree:

            #Special case of an "initial operation" that can be filled directly
            if iRight is None:  # then iLeft gives operation:
                opLabel = iLeft
                if opLabel is None:
                    hProdCache[iDest] = _np.zeros(hessn_shape)
                elif not self.model.circuit_layer_operator(opLabel, 'op').has_nonzero_hessian():
                    #all gate elements are at most linear in params, so
                    # all hessians for single- or zero-circuits are zero.
                    hProdCache[iDest] = _np.zeros(hessn_shape)
                else:
                    hoperation = self._hoperation(opLabel,
                                                  wrt_filter1=wrtIndices1,
                                                  wrt_filter2=wrtIndices2)
                    hProdCache[iDest] = hoperation / _np.exp(scale_cache[iDest])
                continue

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from eval_tree because
            # (Dest,iLeft,iRight,iFinal) = tup implies circuit[iDest] = circuit[iLeft] + circuit[iRight], but we want:
            # since then matrixOf(circuit[i]) = matrixOf(circuit[iLeft]) * matrixOf(circuit[iRight])
            L, R = prod_cache[iLeft], prod_cache[iRight]
            dL1, dR1 = d_prod_cache1[iLeft], d_prod_cache1[iRight]
            dL2, dR2 = d_prod_cache2[iLeft], d_prod_cache2[iRight]
            hL, hR = hProdCache[iLeft], hProdCache[iRight]
            # Note: L, R = GxG ; dL,dR = vgs x GxG ; hL,hR = vgs x vgs x GxG

            dLdRa = _np.swapaxes(_np.dot(dL1, dR2), 1, 2)
            dLdRb = _np.swapaxes(_np.dot(dL2, dR1), 1, 2)
            dLdR_sym = dLdRa + _np.swapaxes(dLdRb, 0, 1)

            hProdCache[iDest] = _np.dot(hL, R) + dLdR_sym + _np.transpose(_np.dot(L, hR), (1, 2, 0, 3))

            scale = scale_cache[iDest] - (scale_cache[iLeft] + scale_cache[iRight])
            if abs(scale) > 1e-8:  # _np.isclose(scale,0) is SLOW!
                hProdCache[iDest] /= _np.exp(scale)
                if hProdCache[iDest].max() < _HSMALL and hProdCache[iDest].min() > -_HSMALL:
                    _warnings.warn("Scaled hProd small in order to keep prod managable.")
            elif (_np.count_nonzero(hProdCache[iDest]) and hProdCache[iDest].max() < _HSMALL
                  and hProdCache[iDest].min() > -_HSMALL):
                _warnings.warn("hProd is small (oh well!).")

        return hProdCache

    def create_layout(self, circuits, dataset=None, resource_alloc=None, array_types=('E',),
                      derivative_dimension=None, verbosity=0):

        # Let np = # param groups, so 1 <= np <= num_params, size of each param group = num_params/np
        # Let nc = # circuit groups == # atoms, so 1 <= nc <= max_split_num; size of each group = size of
        #          corresponding atom
        # With nprocs processors, split into Ng comms of ~nprocs/Ng procs each.  These comms are each assigned some
        #  number of circuit groups, where their ~nprocs/Ng processors are used to partition the np param
        #  groups. Note that 1 <= Ng <= min(nc,nprocs).
        # Notes:
        #  - making np or nc > nprocs can be useful for saving memory.  Raising np saves *Jacobian* and *Hessian*
        #     function memory without layout overhead, and I think will typically be preferred over raising nc.
        #     Raising nc will additionally save *Probability/Product* function memory, but will incur layout overhead.
        #  - any given CPU will be running a *single* (nc-index,np-index) pair at any given time, and so memory
        #     estimates only depend on nc and np, and not on Ng, *except* when an array is *gathered* from the
        #     end results from a divided computation.
        #  - "circuits" distribute_method: never distribute num_params (np == 1, Ng == nprocs always).
        #     Choose nc such that nc >= nprocs, mem_estimate(nc,np=1) < mem_limit, and nc % nprocs == 0 (nc % Ng == 0).
        #  - "deriv" distribute_method: if possible, set nc=1, nprocs <= np <= num_params, Ng = 1 (np % nprocs == 0?)
        #     If memory constraints don't allow this, set np = num_params, Ng ~= nprocs/num_params (but Ng >= 1),
        #     and nc set by mem_estimate and nc % Ng == 0 (so comms are kept busy)
        #
        # find nc, np, Ng such that:
        # - mem_estimate(nc,np,Ng) < mem_limit
        # - full cpu usage:
        #   - np*nc >= nprocs (all procs used)
        #   - nc % Ng == 0 (each proc has the same number of atoms, so they're kept busy)
        # -nice, but not essential:
        #   - num_params % np == 0 (each param group has same size)
        #   - np % (nprocs/Ng) == 0 would be nice (all procs have same num of param groups to process)

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        comm = resource_alloc.comm
        mem_limit = resource_alloc.mem_limit - resource_alloc.allocated_memory \
            if (resource_alloc.mem_limit is not None) else None  # *per-processor* memory limit
        printer = _VerbosityPrinter.create_printer(verbosity, comm)
        nprocs = 1 if comm is None else comm.Get_size()
        num_params = derivative_dimension if (derivative_dimension is not None) else self.model.num_params
        num_circuits = len(circuits)
        C = 1.0 / (1024.0**3)

        if mem_limit is not None:
            if mem_limit <= 0:
                raise MemoryError("Attempted layout creation w/memory limit = %g <= 0!" % mem_limit)
            printer.log("Layout creation w/mem limit = %.2fGB" % (mem_limit * C))

        if self._mode == "distribute_by_timestamp":
            #Special case: time dependent data that gets grouped & distributed by unique timestamp
            layout = _MatrixTimeDepCOPALayout(circuits, self.model, dataset,
                                              (num_params, num_params), verbosity)
            layout.set_distribution_params(nprocs, (num_params, num_params),
                                           mem_limit * 0.01 if (mem_limit is not None) else None)
            return layout

        def create_layout_candidate(num_atoms):
            return _MatrixCOPALayout(circuits, self.model, dataset, None, num_atoms,
                                     (num_params, num_params), verbosity)

        bNp1Matters = bool("EP" in array_types or "EPP" in array_types)
        bNp2Matters = bool("EPP" in array_types)

        #Start with how we'd like to split processors up (without regard to memory limit):
        nc = Ng = 1
        if bNp2Matters:
            if nprocs > num_params**2:
                np1 = np2 = max(num_params, 1)
                nc = Ng = _mpit.processor_group_size(nprocs, nprocs / max(num_params**2, 1))  # float division
            elif nprocs > num_params:
                np1 = max(num_params, 1)
                np2 = int(_np.ceil(nprocs / max(num_params, 1)))
            else:
                np1 = nprocs; np2 = 1
        elif bNp1Matters:
            np2 = 1
            if nprocs > num_params:
                np1 = max(num_params, 1)
                nc = Ng = _mpit.processor_group_size(nprocs, nprocs / max(num_params, 1))
            else:
                np1 = nprocs
        else:
            np1 = np2 = 1
            nc = Ng = nprocs

        #Create initial layout, and get the "final memory" that is required to hold the final results
        # for each array type.  This amount doesn't depend on how the layout is "split" into atoms.

        # NOTE: This assumes "gather=True" mode, where all processors hold complete final arrays containing
        # *all* derivative columns. In gather=False mode, this amount would change based on the
        # memory of all the atoms and deriv columns assigned to a single processor. E.g. mayb code like:
        #    dist_info = trial_layout.distribution_info(nprocs)
        #    rank_total_els = [sum([trial_layout.atoms[i].num_elements for i in rank_info['atom_indices']])
        #                      for rank_info in dist_info.values()]  # total elements per processor

        layout_cache = {}  # cache of layout candidates indexed on # (minimal) atoms, to avoid re-computation
        layout_cache[nc] = create_layout_candidate(nc)
        num_elements = layout_cache[nc].num_elements  # should be fixed

        if mem_limit is not None:  # the hard case when there's a memory limit
            gather_mem_limit = mem_limit * 0.01  # better?

            def mem_estimate(nc, np1, np2):
                if nc not in layout_cache: layout_cache[nc] = create_layout_candidate(nc)
                layout = layout_cache[nc]
                return _bytes_for_array_types(array_types, layout.num_elements, layout.max_atom_elements,
                                              layout.num_circuits, num_circuits / nc,
                                              layout._additional_dimensions, (num_params / np1, num_params / np2),
                                              layout.max_atom_cachesize, self.model.dim)

            def approx_mem_estimate(nc, np1, np2):
                approx_cachesize = (num_circuits / nc) * 1.3  # inflate expected # of circuits per atom => cache_size
                return _bytes_for_array_types(array_types, num_elements, num_elements / nc,
                                              num_circuits, num_circuits / nc,
                                              (num_params, num_params), (num_params / np1, num_params / np2),
                                              approx_cachesize, self.model.dim)

            mem = mem_estimate(nc, np1, np2)  # initial estimate (to screen)
            #printer.log(f" mem({nc} atoms, {np1},{np2} param-grps, {Ng} proc-grps) = {mem * C}GB")
            printer.log(" mem(%d atoms, %d,%d param-grps, %d proc-grps) = %.2fGB" %
                        (nc, np1, np2, Ng, mem * C))

            #Now do (fast) memory checks that try to increase np1 and/or np2 if memory constraint is unmet.
            ok = False
            if (not ok) and bNp1Matters and np1 < num_params:
                #First try to decrease mem consumption by increasing np1
                for n in range(np1, num_params + 1, nprocs):
                    if mem_estimate(nc, n, np2) < mem_limit:
                        np1 = n; ok = True; break
                else: np1 = num_params

            if (not ok) and bNp2Matters and np2 < num_params:
                #Next try to decrease mem consumption by increasing np2
                for n in range(np2, num_params + 1, nprocs):
                    if mem_estimate(nc, np1, n) < mem_limit:
                        np2 = n; ok = True; break
                else: np2 = num_params

            #Finally, increase nc in amounts of Ng (so nc % Ng == 0).  Start
            # with fast cache_size computation then switch to slow
            if not ok:
                mem = approx_mem_estimate(nc, np1, np2)
                while mem > mem_limit:
                    nc += Ng; _next = mem_estimate(nc, np1, np2)
                    if _next >= mem: raise MemoryError("Not enough memory: increasing #atoms unproductive")
                    mem = _next

                mem = mem_estimate(nc, np1, np2)
                #printer.log(f" mem({nc} atoms, {np1},{np2} param-grps, {Ng} proc-grps) = {mem * C}GB")
                printer.log(" mem(%d atoms, %d,%d param-grps, %d proc-grps) ~= %.2fGB" %
                            (nc, np1, np2, Ng, mem * C))

                while mem > mem_limit:
                    nc += Ng; _next = mem_estimate(nc, np1, np2)
                    #printer.log((f" mem({nc} atoms, {np1},{np2} param-grps, {Ng} proc-grps) ="
                    #             f" {_next * C}GB"))
                    printer.log(" mem(%d atoms, %d,%d param-grps, %d proc-grps) = %.2fGB" %
                                (nc, np1, np2, Ng, _next * C))

                    if _next >= mem: raise MemoryError("Not enough memory: increasing #atoms unproductive")
                    mem = _next
        else:
            gather_mem_limit = None

        layout = layout_cache[nc]

        paramBlkSize1 = num_params / np1
        paramBlkSize2 = num_params / np2  # the *average* param block size
        # (in general *not* an integer), which ensures that the intended # of
        # param blocks is communicated to forwardsim routines (taking ceiling or
        # floor can lead to inefficient MPI distribution)

        nparams = (num_params, num_params) if bNp2Matters else num_params
        np = (np1, np2) if bNp2Matters else np1
        paramBlkSizes = (paramBlkSize1, paramBlkSize2) if bNp2Matters else paramBlkSize1
        #printer.log((f"Created matrix-sim layout for {len(circuits)} circuits over {nprocs} processors:\n"
        #             f" Layout comprised of {nc} atoms, processed in {Ng} groups of ~{nprocs // Ng} processors each.\n"
        #             f" {nparams} parameters divided into {np} blocks of ~{paramBlkSizes} params."))
        printer.log(("Created matrix-sim layout for %d circuits over %d processors:\n"
                     " Layout comprised of %d atoms, processed in %d groups of ~%d processors each.\n"
                     " %s parameters divided into %s blocks of ~%s params.") %
                    (len(circuits), nprocs, nc, Ng, nprocs // Ng, str(nparams), str(np), str(paramBlkSizes)))

        if np1 == 1:  # (paramBlkSize == num_params)
            paramBlkSize1 = None  # == all parameters, and may speed logic in dprobs, etc.
        else:
            if comm is not None:  # check that all procs have *same* paramBlkSize1
                blkSizeTest = comm.bcast(paramBlkSize1, root=0)
                assert(abs(blkSizeTest - paramBlkSize1) < 1e-3)

        if np2 == 1:  # (paramBlkSize == num_params)
            paramBlkSize2 = None  # == all parameters, and may speed logic in hprobs, etc.
        else:
            if comm is not None:  # check that all procs have *same* paramBlkSize2
                blkSizeTest = comm.bcast(paramBlkSize2, root=0)
                assert(abs(blkSizeTest - paramBlkSize2) < 1e-3)

        layout.set_distribution_params(Ng, (paramBlkSize1, paramBlkSize2), gather_mem_limit)
        return layout

    def _scale_exp(self, scale_exps):
        old_err = _np.seterr(over='ignore')
        scaleVals = _np.exp(scale_exps)  # may overflow, but OK if infs occur here
        _np.seterr(**old_err)
        return scaleVals

    def _rho_e_from_spam_tuple(self, spam_tuple):
        # This calculator uses the convention that rho has shape (N,1)
        rholabel, elabel = spam_tuple
        rho = self.model.circuit_layer_operator(rholabel, 'prep').to_dense()[:, None]
        E = _np.conjugate(_np.transpose(self.model.circuit_layer_operator(elabel, 'povm').to_dense()[:, None]))
        return rho, E

    def _probs_from_rho_e(self, rho, e, gs, scale_vals):
        if self.model.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")

        #Compute probability and save in return array
        # want vp[iFinal] = float(dot(e, dot(G, rho)))
        #  vp[i] = sum_k,l e[0,k] gs[i,k,l] rho[l,0] * scale_vals[i]
        #  vp[i] = sum_k e[0,k] dot(gs, rho)[i,k,0]  * scale_vals[i]
        #  vp[i] = dot( e, dot(gs, rho))[0,i,0]      * scale_vals[i]
        #  vp    = squeeze( dot( e, dot(gs, rho)), axis=(0,2) ) * scale_vals
        return _np.squeeze(_np.dot(e, _np.dot(gs, rho)), axis=(0, 2)) * scale_vals
        # shape == (len(circuit_list),) ; may overflow but OK

    def _dprobs_from_rho_e(self, spam_tuple, rho, e, gs, d_gs, scale_vals, wrt_slice=None):
        if self.model.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")

        rholabel, elabel = spam_tuple
        rhoVec = self.model.circuit_layer_operator(rholabel, 'prep')  # distinct from rho,e b/c rho,e are
        EVec = self.model.circuit_layer_operator(elabel, 'povm')   # arrays, these are SPAMVecs
        nCircuits = gs.shape[0]
        rho_wrtFilter, rho_gpindices = self._process_wrt_filter(
            wrt_slice, self.model.circuit_layer_operator(rholabel, 'prep'))
        E_wrtFilter, E_gpindices = self._process_wrt_filter(
            wrt_slice, self.model.circuit_layer_operator(elabel, 'povm'))
        nDerivCols = self.model.num_params if wrt_slice is None else _slct.length(wrt_slice)

        # GATE DERIVS (assume d_gs is already sized/filtered) -------------------
        assert(d_gs.shape[1] == nDerivCols), "d_gs must be pre-filtered!"

        #Compute d(probability)/dOps and save in return list (now have G,dG => product, dprod_dOps)
        #  prod, dprod_dOps = G,dG
        # dp_dOps[i,j] = sum_k,l e[0,k] d_gs[i,j,k,l] rho[l,0]
        # dp_dOps[i,j] = sum_k e[0,k] dot( d_gs, rho )[i,j,k,0]
        # dp_dOps[i,j] = dot( e, dot( d_gs, rho ) )[0,i,j,0]
        # dp_dOps      = squeeze( dot( e, dot( d_gs, rho ) ), axis=(0,3))
        old_err2 = _np.seterr(invalid='ignore', over='ignore')
        dp_dOps = _np.squeeze(_np.dot(e, _np.dot(d_gs, rho)), axis=(0, 3)) * scale_vals[:, None]
        _np.seterr(**old_err2)
        # may overflow, but OK ; shape == (len(circuit_list), nDerivCols)
        # may also give invalid value due to scale_vals being inf and dot-prod being 0. In
        #  this case set to zero since we can't tell whether it's + or - inf anyway...
        dp_dOps[_np.isnan(dp_dOps)] = 0

        #SPAM -------------

        # Get: dp_drhos[i, rho_gpindices] = dot(e,gs[i],drho/drhoP)
        # dp_drhos[i,J0+J] = sum_kl e[0,k] gs[i,k,l] drhoP[l,J]
        # dp_drhos[i,J0+J] = dot(e, gs, drhoP)[0,i,J]
        # dp_drhos[:,J0+J] = squeeze(dot(e, gs, drhoP),axis=(0,))[:,J]

        dp_drhos = _np.zeros((nCircuits, nDerivCols))
        _fas(dp_drhos, [None, rho_gpindices],
             _np.squeeze(_np.dot(_np.dot(e, gs),
                                 rhoVec.deriv_wrt_params(rho_wrtFilter)),
                         axis=(0,)) * scale_vals[:, None])  # may overflow, but OK

        # Get: dp_dEs[i, E_gpindices] = dot(transpose(dE/dEP),gs[i],rho))
        # dp_dEs[i,J0+J] = sum_lj dEPT[J,j] gs[i,j,l] rho[l,0]
        # dp_dEs[i,J0+J] = sum_j dEP[j,J] dot(gs, rho)[i,j]
        # dp_dEs[i,J0+J] = sum_j dot(gs, rho)[i,j,0] dEP[j,J]
        # dp_dEs[i,J0+J] = dot(squeeze(dot(gs, rho),2), dEP)[i,J]
        # dp_dEs[:,J0+J] = dot(squeeze(dot(gs, rho),axis=(2,)), dEP)[:,J]
        dp_dEs = _np.zeros((nCircuits, nDerivCols))
        # may overflow, but OK (deriv w.r.t any of self.effects - independent of which)
        dp_dAnyE = _np.squeeze(_np.dot(gs, rho), axis=(2,)) * scale_vals[:, None]
        _fas(dp_dEs, [None, E_gpindices],
             _np.dot(dp_dAnyE, EVec.deriv_wrt_params(E_wrtFilter)))

        sub_vdp = dp_drhos + dp_dEs + dp_dOps
        return sub_vdp

    def _hprobs_from_rho_e(self, spam_tuple, rho, e, gs, d_gs1, d_gs2, h_gs, scale_vals,
                           wrt_slice1=None, wrt_slice2=None):
        if self.model.evotype == "statevec": raise NotImplementedError("Unitary evolution not fully supported yet!")

        rholabel, elabel = spam_tuple
        rhoVec = self.model.circuit_layer_operator(rholabel, 'prep')  # distinct from rho,e b/c rho,e are
        EVec = self.model.circuit_layer_operator(elabel, 'povm')   # arrays, these are SPAMVecs
        nCircuits = gs.shape[0]

        rho_wrtFilter1, rho_gpindices1 = self._process_wrt_filter(
            wrt_slice1, self.model.circuit_layer_operator(rholabel, 'prep'))
        rho_wrtFilter2, rho_gpindices2 = self._process_wrt_filter(
            wrt_slice2, self.model.circuit_layer_operator(rholabel, 'prep'))
        E_wrtFilter1, E_gpindices1 = self._process_wrt_filter(
            wrt_slice1, self.model.circuit_layer_operator(elabel, 'povm'))
        E_wrtFilter2, E_gpindices2 = self._process_wrt_filter(
            wrt_slice2, self.model.circuit_layer_operator(elabel, 'povm'))

        nDerivCols1 = self.model.num_params if wrt_slice1 is None else _slct.length(wrt_slice1)
        nDerivCols2 = self.model.num_params if wrt_slice2 is None else _slct.length(wrt_slice2)

        #flt1 = self._get_filter_info(wrtSlices1)
        #flt2 = self._get_filter_info(wrtSlices2)

        # GATE DERIVS (assume h_gs is already sized/filtered) -------------------
        assert(h_gs.shape[1] == nDerivCols1), "h_gs must be pre-filtered!"
        assert(h_gs.shape[2] == nDerivCols2), "h_gs must be pre-filtered!"

        #Compute d2(probability)/dGates2 and save in return list
        # d2pr_dOps2[i,j,k] = sum_l,m e[0,l] h_gs[i,j,k,l,m] rho[m,0]
        # d2pr_dOps2[i,j,k] = sum_l e[0,l] dot( d_gs, rho )[i,j,k,l,0]
        # d2pr_dOps2[i,j,k] = dot( e, dot( d_gs, rho ) )[0,i,j,k,0]
        # d2pr_dOps2        = squeeze( dot( e, dot( d_gs, rho ) ), axis=(0,4))
        old_err2 = _np.seterr(invalid='ignore', over='ignore')
        d2pr_dOps2 = _np.squeeze(_np.dot(e, _np.dot(h_gs, rho)), axis=(0, 4)) * scale_vals[:, None, None]
        _np.seterr(**old_err2)

        # may overflow, but OK ; shape == (len(circuit_list), nDerivCols, nDerivCols)
        # may also give invalid value due to scale_vals being inf and dot-prod being 0. In
        #  this case set to zero since we can't tell whether it's + or - inf anyway...
        d2pr_dOps2[_np.isnan(d2pr_dOps2)] = 0

        # SPAM DERIVS (assume d_gs1 and d_gs2 are already sized/filtered) --------
        assert(d_gs1.shape[1] == nDerivCols1), "d_gs1 must be pre-filtered!"
        assert(d_gs2.shape[1] == nDerivCols2), "d_gs1 must be pre-filtered!"

        # Get: d2pr_drhos[i, j, rho_gpindices] = dot(e,d_gs[i,j],drho/drhoP))
        # d2pr_drhos[i,j,J0+J] = sum_kl e[0,k] d_gs[i,j,k,l] drhoP[l,J]
        # d2pr_drhos[i,j,J0+J] = dot(e, d_gs, drhoP)[0,i,j,J]
        # d2pr_drhos[:,:,J0+J] = squeeze(dot(e, d_gs, drhoP),axis=(0,))[:,:,J]
        drho = rhoVec.deriv_wrt_params(rho_wrtFilter2)
        d2pr_drhos1 = _np.zeros((nCircuits, nDerivCols1, nDerivCols2))
        _fas(d2pr_drhos1, [None, None, rho_gpindices2],
             _np.squeeze(_np.dot(_np.dot(e, d_gs1), drho), axis=(0,))
             * scale_vals[:, None, None])  # overflow OK

        # get d2pr_drhos where gate derivatives are wrt the 2nd set of gate parameters
        if d_gs1 is d_gs2 and wrt_slice1 == wrt_slice2:  # TODO: better check for equivalence: maybe let d_gs2 be None?
            assert(nDerivCols1 == nDerivCols2)
            d2pr_drhos2 = _np.transpose(d2pr_drhos1, (0, 2, 1))
        else:
            drho = rhoVec.deriv_wrt_params(rho_wrtFilter1)
            d2pr_drhos2 = _np.zeros((nCircuits, nDerivCols2, nDerivCols1))
            _fas(d2pr_drhos2, [None, None, rho_gpindices1],
                 _np.squeeze(_np.dot(_np.dot(e, d_gs2), drho), axis=(0,))
                 * scale_vals[:, None, None])  # overflow OK
            d2pr_drhos2 = _np.transpose(d2pr_drhos2, (0, 2, 1))

        # Get: d2pr_dEs[i, j, E_gpindices] = dot(transpose(dE/dEP),d_gs[i,j],rho)
        # d2pr_dEs[i,j,J0+J] = sum_kl dEPT[J,k] d_gs[i,j,k,l] rho[l,0]
        # d2pr_dEs[i,j,J0+J] = sum_k dEP[k,J] dot(d_gs, rho)[i,j,k,0]
        # d2pr_dEs[i,j,J0+J] = dot( squeeze(dot(d_gs, rho),axis=(3,)), dEP)[i,j,J]
        # d2pr_dEs[:,:,J0+J] = dot( squeeze(dot(d_gs, rho),axis=(3,)), dEP)[:,:,J]
        d2pr_dEs1 = _np.zeros((nCircuits, nDerivCols1, nDerivCols2))
        dp_dAnyE = _np.squeeze(_np.dot(d_gs1, rho), axis=(3,)) * scale_vals[:, None, None]  # overflow OK
        devec = EVec.deriv_wrt_params(E_wrtFilter2)
        _fas(d2pr_dEs1, [None, None, E_gpindices2],
             _np.dot(dp_dAnyE, devec))

        # get d2pr_dEs where gate derivatives are wrt the 2nd set of gate parameters
        if d_gs1 is d_gs2 and wrt_slice1 == wrt_slice2:  # TODO: better check for equivalence: maybe let d_gs2 be None?
            assert(nDerivCols1 == nDerivCols2)
            d2pr_dEs2 = _np.transpose(d2pr_dEs1, (0, 2, 1))
        else:
            d2pr_dEs2 = _np.zeros((nCircuits, nDerivCols2, nDerivCols1))
            dp_dAnyE = _np.squeeze(_np.dot(d_gs2, rho), axis=(3,)) * scale_vals[:, None, None]  # overflow OK
            devec = EVec.deriv_wrt_params(E_wrtFilter1)
            _fas(d2pr_dEs2, [None, None, E_gpindices1], _np.dot(dp_dAnyE, devec))
            d2pr_dEs2 = _np.transpose(d2pr_dEs2, (0, 2, 1))

        # Get: d2pr_dErhos[i, e_offset[eIndex]:e_offset[eIndex+1], e_offset[rhoIndex]:e_offset[rhoIndex+1]] =
        #    dEP^T * prod[i,:,:] * drhoP
        # d2pr_dErhos[i,J0+J,K0+K] = sum jk dEPT[J,j] prod[i,j,k] drhoP[k,K]
        # d2pr_dErhos[i,J0+J,K0+K] = sum j dEPT[J,j] dot(prod,drhoP)[i,j,K]
        # d2pr_dErhos[i,J0+J,K0+K] = dot(dEPT,prod,drhoP)[J,i,K]
        # d2pr_dErhos[i,J0+J,K0+K] = swapaxes(dot(dEPT,prod,drhoP),0,1)[i,J,K]
        # d2pr_dErhos[:,J0+J,K0+K] = swapaxes(dot(dEPT,prod,drhoP),0,1)[:,J,K]
        d2pr_dErhos1 = _np.zeros((nCircuits, nDerivCols1, nDerivCols2))
        drho = rhoVec.deriv_wrt_params(rho_wrtFilter2)
        dp_dAnyE = _np.dot(gs, drho) * scale_vals[:, None, None]  # overflow OK
        devec = EVec.deriv_wrt_params(E_wrtFilter1)
        _fas(d2pr_dErhos1, (None, E_gpindices1, rho_gpindices2),
             _np.swapaxes(_np.dot(_np.transpose(devec), dp_dAnyE), 0, 1))

        # get d2pr_dEs where e derivatives are wrt the 2nd set of gate parameters
        if wrt_slice1 == wrt_slice2:  # Note: this doesn't involve gate derivatives
            d2pr_dErhos2 = _np.transpose(d2pr_dErhos1, (0, 2, 1))
        else:
            d2pr_dErhos2 = _np.zeros((nCircuits, nDerivCols2, nDerivCols1))
            drho = rhoVec.deriv_wrt_params(rho_wrtFilter1)
            dp_dAnyE = _np.dot(gs, drho) * scale_vals[:, None, None]  # overflow OK
            devec = EVec.deriv_wrt_params(E_wrtFilter2)
            _fas(d2pr_dErhos2, [None, E_gpindices2, rho_gpindices1],
                 _np.swapaxes(_np.dot(_np.transpose(devec), dp_dAnyE), 0, 1))
            d2pr_dErhos2 = _np.transpose(d2pr_dErhos2, (0, 2, 1))

        #Note: these 2nd derivatives are non-zero when the spam vectors have
        # a more than linear dependence on their parameters.
        if self.model.circuit_layer_operator(rholabel, 'prep').has_nonzero_hessian():
            dp_dAnyRho = _np.dot(e, gs).squeeze(0) * scale_vals[:, None]  # overflow OK
            d2pr_d2rhos = _np.zeros((nCircuits, nDerivCols1, nDerivCols2))
            _fas(d2pr_d2rhos, [None, rho_gpindices1, rho_gpindices2],
                 _np.tensordot(dp_dAnyRho, self.model.circuit_layer_operator(rholabel, 'prep').hessian_wrt_params(
                     rho_wrtFilter1, rho_wrtFilter2), (1, 0)))
            # _np.einsum('ij,jkl->ikl', dp_dAnyRho, self.model.circuit_layer_operator(rholabel, 'prep') \
            #    .hessian_wrt_params(rho_wrtFilter1, rho_wrtFilter2))
        else:
            d2pr_d2rhos = 0

        if self.model.circuit_layer_operator(elabel, 'povm').has_nonzero_hessian():
            dp_dAnyE = _np.dot(gs, rho).squeeze(2) * scale_vals[:, None]  # overflow OK
            d2pr_d2Es = _np.zeros((nCircuits, nDerivCols1, nDerivCols2))
            _fas(d2pr_d2Es, [None, E_gpindices1, E_gpindices2],
                 _np.tensordot(dp_dAnyE, self.model.circuit_layer_operator(elabel, 'povm').hessian_wrt_params(
                     E_wrtFilter1, E_wrtFilter2), (1, 0)))
            # _np.einsum('ij,jkl->ikl', dp_dAnyE, self.model.circuit_layer_operator(elabel, 'povm').hessian_wrt_params(
            #    E_wrtFilter1, E_wrtFilter2))
        else:
            d2pr_d2Es = 0

        # END SPAM DERIVS -----------------------

        ret = d2pr_d2rhos + d2pr_dErhos2 + d2pr_drhos2    # wrt rho
        ret += d2pr_dErhos1 + d2pr_d2Es + d2pr_dEs2      # wrt e
        ret += d2pr_drhos1 + d2pr_dEs1 + d2pr_dOps2   # wrt gates

        return ret

    def _bulk_fill_probs_block(self, array_to_fill, layout_atom, resource_alloc):
        # Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller
        #Free memory from previous subtree iteration before computing caches
        scaleVals = Gs = prodCache = scaleCache = None
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim * self.model.dim)  # prod cache

        #Fill cache info
        prodCache, scaleCache = self._compute_product_cache(layout_atom.tree, resource_alloc.comm)

        #use cached data to final values
        scaleVals = self._scale_exp(layout_atom.nonscratch_cache_view(scaleCache))
        Gs = layout_atom.nonscratch_cache_view(prodCache, axis=0)
        # ( n_circuits, dim, dim )

        old_err = _np.seterr(over='ignore')
        for spam_tuple, (element_indices, tree_indices) in layout_atom.indices_by_spamtuple.items():
            # "element indices" index a circuit outcome probability in array_to_fill's first dimension
            # "tree indices" index a quantity for a no-spam circuit in a computed cache, which correspond
            #  to the the element indices when `spamtuple` is used.
            rho, E = self._rho_e_from_spam_tuple(spam_tuple)
            _fas(array_to_fill, [element_indices],
                 self._probs_from_rho_e(rho, E, Gs[tree_indices], scaleVals[tree_indices]))
        _np.seterr(**old_err)

    def _bulk_fill_dprobs_block(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        dim = self.model.dim
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * dim * dim * self.model.num_params)
        prodCache, scaleCache = self._compute_product_cache(layout_atom.tree, resource_alloc.comm)
        dProdCache = self._compute_dproduct_cache(layout_atom.tree, prodCache, scaleCache,
                                                  resource_alloc.comm, param_slice)

        scaleVals = self._scale_exp(layout_atom.nonscratch_cache_view(scaleCache))
        Gs = layout_atom.nonscratch_cache_view(prodCache, axis=0)
        dGs = layout_atom.nonscratch_cache_view(dProdCache, axis=0)

        old_err = _np.seterr(over='ignore')
        for spam_tuple, (element_indices, tree_indices) in layout_atom.indices_by_spamtuple.items():
            rho, E = self._rho_e_from_spam_tuple(spam_tuple)
            _fas(array_to_fill, [element_indices, dest_param_slice], self._dprobs_from_rho_e(
                spam_tuple, rho, E, Gs[tree_indices], dGs[tree_indices], scaleVals[tree_indices], param_slice))

        _np.seterr(**old_err)

    def _bulk_fill_hprobs_block(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                                param_slice1, param_slice2, resource_alloc):
        dim = self.model.dim
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * dim**2 * self.model.num_params**2)
        prodCache, scaleCache = self._compute_product_cache(layout_atom.tree, resource_alloc.comm)
        dProdCache1 = self._compute_dproduct_cache(
            layout_atom.tree, prodCache, scaleCache, resource_alloc.comm, param_slice1)
        dProdCache2 = dProdCache1 if (param_slice1 == param_slice2) else \
            self._compute_dproduct_cache(layout_atom.tree, prodCache, scaleCache,
                                         resource_alloc.comm, param_slice2)

        scaleVals = self._scale_exp(layout_atom.nonscratch_cache_view(scaleCache))
        Gs = layout_atom.nonscratch_cache_view(prodCache, axis=0)
        dGs1 = layout_atom.nonscratch_cache_view(dProdCache1, axis=0)
        dGs2 = layout_atom.nonscratch_cache_view(dProdCache2, axis=0)
        #( n_circuits, nDerivColsX, dim, dim )

        hProdCache = self._compute_hproduct_cache(layout_atom.tree, prodCache, dProdCache1,
                                                  dProdCache2, scaleCache, resource_alloc.comm,
                                                  param_slice1, param_slice2)
        hGs = layout_atom.nonscratch_cache_view(hProdCache, axis=0)
        #( n_circuits, len(wrt_filter1), len(wrt_filter2), dim, dim )

        old_err = _np.seterr(over='ignore')
        for spam_tuple, (element_indices, tree_indices) in layout_atom.indices_by_spamtuple.items():
            rho, E = self._rho_e_from_spam_tuple(spam_tuple)
            _fas(array_to_fill, [element_indices, dest_param_slice1, dest_param_slice2], self._hprobs_from_rho_e(
                spam_tuple, rho, E, Gs[tree_indices], dGs1[tree_indices], dGs2[tree_indices],
                hGs[tree_indices], scaleVals[tree_indices], param_slice1, param_slice2))

        _np.seterr(**old_err)

    def bulk_product(self, circuits, scale=False, resource_alloc=None):
        """
        Compute the products of many circuits at once.

        Parameters
        ----------
        circuits : list of Circuits
            The circuits to compute products for.  These should *not* have any preparation or
            measurement layers.

        scale : bool, optional
            When True, return a scaling factor (see below).

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        Returns
        -------
        prods : numpy array
            Array of shape S x G x G, where:
            - S == the number of operation sequences
            - G == the linear dimension of a operation matrix (G x G operation matrices).
        scaleValues : numpy array
            Only returned when scale == True. A length-S array specifying
            the scaling that needs to be applied to the resulting products
            (final_product[i] = scaleValues[i] * prods[i]).
        """
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        nCircuits = len(circuits)

        eval_tree = _EvalTree.create(circuits)
        prodCache, scaleCache = self._compute_product_cache(eval_tree, resource_alloc.comm)

        # EvalTree evaluates a "cache" which can contain additional (intermediate) elements
        scaleVals = self._scale_exp(scaleCache[0:nCircuits])
        Gs = prodCache[0:nCircuits]

        if scale:
            return Gs, scaleVals
        else:
            old_err = _np.seterr(over='ignore')
            Gs = _np.swapaxes(_np.swapaxes(Gs, 0, 2) * scaleVals, 0, 2)  # may overflow, but ok
            _np.seterr(**old_err)
            return Gs

    def bulk_dproduct(self, circuits, flat=False, return_prods=False,
                      scale=False, resource_alloc=None, wrt_filter=None):
        """
        Compute the derivative of a many operation sequences at once.

        Parameters
        ----------
        circuits : list of Circuits
            The circuits to compute products for.  These should *not* have any preparation or
            measurement layers.

        flat : bool, optional
            Affects the shape of the returned derivative array (see below).

        return_prods : bool, optional
            when set to True, additionally return the probabilities.

        scale : bool, optional
            When True, return a scaling factor (see below).

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which gate parameters
            to include in the derivative.  Each element is an index into an
            array of gate parameters ordered by concatenating each gate's
            parameters (in the order specified by the model).  This argument
            is used internally for distributing derivative calculations across
            multiple processors.

        Returns
        -------
        derivs : numpy array
            * if flat == False, an array of shape S x M x G x G, where:
              - S == len(circuits)
              - M == the length of the vectorized model
              - G == the linear dimension of a operation matrix (G x G operation matrices)
              and derivs[i,j,k,l] holds the derivative of the (k,l)-th entry
              of the i-th operation sequence product with respect to the j-th model
              parameter.
            * if flat == True, an array of shape S*N x M where:
              - N == the number of entries in a single flattened gate (ordering same as numpy.flatten),
              - S,M == as above,
              and deriv[i,j] holds the derivative of the (i % G^2)-th entry of
              the (i / G^2)-th flattened operation sequence product  with respect to
              the j-th model parameter.
        products : numpy array
            Only returned when return_prods == True.  An array of shape
            S x G x G; products[i] is the i-th operation sequence product.
        scaleVals : numpy array
            Only returned when scale == True.  An array of shape S such that
            scaleVals[i] contains the multiplicative scaling needed for
            the derivatives and/or products for the i-th operation sequence.
        """
        nCircuits = len(circuits)
        nDerivCols = self.model.num_params if (wrt_filter is None) else _slct.length(wrt_filter)

        wrtSlice = _slct.list_to_slice(wrt_filter) if (wrt_filter is not None) else None
        #TODO: just allow slices as argument: wrt_filter -> wrtSlice?

        resource_alloc = _ResourceAllocation.cast(resource_alloc)

        eval_tree = _EvalTree.create(circuits)
        prodCache, scaleCache = self._compute_product_cache(eval_tree, resource_alloc.comm)
        dProdCache = self._compute_dproduct_cache(eval_tree, prodCache, scaleCache,
                                                  resource_alloc.comm, wrtSlice)

        # EvalTree evaluates a "cache" which can contain additional (intermediate) elements
        scaleVals = self._scale_exp(scaleCache[0:nCircuits])
        Gs = prodCache[0:nCircuits]
        dGs = dProdCache[0:nCircuits]

        if not scale:
            old_err = _np.seterr(over='ignore', invalid='ignore')
            if return_prods:
                Gs = _np.swapaxes(_np.swapaxes(Gs, 0, 2) * scaleVals, 0, 2)  # may overflow, but ok

            # may overflow or get nans (invalid), but ok
            dGs = _np.swapaxes(_np.swapaxes(dGs, 0, 3) * scaleVals, 0, 3)
            # convert nans to zero, as these occur b/c an inf scaleVal is mult by a zero deriv value, and we
            dGs[_np.isnan(dGs)] = 0
            _np.seterr(**old_err)

        if flat:
            # cols = deriv cols, rows = flattened everything else
            dGs = _np.swapaxes(_np.swapaxes(dGs, 0, 1).reshape(
                (nDerivCols, nCircuits * self.model.dim**2)), 0, 1)

        if return_prods:
            return (dGs, Gs, scaleVals) if scale else (dGs, Gs)
        else:
            return (dGs, scaleVals) if scale else dGs

    ## ---------------------------------------------------------------------------------------------
    ## TIME DEPENDENT functionality ----------------------------------------------------------------
    ## ---------------------------------------------------------------------------------------------

    def _ds_quantities(self, timestamp, ds_cache, layout, dataset, TIMETOL=1e-6):
        if timestamp not in ds_cache:
            if 'truncated_ds' not in ds_cache:
                ds_cache['truncated_ds'] = dataset.truncate(layout.circuits)
            trunc_dataset = ds_cache['truncated_ds']

            if 'ds_for_time' not in ds_cache:
                #tStart = _time.time()
                ds_cache['ds_for_time'] = trunc_dataset.split_by_time()
                #print("DB: Split dataset by time in %.1fs (%d timestamps)" % (_time.time() - tStart,
                #                                                              len(ds_cache['ds_for_time'])))

            if timestamp not in ds_cache['ds_for_time']:
                return (None, None, None, None, None)

            #Similar to MDC store's add_count_vectors function -- maybe consolidate in FUTURE?
            counts = _np.empty(layout.num_elements, 'd')
            totals = _np.empty(layout.num_elements, 'd')
            dataset_at_t = ds_cache['ds_for_time'][timestamp]  # trunc_dataset.time_slice(timestamp, timestamp+TIMETOL)

            #DEBUG CHECK REMOVE
            #tStart = _time.time()
            #dataset_at_t_check = trunc_dataset.time_slice(timestamp, timestamp+TIMETOL)
            #print("DB: Took time slice at t=%g in %.1fs" % (timestamp, _time.time() - tStart))
            #for c in dataset_at_t:
            #    if(dataset_at_t[c].counts != dataset_at_t_check[c].counts):
            #        import bpdb; bpdb.set_trace()

            firsts = []; indicesOfCircuitsWithOmittedData = []
            for (i, circuit) in enumerate(layout.circuits):  # should be 'ds_circuits' really
                inds = layout.indices_for_index(i)
                if circuit in dataset_at_t:
                    cnts = dataset_at_t[circuit].counts
                else:
                    cnts = {}  # Note: this will cause 0 totals, which will need to be handled downstream
                totals[inds] = sum(cnts.values())  # dataset[opStr].total
                counts[inds] = [cnts.get(x, 0) for x in layout.outcomes_for_index(i)]
                lklen = _slct.length(inds)  # consolidate w/ `add_omitted_freqs`?
                if 0 < lklen < self.model.compute_num_outcomes(circuit):
                    firsts.append(_slct.to_array(inds)[0])
                    indicesOfCircuitsWithOmittedData.append(i)

            if len(firsts) > 0:
                firsts = _np.array(firsts, 'i')
                indicesOfCircuitsWithOmittedData = _np.array(indicesOfCircuitsWithOmittedData, 'i')
                #print("DB: SPARSE DATA: %d of %d rows have sparse data" % (len(firsts), len(layout.circuits)))
            else:
                firsts = indicesOfCircuitsWithOmittedData = None

            #if self.circuits.circuit_weights is not None:
            #  SEE add_count_vectors

            nonzero_totals = _np.where(_np.abs(totals) < 1e-10, 1e-10, totals)  # avoid divide-by-zero error on nxt line
            freqs = counts / nonzero_totals
            ds_cache[timestamp] = (counts, totals, freqs, firsts, indicesOfCircuitsWithOmittedData)

        return ds_cache[timestamp]

    def _bulk_fill_timedep_objfn(self, raw_objective, array_to_fill, layout, ds_circuits,
                                 num_total_outcomes, dataset, resource_alloc=None, ds_cache=None):

        assert(self._mode == "distribute_by_timestamp"), \
            ("Must set `distribute_by_timestamp=True` to use a "
             "time-dependent objective function with MatrixForwardSimulator!")

        probs_array = _np.empty(layout.num_elements, 'd')
        array_to_fill[:] = 0.0

        def compute_timedep(layout_atom, sub_resource_alloc):
            # compute objective at time layout_atom.time
            for _, obj in self.model._iter_parameterized_objs():
                obj.set_time(layout_atom.timestamp)
            for opcache in self.model._opcaches.values():
                for obj in opcache.values():
                    obj.set_time(layout_atom.timestamp)

            self._bulk_fill_probs_block(probs_array, layout_atom, sub_resource_alloc)
            counts, totals, freqs, firsts, indicesOfCircuitsWithOmittedData = \
                self._ds_quantities(layout_atom.timestamp, ds_cache, layout, dataset)
            if counts is None: return  # no data at this time => no contribution

            terms = raw_objective.terms(probs_array, counts, totals, freqs)
            if firsts is not None:  # consolidate with `_update_terms_for_omitted_probs`
                omitted_probs = 1.0 - _np.array([_np.sum(probs_array[layout.indices_for_index(i)])
                                                 for i in indicesOfCircuitsWithOmittedData])
                terms[firsts] += raw_objective.zero_freq_terms(totals[firsts], omitted_probs)

            array_to_fill[layout_atom.element_slice] += terms

        atomOwners = self._compute_on_atoms(layout, compute_timedep, resource_alloc)

        #collect/gather results (SUM local arrays together)
        summed = _mpit.sum_arrays(array_to_fill, set(atomOwners.values()), resource_alloc.comm)
        array_to_fill[:] = summed  # this could probably be done more efficiently

    def _bulk_fill_timedep_dobjfn(self, raw_objective, array_to_fill, layout, ds_circuits,
                                  num_total_outcomes, dataset, resource_alloc=None, ds_cache=None):

        assert(self._mode == "distribute_by_timestamp"), \
            ("Must set `distribute_by_timestamp=True` to use a "
             "time-dependent objective function with MatrixForwardSimulator!")

        probs_array = _np.empty(layout.num_elements, 'd')
        dprobs_array = _np.empty((layout.num_elements, self.model.num_params), 'd')
        all_param_slice = slice(0, self.model.num_params)  # All params computed at once for now
        array_to_fill[:] = 0.0

        def compute_timedep(layout_atom, sub_resource_alloc):
            # compute objective at time layout_atom.time
            #print("DB: Rank %d : layout atom for t=" % resource_alloc.comm.rank, layout_atom.timestamp)
            for _, obj in self.model._iter_parameterized_objs():
                obj.set_time(layout_atom.timestamp)
            for opcache in self.model._opcaches.values():
                for obj in opcache.values():
                    obj.set_time(layout_atom.timestamp)

            self._bulk_fill_probs_block(probs_array, layout_atom, sub_resource_alloc)
            self._bulk_fill_dprobs_block(dprobs_array, all_param_slice, layout_atom,
                                         all_param_slice, sub_resource_alloc)

            #import bpdb; bpdb.set_trace()
            counts, totals, freqs, firsts, indicesOfCircuitsWithOmittedData = \
                self._ds_quantities(layout_atom.timestamp, ds_cache, layout, dataset)

            #Note: computing jacobian of objective here doesn't support sparse data yet!!! TODO
            if firsts is not None:  # consolidate with TimeIndependentMDCObjectiveFunction.dterms?
                dprobs_omitted_rowsum = _np.empty((len(firsts), self.model.num_params), 'd')
                for ii, i in enumerate(indicesOfCircuitsWithOmittedData):
                    dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs_array[layout.indices_for_index(i), :], axis=0)

            #print("Layout atom t=",layout_atom.timestamp)
            #print("p = ",probs_array)
            #print("dp[29] = ",dprobs_array[:,29])
            #print("cnts = ",counts)
            dprobs_array[:, :] = dprobs_array * raw_objective.dterms(probs_array, counts, totals, freqs)[:, None]
            #print("raw dterms = ",raw_objective.dterms(probs_array, counts, totals, freqs))
            #print("jac_contrib[:,29] = ",dprobs_array[:,29])

            if firsts is not None:  # consolidate with _update_dterms_for_omitted_probs?
                omitted_probs = 1.0 - _np.array([_np.sum(probs_array[layout.indices_for_index(i)])
                                                 for i in indicesOfCircuitsWithOmittedData])
                dprobs_array[firsts] -= raw_objective.zero_freq_dterms(totals[firsts], omitted_probs)[:, None] \
                    * dprobs_omitted_rowsum
            #print("jac_contrib[:,29] post = ",dprobs_array[:,29])

            array_to_fill[layout_atom.element_slice] += dprobs_array
            #print("array_to_fill[:,29] = ",array_to_fill[:,29], " sum = ",_np.sum(array_to_fill[:,29]))
            #print()

        atomOwners = self._compute_on_atoms(layout, compute_timedep, resource_alloc)

        #collect/gather results (SUM local arrays together)
        summed = _mpit.sum_arrays(array_to_fill, set(atomOwners.values()), resource_alloc.comm)
        array_to_fill[:] = summed  # this could probably be done more efficiently

    def bulk_fill_timedep_chi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                               min_prob_clip_for_weighting, prob_clip_interval, resource_alloc=None,
                               ds_cache=None):
        """
        Compute the chi2 contributions for an entire tree of circuits, allowing for time dependent operations.

        Computation is performed by summing together the contributions for each time the circuit is
        run, as given by the timestamps in `dataset`.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. layout.num_elements)

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the chi2 contributions.

        min_prob_clip_for_weighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: N/(p*(1-p)) by clipping probability p values to lie within
            the interval [ min_prob_clip_for_weighting, 1-min_prob_clip_for_weighting ].

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        raw_obj = _RawChi2Function({'min_prob_clip_for_weighting': min_prob_clip_for_weighting},
                                   resource_alloc)
        return self._bulk_fill_timedep_objfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                             dataset, resource_alloc, ds_cache)

    def bulk_fill_timedep_dchi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                min_prob_clip_for_weighting, prob_clip_interval, chi2_array_to_fill=None,
                                wrt_filter=None, resource_alloc=None, ds_cache=None):
        """
        Compute the chi2 jacobian contributions for an entire tree of circuits, allowing for time dependent operations.

        Similar to :method:`bulk_fill_timedep_chi2` but compute the *jacobian*
        of the summed chi2 contributions for each circuit with respect to the
        model's parameters.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. layout.num_elements) and M is the
            number of model parameters.

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the chi2 contributions.

        min_prob_clip_for_weighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: N/(p*(1-p)) by clipping probability p values to lie within
            the interval [ min_prob_clip_for_weighting, 1-min_prob_clip_for_weighting ].

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        chi2_array_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit chi2 contributions, just like in
            bulk_fill_timedep_chi2(...).

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        raw_obj = _RawChi2Function({'min_prob_clip_for_weighting': min_prob_clip_for_weighting},
                                   resource_alloc)
        return self._bulk_fill_timedep_dobjfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                              dataset, resource_alloc, ds_cache)

    def bulk_fill_timedep_loglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                 min_prob_clip, radius, prob_clip_interval, resource_alloc=None,
                                 ds_cache=None):
        """
        Compute the log-likelihood contributions (within the "poisson picture") for an entire tree of circuits.

        Computation is performed by summing together the contributions for each time the circuit is run,
        as given by the timestamps in `dataset`.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. layout.num_elements)

       layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the logl contributions.

        min_prob_clip : float, optional
            The minimum probability treated normally in the evaluation of the
            log-likelihood.  A penalty function replaces the true log-likelihood
            for probabilities that lie below this threshold so that the
            log-likelihood never becomes undefined (which improves optimizer
            performance).

        radius : float, optional
            Specifies the severity of rounding used to "patch" the
            zero-frequency terms of the log-likelihood.

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        raw_obj = _RawPoissonPicDeltaLogLFunction({'min_prob_clip': min_prob_clip, 'radius': radius},
                                                  resource_alloc)
        return self._bulk_fill_timedep_objfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                             dataset, resource_alloc, ds_cache)

    def bulk_fill_timedep_dloglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                  min_prob_clip, radius, prob_clip_interval, logl_array_to_fill=None,
                                  wrt_filter=None, resource_alloc=None, ds_cache=None):
        """
        Compute the ("poisson picture")log-likelihood jacobian contributions for an entire tree of circuits.

        Similar to :method:`bulk_fill_timedep_loglpp` but compute the *jacobian*
        of the summed logl (in posison picture) contributions for each circuit
        with respect to the model's parameters.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. layout.num_elements) and M is the
            number of model parameters.

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :method:`create_layout`.

        ds_circuits : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `layout` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(ds_circuits_to_use)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the logl contributions.

        min_prob_clip : float
            a regularization parameter for the log-likelihood objective function.

        radius : float
            a regularization parameter for the log-likelihood objective function.

        prob_clip_interval : 2-tuple or None, optional
            (min,max) values used to clip the predicted probabilities to.
            If None, no clipping is performed.

        logl_array_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit logl contributions, just like in
            bulk_fill_timedep_loglpp(...).

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        Returns
        -------
        None
        """
        raw_obj = _RawPoissonPicDeltaLogLFunction({'min_prob_clip': min_prob_clip, 'radius': radius},
                                                  resource_alloc)
        return self._bulk_fill_timedep_dobjfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                              dataset, resource_alloc, ds_cache)

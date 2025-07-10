"""
Defines the MatrixForwardSimulator calculator class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time
import collections as _collections
import time as _time
import warnings as _warnings

import numpy as _np
import numpy.linalg as _nla

from pygsti.forwardsims.distforwardsim import DistributableForwardSimulator as _DistributableForwardSimulator
from pygsti.forwardsims.forwardsim import ForwardSimulator as _ForwardSimulator
from pygsti.forwardsims.forwardsim import _bytes_for_array_types
from pygsti.layouts.evaltree import EvalTree as _EvalTree
from pygsti.layouts.evaltree import EvalTreeBasedUponLongestCommonSubstring as _EvalTreeLCS
from pygsti.layouts.evaltree import setup_circuit_list_for_LCS_computations, CollectionOfLCSEvalTrees
from pygsti.layouts.matrixlayout import MatrixCOPALayout as _MatrixCOPALayout
from pygsti.layouts.matrixlayout import _MatrixCOPALayoutAtomWithLCS
from pygsti.baseobjs.profiler import DummyProfiler as _DummyProfiler
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.tools import mpitools as _mpit
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct
from pygsti.tools.matrixtools import _fas
from pygsti.tools import listtools as _lt
from pygsti.circuits import CircuitList as _CircuitList
from pygsti.tools.internalgates import internal_gate_unitaries
from pygsti.tools.optools import unitary_to_superop
from pygsti.baseobjs.label import LabelTup, LabelTupTup


_dummy_profiler = _DummyProfiler()

# Smallness tolerances, used internally for conditional scaling required
# to control bulk products, their gradients, and their Hessians.
_PSMALL = 1e-100
_DSMALL = 1e-100
_HSMALL = 1e-100


class SimpleMatrixForwardSimulator(_ForwardSimulator):
    """
    A forward simulator that uses matrix-matrix products to compute circuit outcome probabilities.

    This is "simple" in that it adds a minimal implementation to its :class:`ForwardSimulator`
    base class.  Because of this, it lacks some of the efficiency of a :class:`MatrixForwardSimulator`
    object, and is mainly useful as a reference implementation and check for other simulators.
    """
    # NOTE: It is currently not a *distributed* forward simulator, but after the addition of
    # the `as_layout` method to distributed atoms, this class could instead derive from
    # DistributableForwardSimulator and (I think) not need any more implementation.
    # If this is done, then MatrixForwardSimulator wouldn't need to separately subclass DistributableForwardSimulator

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
            G = _np.identity(self.model.evotype.minimal_dim(self.model.state_space))
            for lOp in circuit:
                if lOp not in scaledGatesAndExps:
                    opmx = self.model.circuit_layer_operator(lOp, 'op').to_dense(on_space='minimal')
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
            G = _np.identity(self.model.evotype.minimal_dim(self.model.state_space))
            for lOp in circuit:
                G = _np.dot(self.model.circuit_layer_operator(lOp, 'op').to_dense(on_space='minimal'), G)
                # above line: LEXICOGRAPHICAL VS MATRIX ORDER
            return G

    def _rho_es_from_spam_tuples(self, rholabel, elabels):
        # This calculator uses the convention that rho has shape (N,1)
        rho = self.model.circuit_layer_operator(rholabel, 'prep').to_dense(on_space='minimal')[:, None]
        Es = [_np.conjugate(_np.transpose(self.model.circuit_layer_operator(
              elabel, 'povm').to_dense(on_space='minimal')[:, None]))
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

            if isinstance(obj.gpindices, slice):
                gpindices_list = _slct.indices(obj.gpindices)
            elif obj.gpindices is None:
                gpindices_list = []
            else:
                gpindices_list = list(obj.gpindices)
            gpindices_set = set(gpindices_list)

            for ii, i in enumerate(wrt_filter):
                if i in gpindices_set:
                    relevant_gpindices.append(ii)
                    obj_wrtFilter.append(gpindices_list.index(i))
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
        dim = self.model.evotype.minimal_dim(self.model.state_space)
        gate = self.model.circuit_layer_operator(op_label, 'op')

        # Allocate memory for the final result
        num_deriv_cols = self.model.num_params if (wrt_filter is None) else len(wrt_filter)
        #num_op_params = self.model._param_interposer.num_op_params \
        #    if (self.model._param_interposer is not None) else self.model.num_params

        #Note: deriv_wrt_params is more accurately deriv wrt *op* params when there is an interposer
        # d(op)/d(params) = d(op)/d(op_params) * d(op_params)/d(params)
        if self.model._param_interposer is not None:
            #When there is an interposer, we compute derivs wrt *all* the ops params (inefficient?),
            # then apply interposer, then take desired wrt_filter columns:
            assert(self.model._param_interposer.num_params == self.model.num_params)
            num_op_params = self.model._param_interposer.num_op_params
            deriv_wrt_op_params = _np.zeros((dim**2, num_op_params), 'd')
            deriv_wrt_op_params[:, gate.gpindices] = gate.deriv_wrt_params()  # *don't* apply wrt filter here
            deriv_wrt_params = _np.dot(deriv_wrt_op_params,
                                       self.model._param_interposer.deriv_op_params_wrt_model_params())

            # deriv_wrt_params is a derivative matrix with respect to *all* the model's parameters, so
            # now just take requested subset:
            flattened_dprod = deriv_wrt_params[:, wrt_filter] if (wrt_filter is not None) else deriv_wrt_params
        else:
            #Simpler case of no interposer: use _process_wrt_filter to "convert" from op params to model params
            # (the desired op params are just some subset, given by gpindices and op_wrtFilter, of the model parameters)
            flattened_dprod = _np.zeros((dim**2, num_deriv_cols), 'd')
            op_wrtFilter, gpindices = self._process_wrt_filter(wrt_filter, gate)

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
        if isinstance(wrt_filter1, slice): wrt_filter1 = _slct.indices(wrt_filter1)
        if isinstance(wrt_filter2, slice): wrt_filter2 = _slct.indices(wrt_filter2)

        dim = self.model.evotype.minimal_dim(self.model.state_space)
        gate = self.model.circuit_layer_operator(op_label, 'op')

        # Get operation parameters corresponding to model parameters in wrt filters if needed
        interposer = self.model._param_interposer
        ops_wrt_filter1 = interposer.ops_params_dependent_on_model_params(wrt_filter1) \
            if (interposer is not None and wrt_filter1 is not None) else wrt_filter1
        ops_wrt_filter2 = interposer.ops_params_dependent_on_model_params(wrt_filter2) \
            if (interposer is not None and wrt_filter2 is not None) else wrt_filter2

        # Allocate memory for the op-param Hessian (possibly the final result)
        num_op_params = self.model._param_interposer.num_op_params \
            if (self.model._param_interposer is not None) else self.model.num_params
        num_deriv_cols1 = num_op_params if (ops_wrt_filter1 is None) else len(ops_wrt_filter1)
        num_deriv_cols2 = num_op_params if (ops_wrt_filter2 is None) else len(ops_wrt_filter2)
        flattened_hprod = _np.zeros((dim**2, num_deriv_cols1, num_deriv_cols2), 'd')

        op_wrtFilter1, gpindices1 = self._process_wrt_filter(ops_wrt_filter1, gate)
        op_wrtFilter2, gpindices2 = self._process_wrt_filter(ops_wrt_filter2, gate)

        if _slct.length(gpindices1) > 0 and _slct.length(gpindices2) > 0:  # works for arrays too
            # Compute the derivative of the entire circuit with respect to the
            # gate's parameters and fill appropriate columns of flattened_dprod.
            _fas(flattened_hprod, [None, gpindices1, gpindices2],
                 gate.hessian_wrt_params(op_wrtFilter1, op_wrtFilter2))

        #Note: deriv_wrt_params is more accurately derive wrt *op* params when there is an interposer
        # d2(op)/d(p1)d(p2) = d2(op)/d(op_p1)d(op_p2) * d(op_p1)/d(p1) d(op_p2)/d(p2)
        if self.model._param_interposer is not None:
            def smx(mx, rows, cols):  # extract sub-matrix, as you might like mx[rows, cols] to work...
                rows = _np.array(rows).reshape(-1, 1); cols = _np.array(cols).reshape(1, -1)
                return mx[rows, cols]  # row index is m x 1, cols index is 1 x n so numpy broadcast works
            d_opp_wrt_p = self.model._param_interposer.deriv_op_params_wrt_model_params()
            d_opp_wrt_p1 = d_opp_wrt_p if (wrt_filter1 is None) else smx(d_opp_wrt_p, ops_wrt_filter1, wrt_filter1)
            d_opp_wrt_p2 = d_opp_wrt_p if (wrt_filter2 is None) else smx(d_opp_wrt_p, ops_wrt_filter2, wrt_filter2)
            flattened_hprod = _np.einsum('ijk,jl,km->ilm', flattened_hprod, d_opp_wrt_p1, d_opp_wrt_p2)

            #Update num_deriv_cols variabls (currently = # of op-params) to # of model params in each dimension
            num_deriv_cols1 = self.model._param_interposer.num_params if (wrt_filter1 is None) else len(wrt_filter1)
            num_deriv_cols2 = self.model._param_interposer.num_params if (wrt_filter2 is None) else len(wrt_filter2)

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

        dim = self.model.evotype.minimal_dim(self.model.state_space)

        #Cache partial products (relatively little mem required)
        leftProds = []
        G = _np.identity(dim); leftProds.append(G)
        for opLabel in revOpLabelList:
            G = _np.dot(G, self.model.circuit_layer_operator(opLabel, 'op').to_dense(on_space='minimal'))
            leftProds.append(G)

        rightProdsT = []
        G = _np.identity(dim); rightProdsT.append(_np.transpose(G))
        for opLabel in reversed(revOpLabelList):
            G = _np.dot(self.model.circuit_layer_operator(opLabel, 'op').to_dense(on_space='minimal'), G)
            rightProdsT.append(_np.transpose(G))

        # Allocate memory for the final result
        num_deriv_cols = self.model.num_params if (wrt_filter is None) else len(wrt_filter)
        flattened_dprod = _np.zeros((dim**2, num_deriv_cols), 'd')

        # For each operation label, compute the derivative of the entire circuit
        #  with respect to only that gate's parameters and fill the appropriate
        #  columns of flattened_dprod.
        uniqueOpLabels = sorted(list(set(revOpLabelList)))
        for opLabel in uniqueOpLabels:
            #REMOVE gate = self.model.circuit_layer_operator(opLabel, 'op')
            #REMOVE op_wrtFilter, gpindices = self._process_wrt_filter(wrt_filter, gate)
            #REMOVE dop_dopLabel = gate.deriv_wrt_params(op_wrtFilter)
            dop_dopLabel = self._doperation(opLabel, flat=True, wrt_filter=wrt_filter)  # (dim**2, num_deriv_cols)

            for (i, gl) in enumerate(revOpLabelList):
                if gl != opLabel: continue  # loop over locations of opLabel
                LRproduct = _np.kron(leftProds[i], rightProdsT[N - 1 - i])  # (dim**2, dim**2)
                flattened_dprod += _np.dot(LRproduct, dop_dopLabel)  # (dim**2, num_deriv_cols)
                #REMOVE _fas(flattened_dprod, [None, gpindices],
                #REMOVE      _np.dot(LRproduct, dop_dopLabel), add=True)  # (dim**2, n_params[opLabel])


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

        dim = self.model.evotype.minimal_dim(self.model.state_space)

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
                G = _np.dot(G, self.model.circuit_layer_operator(opLabel2, 'op').to_dense(on_space='minimal'))
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

        expanded_circuit_outcomes = self.model.expand_instruments_and_separate_povm(circuit, outcomes)
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
                # TODO - add a ".dense_space_type" attribute of evotype that == either "Hilbert" or "Hilbert-Schmidt"?
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


class MatrixForwardSimulator(_DistributableForwardSimulator, SimpleMatrixForwardSimulator):
    """
    Computes circuit outcome probabilities by multiplying together circuit-layer process matrices.

    Interfaces with a model via its `circuit_layer_operator` method and extracts a dense matrix
    representation of operators by calling their `to_dense` method.  An "evaluation tree" that
    composes all of the circuits using pairwise "joins"  is constructed by a :class:`MatrixCOPALayout`
    layout object, and this tree then directs pairwise multiplications of process matrices to compute
    circuit outcome probabilities.  Derivatives are computed analytically, using operators'
    `deriv_wrt_params` methods.

    Parameters
    ----------
    model : Model, optional
        The parent model of this simulator.  It's fine if this is `None` at first,
        but it will need to be set (by assigning `self.model` before using this simulator.

    distribute_by_timestamp : bool, optional
        When `True`, treat the data as time dependent, and distribute the computation of outcome
        probabilitiesby assigning groups of processors to the distinct time stamps within the
        dataset.  This means of distribution be used only when the circuits themselves contain
        no time delay infomation (all circuit layer durations are 0), as operators are cached
        at the "start" time of each circuit, i.e., the timestamp in the data set.  If `False`,
        then the data is treated in a time-independent way, and the overall counts for each outcome
        are used.  If support for intra-circuit time dependence is needed, you must use a different
        forward simulator (e.g. :class:`MapForwardSimulator`).

    num_atoms : int, optional
        The number of atoms (sub-evaluation-trees) to use when creating the layout (i.e. when calling
        :meth:`create_layout`).  This determines how many units the element (circuit outcome
        probability) dimension is divided into, and doesn't have to correclate with the number of
        processors.  When multiple processors are used, if `num_atoms` is less than the number of
        processors then `num_atoms` should divide the number of processors evenly, so that
        `num_atoms // num_procs` groups of processors can be used to divide the computation
        over parameter dimensions.

    processor_grid : tuple optional
        Specifies how the total number of processors should be divided into a number of
        atom-processors, 1st-parameter-deriv-processors, and 2nd-parameter-deriv-processors.
        Each level of specification is optional, so this can be a 1-, 2-, or 3- tuple of
        integers (or None).  Multiplying the elements of `processor_grid` together should give
        at most the total number of processors.

    param_blk_sizes : tuple, optional
        The parameter block sizes along the first or first & second parameter dimensions - so
        this can be a 0-, 1- or 2-tuple of integers or `None` values.  A block size of `None`
        means that there should be no division into blocks, and that each block processor
        computes all of its parameter indices at once.
    """

    @classmethod
    def _array_types_for_method(cls, method_name):
        # The array types of *intermediate* or *returned* values within various class methods (for memory estimates)
        if method_name == '_bulk_fill_probs_block': return cls._array_types_for_method('_compute_product_cache')
        if method_name == '_bulk_fill_dprobs_block':
            return cls._array_types_for_method('_compute_product_cache') \
                + cls._array_types_for_method('_compute_dproduct_cache')
        if method_name == '_bulk_fill_hprobs_block':
            return cls._array_types_for_method('_compute_product_cache') \
                + cls._array_types_for_method('_compute_dproduct_cache') \
                + cls._array_types_for_method('_compute_hproduct_cache')

        if method_name == '_compute_product_cache': return ('zdd', 'z', 'z')  # cache of gates, scales, and scaleVals
        if method_name == '_compute_dproduct_cache': return ('zddb',)  # cache x dim x dim x distributed_nparams
        if method_name == '_compute_hproduct_cache': return ('zddbb',)  # cache x dim x dim x dist_np1 x dist_np2
        return super()._array_types_for_method(method_name)

    def __init__(self, model=None, distribute_by_timestamp=False, num_atoms=None, processor_grid=None,
                 param_blk_sizes=None, cache_doperations=True):
        super().__init__(model, num_atoms, processor_grid, param_blk_sizes)
        self._cache_dops = cache_doperations
        self._mode = "distribute_by_timestamp" if distribute_by_timestamp else "time_independent"

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'mode': self._mode,
                      # (don't serialize parent model or processor distribution info)
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        #Note: resets processor-distribution information
        return cls(None, state['mode'] == "distribute_by_timestamp")

    def copy(self):
        """
        Return a shallow copy of this MatrixForwardSimulator

        Returns
        -------
        MatrixForwardSimulator
        """
        return MatrixForwardSimulator(self.model)

    def _compute_product_cache(self, layout_atom_tree, resource_alloc):
        """
        Computes an array of operation sequence products (process matrices).

        Note: will *not* parallelize computation:  parallelization should be
        done at a higher level.
        """
        dim = self.model.evotype.minimal_dim(self.model.state_space)

        #Note: resource_alloc gives procs that could work together to perform
        # computation, e.g. paralllel dot products but NOT to just partition
        # futher (e.g. among the wrt_slices) as this is done in the layout.
        # This function doesn't make use of resource_alloc - all procs compute the same thing.

        eval_tree = layout_atom_tree
        cacheSize = len(eval_tree)
        prodCache = _np.zeros((cacheSize, dim, dim), 'd')
        # ^ This assumes assignments prodCache[i] = <2d numpy array>.
        #   It would be better for this to be a dict (mapping _most likely_
        #   to ndarrays) if we don't need slicing or other axis indexing.
        scaleCache = _np.zeros(cacheSize, 'd')

        for iDest, iRight, iLeft in eval_tree:

            #Special case of an "initial operation" that can be filled directly
            if iRight is None:  # then iLeft gives operation:
                opLabel = iLeft
                if opLabel is None:
                    prodCache[iDest] = _np.identity(dim)
                    # Note: scaleCache[i] = 0.0 from initialization
                else:
                    gate = self.model.circuit_layer_operator(opLabel, 'op').to_dense(on_space='minimal')
                    nG = max(_nla.norm(gate), 1.0)
                    # ^ This indicates a need to compute norms of the operation matrices. Can't do this
                    #   with scipy.linalg if gate is represented implicitly. 
                    prodCache[iDest] = gate / nG
                    # ^ Indicates a need to overload division by scalars.
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
                nL = max(_nla.norm(L), _np.exp(-scaleCache[iLeft]),  1e-300)
                nR = max(_nla.norm(R), _np.exp(-scaleCache[iRight]), 1e-300)
                # ^ I want to allow L,R to be tensor product operators. That precludes
                #   calling _nla.norm.
                sL, sR = L / nL, R / nR
                # ^ Again, shows the need to overload division by scalars.
                prodCache[iDest] = sL @ sR
                scaleCache[iDest] += _np.log(nL) + _np.log(nR)

        nanOrInfCacheIndices = (~_np.isfinite(prodCache)).nonzero()[0]  # may be duplicates (a list, not a set)
        # since all scaled gates start with norm <= 1, products should all have norm <= 1
        assert(len(nanOrInfCacheIndices) == 0)

        return prodCache, scaleCache

    def _compute_dproduct_cache(self, layout_atom_tree, prod_cache, scale_cache,
                                resource_alloc=None, wrt_slice=None, profiler=None):
        """
        Computes a tree of product derivatives in a linear cache space. Will
        use derivative columns to parallelize computation.
        """

        if profiler is None: profiler = _dummy_profiler
        dim = self.model.evotype.minimal_dim(self.model.state_space)
        nDerivCols = self.model.num_params if (wrt_slice is None) \
            else _slct.length(wrt_slice)
        deriv_shape = (nDerivCols, dim, dim)
        eval_tree = layout_atom_tree
        cacheSize = len(eval_tree)

        #Note: resource_alloc gives procs that could work together to perform
        # computation, e.g. paralllel dot products but NOT to just partition
        # futher (e.g. among the wrt_slices) as this is done in the layout.
        # This function doesn't make use of resource_alloc - all procs compute the same thing.

        ## ------------------------------------------------------------------
        #
        ##print("MPI: _compute_dproduct_cache begin: %d deriv cols" % nDerivCols)
        #if resource_alloc is not None and resource_alloc.comm is not None and resource_alloc.comm.Get_size() > 1:
        #    #print("MPI: _compute_dproduct_cache called w/comm size %d" % comm.Get_size())
        #    # parallelize of deriv cols, then sub-trees (if available and necessary)
        #
        #    if resource_alloc.comm.Get_size() > nDerivCols:
        #
        #        #If there are more processors than deriv cols, give a
        #        # warning -- note that we *cannot* make use of a tree being
        #        # split because there's no good way to reconstruct the
        #        # *non-final* parent-tree elements from those of the sub-trees.
        #        _warnings.warn("Increased speed could be obtained by giving dproduct cache computation"
        #                       " *fewer* processors, as there are more cpus than derivative columns.")
        #
        #    # Use comm to distribute columns
        #    allDerivColSlice = slice(0, nDerivCols) if (wrt_slice is None) else wrt_slice
        #    _, myDerivColSlice, _, sub_resource_alloc = \
        #        _mpit.distribute_slice(allDerivColSlice, resource_alloc.comm)
        #    #print("MPI: _compute_dproduct_cache over %d cols (%s) (rank %d computing %s)" \
        #    #    % (nDerivCols, str(allDerivColIndices), comm.Get_rank(), str(myDerivColIndices)))
        #    if sub_resource_alloc is not None and sub_resource_alloc.comm is not None \
        #       and sub_resource_alloc.comm.Get_size() > 1:
        #        _warnings.warn("Too many processors to make use of in "
        #                       " _compute_dproduct_cache.")
        #        if sub_resource_alloc.comm.Get_rank() > 0: myDerivColSlice = slice(0, 0)
        #        #don't compute anything on "extra", i.e. rank != 0, cpus
        #
        #    my_results = self._compute_dproduct_cache(
        #        layout_atom_tree, prod_cache, scale_cache, None, myDerivColSlice, profiler)
        #    # pass None as comm, *not* mySubComm, since we can't do any
        #    #  further parallelization
        #
        #    tm = _time.time()
        #    all_results = resource_alloc.comm.allgather(my_results)
        #    profiler.add_time("MPI IPC", tm)
        #    return _np.concatenate(all_results, axis=1)  # TODO: remove this concat w/better gather?
        #
        ## ------------------------------------------------------------------

        tSerialStart = _time.time()
        dProdCache = _np.zeros((cacheSize,) + deriv_shape)
        # ^ I think that deriv_shape will be a tuple of length > 2.
        #   (Based on how swapaxes is used in the loop below ...)
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
                    # ^ Need a way to track tensor product structure in whatever's 
                    #   being returned by self._doperation (presumably it's a tensor ...)

                continue

            tm = _time.time()

            # combine iLeft + iRight => i
            # LEXICOGRAPHICAL VS MATRIX ORDER Note: we reverse iLeft <=> iRight from eval_tree because
            # (iRight,iLeft,iFinal) = tup implies circuit[i] = circuit[iLeft] + circuit[iRight], but we want:
            # since then matrixOf(circuit[i]) = matrixOf(circuit[iLeft]) * matrixOf(circuit[iRight])
            L, R = prod_cache[iLeft], prod_cache[iRight]
            dL, dR = dProdCache[iLeft], dProdCache[iRight]
            term1 = _np.dot(dL, R)
            term2 = _np.swapaxes(_np.dot(L, dR), 0, 1) 
            # ^ From the numpy docs on .dot :
            #
            #   If a is an N-D array and b is an M-D array (where M>=2),
            #   it is a sum product over the last axis of a and the second-to-last axis of b:
            #
            #       dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
            #
            dProdCache[iDest] = term1 +  term2  # dot(dS, T) + dot(S, dT)
            # ^ We need addition of tensor-product-structured "doperators."

            profiler.add_time("compute_dproduct_cache: dots", tm)
            profiler.add_count("compute_dproduct_cache: dots")

            scale = scale_cache[iDest] - (scale_cache[iLeft] + scale_cache[iRight])
            if abs(scale) > 1e-8:  # _np.isclose(scale,0) is SLOW!
                dProdCache[iDest] /= _np.exp(scale)
                if dProdCache[iDest].max() < _DSMALL and dProdCache[iDest].min() > -_DSMALL:
                    # ^ Need the tensor-product-structured "doperators" to have .max() and .min()
                    #   methods.
                    _warnings.warn("Scaled dProd small in order to keep prod managable.")
            elif (_np.count_nonzero(dProdCache[iDest]) and dProdCache[iDest].max() < _DSMALL
                  and dProdCache[iDest].min() > -_DSMALL):
                # ^ Need to bypass the call to _np.count_nonzero(...).
                _warnings.warn("Would have scaled dProd but now will not alter scale_cache.")

        #profiler.print_mem("DEBUGMEM: POINT2"); profiler.comm.barrier()

        profiler.add_time("compute_dproduct_cache: serial", tSerialStart)
        profiler.add_count("compute_dproduct_cache: num columns", nDerivCols)

        return dProdCache

    def _compute_hproduct_cache(self, layout_atom_tree, prod_cache, d_prod_cache1,
                                d_prod_cache2, scale_cache, resource_alloc=None,
                                wrt_slice1=None, wrt_slice2=None):
        """
        Computes a tree of product 2nd derivatives in a linear cache space. Will
        use derivative rows and columns to parallelize computation.
        """

        dim = self.model.evotype.minimal_dim(self.model.state_space)

        # Note: dProdCache?.shape = (#circuits,#params_to_diff_wrt,dim,dim)
        nDerivCols1 = d_prod_cache1.shape[1]
        nDerivCols2 = d_prod_cache2.shape[1]
        assert(wrt_slice1 is None or _slct.length(wrt_slice1) == nDerivCols1)
        assert(wrt_slice2 is None or _slct.length(wrt_slice2) == nDerivCols2)
        hessn_shape = (nDerivCols1, nDerivCols2, dim, dim)
        eval_tree = layout_atom_tree
        cacheSize = len(eval_tree)

        #Note: resource_alloc gives procs that could work together to perform
        # computation, e.g. paralllel dot products but NOT to just partition
        # futher (e.g. among the wrt_slices) as this is done in the layout.
        # This function doesn't make use of resource_alloc - all procs compute the same thing.

        ## ------------------------------------------------------------------
        #
        #if resource_alloc is not None and resource_alloc.comm is not None and resource_alloc.comm.Get_size() > 1:
        #    # parallelize of deriv cols, then sub-trees (if available and necessary)
        #
        #    if resource_alloc.comm.Get_size() > nDerivCols1 * nDerivCols2:
        #        #If there are more processors than deriv cells, give a
        #        # warning -- note that we *cannot* make use of a tree being
        #        # split because there's no good way to reconstruct the
        #        # *non-final* parent-tree elements from those of the sub-trees.
        #        _warnings.warn("Increased speed could be obtained"
        #                       " by giving hproduct cache computation"
        #                       " *fewer* processors and *smaller* (sub-)tree"
        #                       " (e.g. by splitting tree beforehand), as there"
        #                       " are more cpus than hessian elements.")  # pragma: no cover
        #
        #    # allocate final result memory
        #    hProdCache = _np.zeros((cacheSize,) + hessn_shape)
        #
        #    # Use comm to distribute columns
        #    allDeriv1ColSlice = slice(0, nDerivCols1)
        #    allDeriv2ColSlice = slice(0, nDerivCols2)
        #    deriv1Slices, myDeriv1ColSlice, deriv1Owners, mySubComm = \
        #        _mpit.distribute_slice(allDeriv1ColSlice, resource_alloc.comm)
        #
        #    # Get slice into entire range of model params so that
        #    #  per-gate hessians can be computed properly
        #    if wrt_slice1 is not None and wrt_slice1.start is not None:
        #        myHessianSlice1 = _slct.shift(myDeriv1ColSlice, wrt_slice1.start)
        #    else: myHessianSlice1 = myDeriv1ColSlice
        #
        #    #print("MPI: _compute_hproduct_cache over %d cols (rank %d computing %s)" \
        #    #    % (nDerivCols2, comm.Get_rank(), str(myDerivColSlice)))
        #
        #    if mySubComm is not None and mySubComm.Get_size() > 1:
        #        deriv2Slices, myDeriv2ColSlice, deriv2Owners, mySubSubComm = \
        #            _mpit.distribute_slice(allDeriv2ColSlice, mySubComm)
        #
        #        # Get slice into entire range of model params (see above)
        #        if wrt_slice2 is not None and wrt_slice2.start is not None:
        #            myHessianSlice2 = _slct.shift(myDeriv2ColSlice, wrt_slice2.start)
        #        else: myHessianSlice2 = myDeriv2ColSlice
        #
        #        if mySubSubComm is not None and mySubSubComm.Get_size() > 1:
        #            _warnings.warn("Too many processors to make use of in "
        #                           " _compute_hproduct_cache.")
        #            #TODO: remove: not needed now that we track owners
        #            #if mySubSubComm.Get_rank() > 0: myDeriv2ColSlice = slice(0,0)
        #            #  #don't compute anything on "extra", i.e. rank != 0, cpus
        #
        #        hProdCache[:, myDeriv1ColSlice, myDeriv2ColSlice] = self._compute_hproduct_cache(
        #            layout_atom_tree, prod_cache, d_prod_cache1[:, myDeriv1ColSlice],
        #            d_prod_cache2[:, myDeriv2ColSlice], scale_cache, None, myHessianSlice1, myHessianSlice2)
        #        # pass None as comm, *not* mySubSubComm, since we can't do any further parallelization
        #
        #        #NOTE: we only need to gather to the root processor (TODO: update this)
        #        _mpit.gather_slices(deriv2Slices, deriv2Owners, hProdCache, [None, myDeriv1ColSlice],
        #                            2, mySubComm)  # , gather_mem_limit) #gather over col-distribution (Deriv2)
        #        #note: gathering axis 2 of hProdCache[:,myDeriv1ColSlice],
        #        #      dim=(cacheSize,nDerivCols1,nDerivCols2,dim,dim)
        #    else:
        #        #compute "Deriv1" row-derivatives distribution only; don't use column distribution
        #        hProdCache[:, myDeriv1ColSlice] = self._compute_hproduct_cache(
        #            layout_atom_tree, prod_cache, d_prod_cache1[:, myDeriv1ColSlice], d_prod_cache2,
        #            scale_cache, None, myHessianSlice1, wrt_slice2)
        #        # pass None as comm, *not* mySubComm (this is ok, see "if" condition above)
        #
        #    #NOTE: we only need to gather to the root processor (TODO: update this)
        #    _mpit.gather_slices(deriv1Slices, deriv1Owners, hProdCache, [], 1, resource_alloc.comm)
        #    #, gather_mem_limit) #gather over row-distribution (Deriv1)
        #    #note: gathering axis 1 of hProdCache,
        #    #      dim=(cacheSize,nDerivCols1,nDerivCols2,dim,dim)
        #
        #    return hProdCache
        #
        ## ------------------------------------------------------------------

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
                      derivative_dimensions=None, verbosity=0, layout_creation_circuit_cache= None, use_old_tree_style: bool = True):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for a list of circuits.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be included in the layout.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        array_types : tuple, optional
            A tuple of string-valued array types.  See :meth:`ForwardSimulator.create_layout`.

        derivative_dimensions : int or tuple[int], optional
            Optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            non-None when `array_types` contains `'ep'` or `'epp'` types.
            If a tuple, then must be length 1.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.

        layout_creation_circuit_cache : dict, optional (default None)
            A precomputed dictionary serving as a cache for completed
            circuits. I.e. circuits with prep labels and POVM labels appended.
            Along with other useful pre-computed circuit structures used in layout
            creation.
            
        Returns
        -------
        MatrixCOPALayout
        """
        # There are two types of quantities we adjust to create a good layout: "group-counts" and "processor-counts"
        #  - group counts:  natoms, nblks, nblks2 give how many indpendently computed groups/ranges of circuits,
        #                   1st parameters, and 2nd parameters are used.  Making these larger can reduce memory
        #                   consumption by reducing intermediate memory usage.
        #  - processor counts: na, np, np2 give how many "atom-processors", "param-processors" and "param2-processors"
        #                      are used to process data along each given direction.  These values essentially specify
        #                      how the physical procesors are divided by giving the number of (roughly equal) intervals
        #                      exist along each dimension of the physical processor "grid".  Thus, thees values are set
        #                      based on the total number of cores available and how many dimensions are being computed.

        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        mem_limit = resource_alloc.mem_limit - resource_alloc.allocated_memory \
            if (resource_alloc.mem_limit is not None) else None  # *per-processor* memory limit
        printer = _VerbosityPrinter.create_printer(verbosity, resource_alloc)
        nprocs = resource_alloc.comm_size
        comm = resource_alloc.comm
        if isinstance(derivative_dimensions, int):
            num_params = derivative_dimensions
        elif isinstance(derivative_dimensions, tuple):
            assert len(derivative_dimensions) == 1
            num_params = derivative_dimensions[0]
        else:
            num_params = self.model.num_params
        C = 1.0 / (1024.0**3)

        if mem_limit is not None:
            if mem_limit <= 0:
                raise MemoryError("Attempted layout creation w/memory limit = %g <= 0!" % mem_limit)
            printer.log("Layout creation w/mem limit = %.2fGB" % (mem_limit * C))

        natoms, na, npp, param_dimensions, param_blk_sizes = self._compute_processor_distribution(
            array_types, nprocs, num_params, len(circuits), default_natoms=1)

        if self._mode == "distribute_by_timestamp":
            #Special case: time dependent data that gets grouped & distributed by unique timestamp
            # To to this, we override above values of natoms, na, and npp:
            natoms = 1  # save all processor division for within the (single) atom, for different timestamps
            na, npp = 1, (1, 1)  # save all processor division for within the (single) atom, for different timestamps

        printer.log("MatrixLayout: %d processors divided into %s (= %d) grid along circuit and parameter directions." %
                    (nprocs, ' x '.join(map(str, (na,) + npp)), _np.prod((na,) + npp)))
        printer.log("   %d atoms, parameter block size limits %s" % (natoms, str(param_blk_sizes)))
        assert(_np.prod((na,) + npp) <= nprocs), "Processor grid size exceeds available processors!"

        layout = _MatrixCOPALayout(circuits, self.model, dataset, natoms,
                                   na, npp, param_dimensions, param_blk_sizes, resource_alloc, verbosity, 
                                   layout_creation_circuit_cache=layout_creation_circuit_cache, use_old_tree_style=use_old_tree_style)

        if mem_limit is not None:
            loc_nparams1 = num_params / npp[0] if len(npp) > 0 else 0
            loc_nparams2 = num_params / npp[1] if len(npp) > 1 else 0
            blk1 = param_blk_sizes[0] if len(param_blk_sizes) > 0 else 0
            blk2 = param_blk_sizes[1] if len(param_blk_sizes) > 1 else 0
            if blk1 is None: blk1 = loc_nparams1
            if blk2 is None: blk2 = loc_nparams2
            global_layout = layout.global_layout
            if comm is not None:
                from mpi4py import MPI
                max_local_els = comm.allreduce(layout.num_elements, op=MPI.MAX)    # layout.max_atom_elements
                max_atom_els = comm.allreduce(layout.max_atom_elements, op=MPI.MAX)
                max_local_circuits = comm.allreduce(layout.num_circuits, op=MPI.MAX)
                max_atom_cachesize = comm.allreduce(layout.max_atom_cachesize, op=MPI.MAX)
            else:
                max_local_els = layout.num_elements
                max_atom_els = layout.max_atom_elements
                max_local_circuits = layout.num_circuits
                max_atom_cachesize = layout.max_atom_cachesize
            mem_estimate = _bytes_for_array_types(array_types, global_layout.num_elements, max_local_els, max_atom_els,
                                                  global_layout.num_circuits, max_local_circuits,
                                                  layout._param_dimensions, (loc_nparams1, loc_nparams2),
                                                  (blk1, blk2), max_atom_cachesize,
                                                  self.model.evotype.minimal_dim(self.model.state_space))

            GB = 1.0 / 1024.0**3
            if mem_estimate > mem_limit:
                raise MemoryError("Not enough memory for desired layout! (limit=%.1fGB, required=%.1fGB)" % (
                    mem_limit * GB, mem_estimate * GB))
            else:
                printer.log("   Esimated memory required = %.1fGB" % (mem_estimate * GB))

        return layout
    
    @staticmethod
    def create_copa_layout_circuit_cache(circuits, model, dataset=None):
        """
        Helper function for pre-computing/pre-processing circuits structures
        used in matrix layout creation.
        """
        cache = dict()
        completed_circuits, split_circuits = model.complete_circuits(circuits, return_split=True)

        cache['completed_circuits'] = {ckt: comp_ckt for ckt, comp_ckt in zip(circuits, completed_circuits)}
        cache['split_circuits'] = {ckt: split_ckt for ckt, split_ckt in zip(circuits, split_circuits)}

        if dataset is not None:
            aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
            ds_circuits = _lt.apply_aliases_to_circuits(circuits, aliases)
            unique_outcomes_list = []
            for ckt in ds_circuits:
                ds_row = dataset[ckt]
                unique_outcomes_list.append(ds_row.unique_outcomes if ds_row is not None else None)
        else:
            unique_outcomes_list = [None]*len(circuits)

        expanded_circuit_outcome_list = model.bulk_expand_instruments_and_separate_povm(circuits, 
                                                                                        observed_outcomes_list = unique_outcomes_list, 
                                                                                        split_circuits = split_circuits)
        
        expanded_circuit_cache = {ckt: expanded_ckt for ckt,expanded_ckt in zip(circuits, expanded_circuit_outcome_list)}
                    
        cache['expanded_and_separated_circuits'] = expanded_circuit_cache

        expanded_subcircuits_no_spam_cache = dict()
        for expc_outcomes in cache['expanded_and_separated_circuits'].values():
            for sep_povm_c, _ in expc_outcomes.items():  # for each expanded cir from unique_i-th circuit
                exp_nospam_c = sep_povm_c.circuit_without_povm[1:] 
                expanded_subcircuits_no_spam_cache[exp_nospam_c] = exp_nospam_c.expand_subcircuits()

        cache['expanded_subcircuits_no_spam'] = expanded_subcircuits_no_spam_cache

        return cache

    def _scale_exp(self, scale_exps):
        old_err = _np.seterr(over='ignore')
        scaleVals = _np.exp(scale_exps)  # may overflow, but OK if infs occur here
        _np.seterr(**old_err)
        return scaleVals

    def _rho_e_from_spam_tuple(self, spam_tuple):
        # This calculator uses the convention that rho has shape (N,1)
        rholabel, elabel = spam_tuple
        rho = self.model.circuit_layer_operator(rholabel, 'prep').to_dense(on_space='minimal')[:, None]
        E = _np.conjugate(_np.transpose(self.model.circuit_layer_operator(
            elabel, 'povm').to_dense(on_space='minimal')[:, None]))
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
        EVec = self.model.circuit_layer_operator(elabel, 'povm')   # arrays, these are State/POVMEffect objects
        nCircuits = gs.shape[0]

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
        path = _np.einsum_path('hk,ijkl,lm->ij', e, d_gs, rho, optimize='optimal')
        dp_dOps = _np.einsum('hk,ijkl,lm->ij', e, d_gs, rho, optimize=path[0]) * scale_vals[:, None]
        _np.seterr(**old_err2)
        # may overflow, but OK ; shape == (len(circuit_list), nDerivCols)
        # may also give invalid value due to scale_vals being inf and dot-prod being 0. In
        #  this case set to zero since we can't tell whether it's + or - inf anyway...
        dp_dOps[_np.isnan(dp_dOps)] = 0

        #SPAM -------------

        if self.model._param_interposer is not None:
            #When there is an interposer, we compute derivs wrt *all* the ops params (inefficient?),
            # then apply interposer, then take desired wrt_filter columns:
            nOpDerivCols = self.model._param_interposer.num_op_params

            dp_drhos = _np.zeros((nCircuits, nOpDerivCols))
            _fas(dp_drhos, [None, rhoVec.gpindices],
                 _np.squeeze(_np.dot(_np.dot(e, gs), rhoVec.deriv_wrt_params()),  # *don't* apply wrt filter here
                             axis=(0,)) * scale_vals[:, None])  # may overflow, but OK
            dp_drhos = _np.dot(dp_drhos, self.model._param_interposer.deriv_op_params_wrt_model_params())
            if wrt_slice is not None: dp_drhos = dp_drhos[:, wrt_slice]

            dp_dEs = _np.zeros((nCircuits, nOpDerivCols))
            dp_dAnyE = _np.squeeze(_np.dot(gs, rho), axis=(2,)) * scale_vals[:, None]
            _fas(dp_dEs, [None, EVec.gpindices], _np.dot(dp_dAnyE, EVec.deriv_wrt_params()))
            dp_dEs = _np.dot(dp_dEs, self.model._param_interposer.deriv_op_params_wrt_model_params())
            if wrt_slice is not None: dp_dEs = dp_dEs[:, wrt_slice]

        else:
            #Simpler case of no interposer
            nOpDerivCols = nDerivCols

            rho_wrtFilter, rho_gpindices = self._process_wrt_filter(
                wrt_slice, self.model.circuit_layer_operator(rholabel, 'prep'))
            E_wrtFilter, E_gpindices = self._process_wrt_filter(
                wrt_slice, self.model.circuit_layer_operator(elabel, 'povm'))

            # Get: dp_drhos[i, rho_gpindices] = dot(e,gs[i],drho/drhoP)
            # dp_drhos[i,J0+J] = sum_kl e[0,k] gs[i,k,l] drhoP[l,J]
            # dp_drhos[i,J0+J] = dot(e, gs, drhoP)[0,i,J]
            # dp_drhos[:,J0+J] = squeeze(dot(e, gs, drhoP),axis=(0,))[:,J]
            dp_drhos = _np.zeros((nCircuits, nOpDerivCols))
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
            dp_dEs = _np.zeros((nCircuits, nOpDerivCols))
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
        EVec = self.model.circuit_layer_operator(elabel, 'povm')   # arrays, these are State/POVMEffect objects
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

    def _bulk_fill_probs_atom(self, array_to_fill, layout_atom, resource_alloc):
        #Free memory from previous subtree iteration before computing caches
        scaleVals = Gs = prodCache = scaleCache = None
        dim = self.model.evotype.minimal_dim(self.model.state_space)
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * dim**2)  # prod cache

        #Fill cache info
        prodCache, scaleCache = self._compute_product_cache(layout_atom.tree, resource_alloc)

        if not resource_alloc.is_host_leader:
            # (same as "if resource_alloc.host_comm is not None and resource_alloc.host_comm.rank != 0")
            # we cannot further utilize multiplie processors when computing a single block.  The required
            # ending condition is that array_to_fill on each processor has been filled.  But if memory
            # is being shared and resource_alloc contains multiple processors on a single host, we only
            # want *one* (the rank=0) processor to perform the computation, since array_to_fill will be
            # shared memory that we don't want to have muliple procs using simultaneously to compute the
            # same thing.  Thus, we just do nothing on all of the non-root host_comm processors.
            # We could also print a warning (?), or we could carefully guard any shared mem updates
            # using "if resource_alloc.is_host_leader" conditions (if we could use  multiple procs elsewhere).
            return

        #use cached data to final values
        scaleVals = self._scale_exp(layout_atom.nonscratch_cache_view(scaleCache))
        Gs = layout_atom.nonscratch_cache_view(prodCache, axis=0)
        # ( n_circuits, dim, dim )

        old_err = _np.seterr(over='ignore')
        for spam_tuple, (element_indices, tree_indices) in layout_atom.indices_by_spamtuple.items():
            # "element indices" index a circuit outcome probability in array_to_fill's first dimension
            # "tree indices" index a quantity for a no-spam circuit in a computed cache, which correspond
            #  to the the element indices when `spamtuple` is used.
            # (Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller)
            rho, E = self._rho_e_from_spam_tuple(spam_tuple)
            _fas(array_to_fill, [element_indices],
                 self._probs_from_rho_e(rho, E, Gs[tree_indices], scaleVals[tree_indices]))
        _np.seterr(**old_err)

    def _bulk_fill_dprobs_atom(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        if not self._cache_dops:
            # This call errors because it tries to compute layout_atom.as_layout(resource_alloc),
            # which isn't implemented. Looking at how layout_atom is used in the other branch
            # of this if-statement it isn't clear how to work around this. Can look at 
            # MapForwardSimulator._bulk_fill_dprobs_atom(...).
            # 
            # Verbatim contents:
            #
            #       resource_alloc.check_can_allocate_memory(layout_atom.cache_size * self.model.dim * _slct.length(param_slice))
            #       self.calclib.mapfill_dprobs_atom(self, array_to_fill, slice(0, array_to_fill.shape[0]), dest_param_slice,
            #                                  layout_atom, param_slice, resource_alloc, self.derivative_eps)
            #
            # where
            #
            #       self.calclib = _importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_" + evotype.name).
            # 
            # and an implementation can be found at
            #
            #       /Users/rjmurr/Documents/pg-xfgst/repo/pygsti/forwardsims/mapforwardsim_calc_generic.py.
            #
            # Specifically, in mapfill_probs_atom. But that doesn't do anything like layout_atom.as_layout(resource_alloc) .... :(
            # 

            _DistributableForwardSimulator._bulk_fill_dprobs_atom(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc)
            return
        
        dim = self.model.evotype.minimal_dim(self.model.state_space)
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * dim * dim * _slct.length(param_slice))
        prodCache, scaleCache = self._compute_product_cache(layout_atom.tree, resource_alloc)
        dProdCache = self._compute_dproduct_cache(layout_atom.tree, prodCache, scaleCache,
                                                  resource_alloc, param_slice)
        if not resource_alloc.is_host_leader:
            return  # Non-root host processors aren't used anymore to compute the result on the root proc

        scaleVals = self._scale_exp(layout_atom.nonscratch_cache_view(scaleCache))
        Gs = layout_atom.nonscratch_cache_view(prodCache, axis=0)
        dGs = layout_atom.nonscratch_cache_view(dProdCache, axis=0)

        old_err = _np.seterr(over='ignore')
        for spam_tuple, (element_indices, tree_indices) in layout_atom.indices_by_spamtuple.items():
            rho, E = self._rho_e_from_spam_tuple(spam_tuple)
            _fas(array_to_fill, [element_indices, dest_param_slice], self._dprobs_from_rho_e(
                spam_tuple, rho, E, Gs[tree_indices], dGs[tree_indices], scaleVals[tree_indices], param_slice))

        _np.seterr(**old_err)

    def _bulk_fill_hprobs_atom(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                               param_slice1, param_slice2, resource_alloc):
        dim = self.model.evotype.minimal_dim(self.model.state_space)
        resource_alloc.check_can_allocate_memory(layout_atom.cache_size * dim**2
                                                 * _slct.length(param_slice1) * _slct.length(param_slice2))
        prodCache, scaleCache = self._compute_product_cache(layout_atom.tree, resource_alloc)
        dProdCache1 = self._compute_dproduct_cache(
            layout_atom.tree, prodCache, scaleCache, resource_alloc, param_slice1)  # computed on rank=0 only
        dProdCache2 = dProdCache1 if (param_slice1 == param_slice2) else \
            self._compute_dproduct_cache(layout_atom.tree, prodCache, scaleCache,
                                         resource_alloc, param_slice2)  # computed on rank=0 only
        hProdCache = self._compute_hproduct_cache(layout_atom.tree, prodCache, dProdCache1,
                                                  dProdCache2, scaleCache, resource_alloc,
                                                  param_slice1, param_slice2)  # computed on rank=0 only

        if not resource_alloc.is_host_leader:
            return  # Non-root host processors aren't used anymore to compute the result on the root proc

        scaleVals = self._scale_exp(layout_atom.nonscratch_cache_view(scaleCache))
        Gs = layout_atom.nonscratch_cache_view(prodCache, axis=0)
        dGs1 = layout_atom.nonscratch_cache_view(dProdCache1, axis=0)
        dGs2 = layout_atom.nonscratch_cache_view(dProdCache2, axis=0)
        #( n_circuits, nDerivColsX, dim, dim )

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
            dim = self.model.evotype.minimal_dim(self.model.state_space)
            dGs = _np.swapaxes(_np.swapaxes(dGs, 0, 1).reshape(
                (nDerivCols, nCircuits * dim**2)), 0, 1)

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
                                 num_total_outcomes, dataset, ds_cache=None):

        assert(self._mode == "distribute_by_timestamp"), \
            ("Must set `distribute_by_timestamp=True` to use a "
             "time-dependent objective function with MatrixForwardSimulator!")

        resource_alloc = layout.resource_alloc()
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        atom_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we begin

        #Split timestamps up between processors - maybe do this in a time-dep layout?
        all_timestamps = {i: t for i, t in enumerate(dataset.timestamps)}
        my_timestamp_inds, timestampOwners, timestamp_processing_ralloc = \
            _mpit.distribute_indices(list(range(len(all_timestamps))), atom_resource_alloc)
        shared_mem_leader = timestamp_processing_ralloc.is_host_leader

        probs_array, probs_array_shm = _smt.create_shared_ndarray(timestamp_processing_ralloc,
                                                                  (layout.num_elements,), 'd')
        # Allocated this way b/c, e.g.,  say we have 4 procs on a single node and 2 timestamps: then
        # timestamp_processing_ralloc will have 2 procs and only the first will fill probs_array below since
        #_bulk_fill_probs_atom assumes it's given shared mem allocated using the resource alloc object it's given.

        array_to_fill[:] = 0.0
        my_array_to_fill = _np.zeros(array_to_fill.shape, 'd')  # purely local array to accumulate results
        assert(my_array_to_fill.shape == (layout.num_elements,))

        for timestamp_index in my_timestamp_inds:
            timestamp = all_timestamps[timestamp_index]

            # compute objective at time timestamp
            counts, totals, freqs, firsts, indicesOfCircuitsWithOmittedData = \
                self._ds_quantities(timestamp, ds_cache, layout, dataset)
            if counts is None: return  # no data at this time => no contribution

            for _, obj in self.model._iter_parameterized_objs():
                obj.set_time(timestamp)
            for opcache in self.model._opcaches.values():
                for obj in opcache.values():
                    obj.set_time(timestamp)

            for atom in layout.atoms:  # layout only holds local atoms
                self._bulk_fill_probs_atom(probs_array[atom.element_slice], atom, timestamp_processing_ralloc)

            timestamp_processing_ralloc.host_comm_barrier()  # don't exit until all proc's array_to_fill is ready
            # (similar to DistributableForwardSimulator._bulk_fill_probs)

            terms = raw_objective.terms(probs_array, counts, totals, freqs)
            if firsts is not None and shared_mem_leader:  # consolidate with `_update_terms_for_omitted_probs`
                omitted_probs = 1.0 - _np.array([_np.sum(probs_array[layout.indices_for_index(i)])
                                                 for i in indicesOfCircuitsWithOmittedData])
                terms[firsts] += raw_objective.zero_freq_terms(totals[firsts], omitted_probs)
            timestamp_processing_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

            my_array_to_fill += terms

        #collect/gather results (SUM local arrays together)
        resource_alloc.allreduce_sum(array_to_fill, my_array_to_fill, unit_ralloc=timestamp_processing_ralloc)

        _smt.cleanup_shared_ndarray(probs_array_shm)

    def _bulk_fill_timedep_dobjfn(self, raw_objective, array_to_fill, layout, ds_circuits,
                                  num_total_outcomes, dataset, ds_cache=None):

        assert(self._mode == "distribute_by_timestamp"), \
            ("Must set `distribute_by_timestamp=True` to use a "
             "time-dependent objective function with MatrixForwardSimulator!")

        resource_alloc = layout.resource_alloc()
        param_resource_alloc = layout.resource_alloc('param-processing')
        param_resource_alloc.host_comm_barrier()  # ensure all procs have finished w/shared memory before we begin

        #Split timestamps up between processors - maybe do this in a time-dep layout?
        all_timestamps = {i: t for i, t in enumerate(dataset.timestamps)}
        my_timestamp_inds, timestampOwners, timestamp_processing_ralloc = \
            _mpit.distribute_indices(list(range(len(all_timestamps))), param_resource_alloc)
        shared_mem_leader = timestamp_processing_ralloc.is_host_leader

        probs_array, probs_array_shm = _smt.create_shared_ndarray(timestamp_processing_ralloc,
                                                                  (layout.num_elements,), 'd')
        dprobs_array, dprobs_array_shm = _smt.create_shared_ndarray(timestamp_processing_ralloc,
                                                                    (layout.num_elements, self.model.num_params), 'd')
        # Allocated this way b/c, e.g.,  say we have 4 procs on a single node and 2 timestamps: then
        # timestamp_processing_ralloc will have 2 procs and only the first will fill probs_array below since
        #_bulk_fill_probs_atom assumes it's given shared mem allocated using the resource alloc object it's given.

        array_to_fill[:] = 0.0
        my_array_to_fill = _np.zeros(array_to_fill.shape, 'd')  # purely local array to accumulate results
        all_param_slice = slice(0, self.model.num_params)  # All params computed at once for now
        assert(my_array_to_fill.shape == (layout.num_elements, self.model.num_params))

        for timestamp_index in my_timestamp_inds:
            timestamp = all_timestamps[timestamp_index]
            # compute objective at time layout_atom.time
            #print("DB: Rank %d : layout atom for t=" % resource_alloc.comm.rank, layout_atom.timestamp)

            counts, totals, freqs, firsts, indicesOfCircuitsWithOmittedData = \
                self._ds_quantities(timestamp, ds_cache, layout, dataset)

            for _, obj in self.model._iter_parameterized_objs():
                obj.set_time(timestamp)
            for opcache in self.model._opcaches.values():
                for obj in opcache.values():
                    obj.set_time(timestamp)

            for atom in layout.atoms:  # layout only holds local atoms
                self._bulk_fill_probs_atom(probs_array, atom, timestamp_processing_ralloc)
                self._bulk_fill_dprobs_atom(dprobs_array, all_param_slice, atom,
                                            all_param_slice, timestamp_processing_ralloc)

            timestamp_processing_ralloc.host_comm_barrier()  # don't exit until all proc's array_to_fill is ready
            # (similar to DistributableForwardSimulator._bulk_fill_probs)

            if shared_mem_leader:
                if firsts is not None:  # consolidate with TimeIndependentMDCObjectiveFunction.dterms?
                    dprobs_omitted_rowsum = _np.empty((len(firsts), self.model.num_params), 'd')
                    for ii, i in enumerate(indicesOfCircuitsWithOmittedData):
                        dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs_array[layout.indices_for_index(i), :], axis=0)

                dprobs_array *= raw_objective.dterms(probs_array, counts, totals, freqs)[:, None]

                if firsts is not None:  # consolidate with _update_dterms_for_omitted_probs?
                    omitted_probs = 1.0 - _np.array([_np.sum(probs_array[layout.indices_for_index(i)])
                                                     for i in indicesOfCircuitsWithOmittedData])
                    dprobs_array[firsts] -= raw_objective.zero_freq_dterms(totals[firsts], omitted_probs)[:, None] \
                        * dprobs_omitted_rowsum
            timestamp_processing_ralloc.host_comm_barrier()  # have non-leader procs wait for leaders to set shared mem

            my_array_to_fill += dprobs_array

        #collect/gather results (SUM local arrays together)
        resource_alloc.allreduce_sum(array_to_fill, my_array_to_fill, unit_ralloc=timestamp_processing_ralloc)

        _smt.cleanup_shared_ndarray(probs_array_shm)
        _smt.cleanup_shared_ndarray(dprobs_array_shm)

    def bulk_fill_timedep_chi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                               min_prob_clip_for_weighting, prob_clip_interval, ds_cache=None):
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
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

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

        Returns
        -------
        None
        """
        from pygsti.objectivefns.objectivefns import RawChi2Function as _RawChi2Function
        raw_obj = _RawChi2Function({'min_prob_clip_for_weighting': min_prob_clip_for_weighting},
                                   layout.resource_alloc())
        return self._bulk_fill_timedep_objfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                             dataset, ds_cache)

    def bulk_fill_timedep_dchi2(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                min_prob_clip_for_weighting, prob_clip_interval, chi2_array_to_fill=None,
                                ds_cache=None):
        """
        Compute the chi2 jacobian contributions for an entire tree of circuits, allowing for time dependent operations.

        Similar to :meth:`bulk_fill_timedep_chi2` but compute the *jacobian*
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
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

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

        Returns
        -------
        None
        """
        from pygsti.objectivefns.objectivefns import RawChi2Function as _RawChi2Function
        raw_obj = _RawChi2Function({'min_prob_clip_for_weighting': min_prob_clip_for_weighting},
                                   layout.resource_alloc())
        return self._bulk_fill_timedep_dobjfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                              dataset, ds_cache)

    def bulk_fill_timedep_loglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                 min_prob_clip, radius, prob_clip_interval, ds_cache=None):
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
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

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

        Returns
        -------
        None
        """
        from pygsti.objectivefns.objectivefns import RawPoissonPicDeltaLogLFunction as _RawPoissonPicDeltaLogLFunction
        raw_obj = _RawPoissonPicDeltaLogLFunction({'min_prob_clip': min_prob_clip, 'radius': radius},
                                                  layout.resource_alloc())
        return self._bulk_fill_timedep_objfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                             dataset, ds_cache)

    def bulk_fill_timedep_dloglpp(self, array_to_fill, layout, ds_circuits, num_total_outcomes, dataset,
                                  min_prob_clip, radius, prob_clip_interval, logl_array_to_fill=None, ds_cache=None):
        """
        Compute the ("poisson picture")log-likelihood jacobian contributions for an entire tree of circuits.

        Similar to :meth:`bulk_fill_timedep_loglpp` but compute the *jacobian*
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
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

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

        Returns
        -------
        None
        """
        from pygsti.objectivefns.objectivefns import RawPoissonPicDeltaLogLFunction as _RawPoissonPicDeltaLogLFunction
        raw_obj = _RawPoissonPicDeltaLogLFunction({'min_prob_clip': min_prob_clip, 'radius': radius},
                                                  layout.resource_alloc())
        return self._bulk_fill_timedep_dobjfn(raw_obj, array_to_fill, layout, ds_circuits, num_total_outcomes,
                                              dataset, ds_cache)


class LCSEvalTreeMatrixForwardSimulator(MatrixForwardSimulator):

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

        my_data = setup_circuit_list_for_LCS_computations(circuits, None)

        full_tree = CollectionOfLCSEvalTrees(my_data[2], my_data[1], my_data[0])

        full_tree.collapse_circuits_to_process_matrices(self.model)
        Gs = full_tree.reconstruct_full_matrices()

        return Gs

    def _bulk_fill_probs_atom(self, array_to_fill, layout_atom: _MatrixCOPALayoutAtomWithLCS, resource_alloc):
        
        # Overestimate the amount of cache usage by assuming everything is the same size.
        dim = self.model.evotype.minimal_dim(self.model.state_space)
        # resource_alloc.check_can_allocate_memory(len(layout_atom.tree.cache) * dim**2)  # prod cache

        starttime =time.time()
        layout_atom.tree.collapse_circuits_to_process_matrices(self.model)
        endtime = time.time()

        print("Time to collapse the process matrices (s): ", endtime - starttime)
        starttime = time.time()
        Gs = layout_atom.tree.reconstruct_full_matrices()
        endtime = time.time()
        print("Time to reconstruct the whole matrices (s): ", endtime - starttime)

        starttime = time.time()
        old_err = _np.seterr(over='ignore')
        for spam_tuple, (element_indices, tree_indices) in layout_atom.indices_by_spamtuple.items():
            # "element indices" index a circuit outcome probability in array_to_fill's first dimension
            # "tree indices" index a quantity for a no-spam circuit in a computed cache, which correspond
            #  to the the element indices when `spamtuple` is used.
            # (Note: *don't* set dest_indices arg = layout.element_slice, as this is already done by caller)
            rho, E = self._rho_e_from_spam_tuple(spam_tuple)
            _fas(array_to_fill, [element_indices],
                 self._probs_from_rho_e(rho, E, Gs[tree_indices], 1))
        _np.seterr(**old_err)
        endtime = time.time()
        print("Time to complete the spam operations (s): ", endtime - starttime)

    def _bulk_fill_dprobs_atom(self, array_to_fill, dest_param_slice, layout_atom: _MatrixCOPALayoutAtomWithLCS, param_slice, resource_alloc):

        
        eps = 1e-7  # hardcoded?
        if param_slice is None:
            param_slice = slice(0, self.model.num_params)
        param_indices = _slct.to_array(param_slice)

        if dest_param_slice is None:
            dest_param_slice = slice(0, len(param_indices))
        dest_param_indices = _slct.to_array(dest_param_slice)

        iParamToFinal = {i: dest_param_indices[ii] for ii, i in enumerate(param_indices)}

        probs = _np.empty(layout_atom.num_elements, 'd')
        self._bulk_fill_probs_atom(probs, layout_atom, resource_alloc)

        probs2 = _np.empty(layout_atom.num_elements, 'd')
        orig_vec = self.model.to_vector().copy()

        for i in range(self.model.num_params):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.model.from_vector(vec, close=True)
                self._bulk_fill_probs_atom(probs2, layout_atom, resource_alloc)
                array_to_fill[:, iFinal] = (probs2 - probs) / eps

    def create_layout(self, circuits, dataset=None, resource_alloc=None, array_types=('E', ), derivative_dimensions=None, verbosity=0, layout_creation_circuit_cache=None):
        return super().create_layout(circuits, dataset, resource_alloc, array_types, derivative_dimensions, verbosity, layout_creation_circuit_cache, use_old_tree_style=False)
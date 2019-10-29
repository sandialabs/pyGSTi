""" Defines the OplessModel class"""
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import collections as _collections

from .model import Model as _Model
from .evaltree import EvalTree as _EvalTree
from .labeldicts import OutcomeLabelDict as _OutcomeLabelDict
from .circuit import Circuit as _Circuit
from .polynomial import Polynomial as _Polynomial
from ..tools import slicetools as _slct

try:
    from . import fastopcalc as _fastopcalc
    from .fastopcalc import fast_compact_deriv as _compact_deriv

    def _bulk_eval_compact_polys(vtape, ctape, paramvec, dest_shape):
        if _np.iscomplexobj(ctape):
            ret = _fastopcalc.fast_bulk_eval_compact_polys_complex(
                vtape, ctape, paramvec, dest_shape)
            assert(_np.linalg.norm(_np.imag(ret)) < 1e-6), \
                "norm(Im part) = %g" % _np.linalg.norm(_np.imag(ret))  # DEBUG CHECK
            return _np.real(ret)
        else:
            return _np.real(_fastopcalc.fast_bulk_eval_compact_polys(
                vtape, ctape, paramvec, dest_shape))
except ImportError:
    from .polynomial import bulk_eval_compact_polys as poly_bulk_eval_compact_polys
    from .polynomial import compact_deriv as _compact_deriv

    def _bulk_eval_compact_polys(vtape, ctape, paramvec, dest_shape):
        ret = poly_bulk_eval_compact_polys(vtape, ctape, paramvec, dest_shape)
        if _np.iscomplexobj(ret):
            #assert(_np.linalg.norm(_np.imag(ret)) < 1e-6 ), \
            #    "norm(Im part) = %g" % _np.linalg.norm(_np.imag(ret)) # DEBUG CHECK
            if _np.linalg.norm(_np.imag(ret)) > 1e-6:
                print("WARNING: norm(Im part) = %g" % _np.linalg.norm(_np.imag(ret)))

            ret = _np.real(ret)
        return ret  # always return a *real* vector


class OplessModelTree(_EvalTree):
    def __init__(self, circuit_list, lookup, outcome_lookup, cache=None):
        _EvalTree.__init__(self, circuit_list)
        self.element_indices = lookup
        self.outcomes = outcome_lookup
        self.num_final_strs = len(circuit_list) #circuits
        max_el_index = -1
        for elIndices in lookup.values():
            max_i = elIndices.stop-1 if isinstance(elIndices, slice) else max(elIndices)
            max_el_index = max(max_el_index, max_i)
        self.num_final_els = max_el_index+1
        self.cache = cache
        

class OplessModel(_Model):
    """
    TODO docstring
    """

    def __init__(self, state_space_labels):
        """
        Creates a new Model.  Rarely used except from derived classes
        `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.
        """
        _Model.__init__(self, state_space_labels)
        
        #Setting things the rest of pyGSTi expects but probably shouldn't...
        self.simtype = "opless"
        self.basis = None
        self.dim = 0

    def get_dimension(self):
        return self.dim

    def get_num_outcomes(self, circuit):  # needed for sparse data detection
        raise NotImplementedError("Derived classes should implement this!")

    def probs(self, circuit, clipTo=None, cache=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        clipTo : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clipTo)
            for each spam label (string) SL.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def dprobs(self, circuit, returnPr=False, clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,returnPr,clipTo)
            for each spam label (string) SL.
        """
        eps = 1e-7
        orig_pvec = self.to_vector()
        Np = self.num_params()
        probs0 = self.probs(circuit, clipTo, None)
               
        deriv = { k:_np.empty(Np, 'd') for k in probs0.keys() }
        for i in range(Np):
            p_plus_dp = p.copy()
            p_plus_dp[i] += eps
            self.from_vector(p_plus_dp)
            probs1 = self.probs(circuit, clipTo, None)
            for k,p0 in probs0.items():
                deriv[k][i] = (probs1[k] - p0) / eps
        self.from_vector(orig_pvec)
        
        if returnPr:
            return {k:(p0,deriv[k]) for k in probs0.keys()}
        else:
            return deriv

    def bulk_evaltree_from_resources(self, circuit_list, comm=None, memLimit=None,
                                     distributeMethod="default", subcalls=[],
                                     dataset=None, verbosity=0):
        #TODO: choose these based on resources, and enable split trees
        minSubtrees=0
        numSubtreeComms=1
        maxTreeSize=None
        evTree = self.bulk_evaltree(circuit_list, minSubtrees, maxTreeSize,
                                    numSubtreeComms, dataset, verbosity)
        return evTree, 0, 0, evTree.element_indices, evTree.outcomes

    def bulk_evaltree(self, circuit_list, minSubtrees=None, maxTreeSize=None,
                      numSubtreeComms=1, dataset=None, verbosity=0):
        raise NotImplementedError("Derived classes should implement this!")

    def bulk_probs(self, circuit_list, clipTo=None, check=False,
                   comm=None, memLimit=None, dataset=None, smartc=None):
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(circuit_list, comm, memLimit, "default", [], dataset)
        vp = _np.empty(evalTree.num_final_elements(), 'd')
        self.bulk_fill_probs(vp, evalTree, clipTo, check, comm)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(evalTree):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            ret[opstr] = _OutcomeLabelDict(
                [(outLbl, vp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_dprobs(self, circuit_list, returnPr=False, clipTo=None,
                    check=False, comm=None, wrtBlockSize=None, dataset=None):
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(circuit_list, comm, memLimit, "default", [], dataset)
        nElements = evalTree.num_final_elements()
        nDerivCols = self.num_params()

        vdp = _np.empty((nElements, nDerivCols), 'd')
        vp = _np.empty(nElements, 'd') if returnPr else None

        self.bulk_fill_dprobs(vdp, evalTree,
                              vp, clipTo, check, comm,
                              wrtFilter, wrtBlockSize)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(evalTree):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            if returnPr:
                ret[opstr] = _OutcomeLabelDict(
                    [(outLbl, (vdp[ei], vp[ei])) for ei, outLbl in zip(elInds, outcomes[i])])
            else:
                ret[opstr] = _OutcomeLabelDict(
                    [(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_fill_probs(self, mxToFill, evalTree, clipTo=None, check=False, comm=None):
        if False and evalTree.cache: #TEST (disabled)
            cpolys = evalTree.cache
            ps = _bulk_eval_compact_polys(cpolys[0], cpolys[1], self._paramvec, (evalTree.num_final_elements(),))
            assert(_np.linalg.norm(_np.imag(ps)) < 1e-6)
            ps = _np.real(ps)
            if clipTo is not None: ps = _np.clip(ps, clipTo[0], clipTo[1])
            mxToFill[:] = ps
        else:
            for i,c in enumerate(evalTree):
                cache = evalTree.cache[i] if evalTree.cache else None
                probs = self.probs(c,clipTo,cache)
                elInds = _slct.indices(evalTree.element_indices[i]) \
                    if isinstance(evalTree.element_indices[i], slice) else evalTree.element_indices[i]
                for k,outcome in zip(elInds, evalTree.outcomes[i]):
                    mxToFill[k] = probs[outcome]

    def bulk_fill_dprobs(self, mxToFill, evalTree, prMxToFill=None, clipTo=None,
                         check=False, comm=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):

        Np = self.num_params()
        p = self.to_vector()
        
        if False and evalTree.cache: #TEST (disabled)
            cpolys = evalTree.cache
            if prMxToFill is not None:
                ps = _bulk_eval_compact_polys(cpolys[0], cpolys[1], p, (evalTree.num_final_elements(),))
                assert(_np.linalg.norm(_np.imag(ps)) < 1e-6)
                ps = _np.real(ps)
                if clipTo is not None: ps = _np.clip(ps, clipTo[0], clipTo[1])
                prMxToFill[:] = ps
            dpolys = _compact_deriv(cpolys[0], cpolys[1], list(range(Np)))
            dps = _bulk_eval_compact_polys(dpolys[0], dpolys[1], p, (evalTree.num_final_elements(), Np))
            mxToFill[:,:] = dps
        else:
            eps = 1e-6
            for i,c in enumerate(evalTree):
                cache = evalTree.cache[i] if evalTree.cache else None
                probs0 = self.probs(c,clipTo,cache)
                elInds = _slct.indices(evalTree.element_indices[i]) \
                    if isinstance(evalTree.element_indices[i], slice) else evalTree.element_indices[i]
                for k,outcome in zip(elInds, evalTree.outcomes[i]):
                    if prMxToFill is not None:
                        prMxToFill[k] = probs0[outcome]
                    for j in range(Np):
                        p_plus_dp = p.copy()
                        p_plus_dp[j] += eps
                        self.from_vector(p_plus_dp)
                        probs1 = self.probs(c,clipTo,cache)
                        mxToFill[k,j] = (probs1[outcome]-probs0[outcome]) / eps
                    self.from_vector(p)

    def __str__(self):
        raise "Derived classes should implement OplessModel.__str__ !!"

    
class SuccessFailModel(OplessModel):
    def __init__(self, state_space_labels, use_cache=False):
        OplessModel.__init__(self, state_space_labels)
        self.use_cache = use_cache

    def get_num_outcomes(self, circuit):  # needed for sparse data detection
        return 2

    def _success_prob(self, circuit):
        raise NotImplementedError("Derived classes should implement this!")
        
    #FUTURE?: def _fill_circuit_probs(self, array_to_fill, outcomes, circuit, clipTo):
    def probs(self, circuit, clipTo=None, cache=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        clipTo : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[outcome] = pr(outcome,circuit,clipTo).
        """
        sp = self._success_prob(circuit, cache)
        if clipTo is not None: sp = _np.clip(sp, clipTo[0], clipTo[1])
        return _OutcomeLabelDict([('success',sp), ('fail',1-sp)])

    def poly_probs(self, circuit):
        """ 
        Same as probs(...) but return polynomials.
        """
        sp = self._success_prob_poly(circuit)
        return _OutcomeLabelDict([('success',sp), ('fail',_Polynomial({(): 1.0})-sp)])

    def bulk_evaltree(self, circuit_list, minSubtrees=None, maxTreeSize=None,
                      numSubtreeComms=1, dataset=None, verbosity=0):
        lookup = { i:slice(2*i,2*i+2,1) for i in range(len(circuit_list)) }
        outcome_lookup = { i:(('success',),('fail',)) for i in range(len(circuit_list)) }

        if self.use_cache == "poly":
            #Do precomputation here
            polys = []
            for i,circuit in enumerate(circuit_list):
                print("Generating probs for circuit %d of %d" % (i+1,len(circuit_list)))
                probs = self.poly_probs(circuit)
                polys.append(probs['success'])
                polys.append(probs['fail'])
            compact_polys = compact_poly_list(polys)
            cache = compact_polys
        elif self.use_cache == True:
            cache = [ self._circuit_cache(circuit) for circuit in circuit_list]
        else:
            cache = None

        return OplessModelTree(circuit_list, lookup, outcome_lookup, cache)

#TODO: move this to polynomial.py??
def compact_poly_list(list_of_polys):
    """Create a single vtape,ctape pair from a list of normal Polynomals """
    tapes = [p.compact() for p in list_of_polys]
    vtape = _np.concatenate([t[0] for t in tapes])
    ctape = _np.concatenate([t[1] for t in tapes])
    return vtape, ctape


class TimsErrorRatesModel(SuccessFailModel):
    def __init__(self, nQubits, error_rates, model_type='GlobalDep'):
        state_space_labels = ['Q%d' % i for i in range(nQubits)]
        SuccessFailModel.__init__(self, state_space_labels, use_cache=True)
        
        gate_error_rate_keys = (list(error_rates['gates'].keys()))
        readout_error_rate_keys = (list(error_rates['readout'].keys()))
        self._gate_error_rate_indices = { k:i for i,k in enumerate(gate_error_rate_keys) }
        self._readout_error_rate_indices = { k:i+len(gate_error_rate_keys) for i,k in enumerate(readout_error_rate_keys) }
        self._paramvec = _np.concatenate( (_np.array([error_rates['gates'][k] for k in gate_error_rate_keys], 'd'),
                                           _np.array([error_rates['readout'][k] for k in readout_error_rate_keys], 'd')) )
        assert(model_type in ('FiE', 'FiE+U', 'GlobalDep'))
        self.model_type = model_type

    def _circuit_cache(self, circuit):
        if not isinstance(circuit, _Circuit):
            circuit = _Circuit.fromtup(circuit)

        depth = circuit.depth()
        width = circuit.width()
        g_inds = self._gate_error_rate_indices
        r_inds = self._readout_error_rate_indices

        if self.model_type == 'GlobalDep':
            inds_to_mult_by_layer = []
            for i in range(depth):

                layer = circuit.get_layer(i)
                inds_to_mult = []
                usedQs = []

                for gate in layer:
                    if len(gate.qubits) > 1:
                        usedQs += list(gate.qubits)
                        inds_to_mult.append(g_inds[frozenset(gate.qubits)])

                for q in circuit.line_labels:
                    if q not in usedQs:
                        inds_to_mult.append(g_inds[q])
                        
                inds_to_mult_by_layer.append(_np.array(inds_to_mult,int))

            # Bit-flip readout error as a pre-measurement depolarizing channel.
            inds_to_mult = [ r_inds[q] for q in circuit.line_labels ]
            inds_to_mult_by_layer.append(_np.array(inds_to_mult,int))

            return (width, depth, inds_to_mult_by_layer)
        else:
            raise NotImplementedError("Other model types not implemented yet!")

    def _success_prob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec
        if cache is None:
            cache = self._circuit_cache(circuit)

        if self.model_type == 'GlobalDep':
            width, depth, inds_to_mult_by_layer = cache
            pvec1 = 1.0 - self._paramvec
            pvec2 = 1.0 - 3*self._paramvec/2
            alpha = 4**width / (4**width - 1)
            
            p = _np.prod([ (1 - alpha * (1 - _np.prod(pvec1[inds_to_mult]))) \
                           for inds_to_mult in inds_to_mult_by_layer[:-1]])
            sp_layer = _np.prod(pvec2[inds_to_mult_by_layer[-1]])
            p *= 1 - alpha * (1 - sp_layer)
            return p + (1 - p) * (1 / 2**width)
        else:
            raise NotImplementedError("Other model types not implemented yet!")

    def ORIGINAL_success_prob(self, circuit, cache):
        """
        todo
        """
        if not isinstance(circuit, _Circuit):
            circuit = _Circuit.fromtup(circuit)

        depth = circuit.depth()
        width = circuit.width()
        pvec = self._paramvec
        g_inds = self._gate_error_rate_indices
        r_inds = self._readout_error_rate_indices

        if self.model_type in ('FE', 'FiE+U'):

            twoQgates = []
            for i in range(depth):
                layer = circuit.get_layer(i)
                twoQgates += [q.qubits for q in layer if len(q.qubits) > 1]

            sp = 1
            oneqs = {q: depth for q in circuit.line_labels}

            for qs in twoQgates:
                sp = sp * (1 - pvec[g_inds[frozenset(qs)]])
                oneqs[qs[0]] += -1
                oneqs[qs[1]] += -1

            sp = sp * _np.prod([(1 - pvec[g_inds[q]])**oneqs[q]
                                * (1 - pvec[r_inds[q]]) for q in circuit.line_labels])

            if self.model_type == 'FiE+U':
                sp = sp + (1 - sp) * (1 / 2**width)

            return sp

        if self.model_type == 'GlobalDep':

            p = 1
            for i in range(depth):

                layer = circuit.get_layer(i)
                sp_layer = 1
                usedQs = []

                for gate in layer:
                    if len(gate.qubits) > 1:
                        usedQs += list(gate.qubits)
                        sp_layer = sp_layer * (1 - pvec[g_inds[frozenset(gate.qubits)]])

                for q in circuit.line_labels:
                    if q not in usedQs:
                        sp_layer = sp_layer * (1 - pvec[g_inds[q]])

                p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
                p = p * p_layer

            # Bit-flip readout error as a pre-measurement depolarizing channel.
            sp_layer = _np.prod([(1 - 3 * pvec[r_inds[q]] / 2) for q in circuit.line_labels])
            p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
            p = p * p_layer
            sp = p + (1 - p) * (1 / 2**width)

            return sp

    def __str__(self):
        s = "Error Rates model with error rates: \n" + \
            "\n".join(["%s = %g" % (k,self._paramvec[i]) for k,i in self._gate_error_rate_indices.items()]) + "\n" + \
            "\n".join(["%s = %g" % (k,self._paramvec[i]) for k,i in self._readout_error_rate_indices.items()])
        return s

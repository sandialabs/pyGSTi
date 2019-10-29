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

class OplessModelTree(_EvalTree):
    def __init__(self, circuit_list):
        _EvalTree.__init__(self, circuit_list)
        self.num_final_els = 2*len(circuit_list)
        self.num_final_strs = len(circuit_list) #circuits

        

class OplessModel(_Model):
    """
    TODO docstring
    """

    def __init__(self, state_space_labels, error_rates, model_type='GlobalDep'):
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
        
        gate_error_rate_keys = (list(error_rates['gates'].keys()))
        readout_error_rate_keys = (list(error_rates['readout'].keys()))
        self._gate_error_rate_indices = { k:i for i,k in enumerate(gate_error_rate_keys) }
        self._readout_error_rate_indices = { k:i+len(gate_error_rate_keys) for i,k in enumerate(readout_error_rate_keys) }
        self._paramvec = _np.concatenate( (_np.array([error_rates['gates'][k] for k in gate_error_rate_keys], 'd'),
                                           _np.array([error_rates['readout'][k] for k in readout_error_rate_keys], 'd')) )
        assert(model_type in ('FiE', 'FiE+U', 'GlobalDep'))
        self.model_type = model_type

        #Setting things the rest of pyGSTi expects but probably shouldn't...
        self.simtype = "other" # !!!
        self.basis = None
        self.dim = 0

    def get_dimension(self):
        return 0

    def get_num_outcomes(self, circuit):  # needed for sparse data detection
        return 2

    def _success_prob(self, circuit):
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

    # todo: remove this.
    def get_model_type(self):
        """
        TODO docstring
        """
        return self.model_type

    def probs(self, circuit, clipTo=None):
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
        assert(clipTo is None)
        sp = self._success_prob(circuit)
        return _OutcomeLabelDict([('success',sp), ('fail',1-sp)])

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
        assert(returnPr == False and clipTo is None)
        eps = 1e-7
        orig_pvec = self.to_vector()
        Np = self.num_params()
        sp0 = self._success_prob(circuit)
               
        deriv = _np.empty(Np, 'd')
        for i in range(Np):
            p_plus_dp = p.copy()
            p_plus_dp[i] += eps
            self.from_vector(p_plus_dp)
            deriv[i] = (self._success_prob(circuit) - spo) / eps
        self.from_vector(orig_pvec)
        return {'success': deriv, 'fail': -1*deriv}

    def bulk_evaltree_from_resources(self, circuit_list, comm=None, memLimit=None,
                                     distributeMethod="default", subcalls=[],
                                     dataset=None, verbosity=0):
        lookup = { i:slice(2*i,2*i+2,1) for i in range(len(circuit_list)) }
        outcome_lookup = { i:(('success',),('fail',)) for i in range(len(circuit_list)) }
        return OplessModelTree(circuit_list), 0, 0, lookup, outcome_lookup

    def bulk_evaltree(self, circuit_list, minSubtrees=None, maxTreeSize=None,
                      numSubtreeComms=1, dataset=None, verbosity=0):
        return OplessModelTree(circuit_list)

    def bulk_probs(self, circuit_list, clipTo=None, check=False,
                   comm=None, memLimit=None, dataset=None, smartc=None):
        evalTree, _, _, _, _ = self.bulk_evaltree_from_resources(circuit_list, comm, memLimit, "default", [], dataset)
        vp = _np.empty(evalTree.num_final_elements(), 'd')
        self.bulk_fill_probs(vp, evalTree, clipTo, check, comm)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(evalTree):
            ret[opstr] = _OutcomeLabelDict([('success', vp[2*i]), ('fail', vp[2*i+1])])
        return ret

    def bulk_dprobs(self, circuit_list, returnPr=False, clipTo=None,
                    check=False, comm=None, wrtBlockSize=None, dataset=None):
        evalTree, _, _, _, _ = self.bulk_evaltree_from_resources(circuit_list, comm, memLimit, "default", [], dataset)
        nElements = evalTree.num_final_elements()
        nDerivCols = self.num_params()

        vdp = _np.empty((nElements, nDerivCols), 'd')
        vp = _np.empty(nElements, 'd') if returnPr else None

        self.bulk_fill_dprobs(vdp, evalTree,
                              vp, clipTo, check, comm,
                              wrtFilter, wrtBlockSize)

        #TODO
        ret = _collections.OrderedDict()
        for i, opstr in enumerate(evalTree):
            if returnPr:
                ret[opstr] = _OutcomeLabelDict(
                    [(outLbl, (vdp[ei], vp[ei])) for ei, outLbl in zip(elInds, outcomes[i])])
            else:
                ret[opstr] = _OutcomeLabelDict(
                    [(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_fill_probs(self, mxToFill, evalTree, clipTo=None, check=False, comm=None):
        for i,c in enumerate(evalTree):
            sp = self._success_prob(c)
            if clipTo is not None: sp = _np.clip(sp, clipTo[0], clipTo[1])
            mxToFill[2*i] = sp
            mxToFill[2*i+1] = 1-sp

    def bulk_fill_dprobs(self, mxToFill, evalTree, prMxToFill=None, clipTo=None,
                         check=False, comm=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):

        eps = 1e-6
        p = self.to_vector()
        Np = self.num_params()
        for i,c in enumerate(evalTree):
            sp0 = self._success_prob(c)
            if prMxToFill is not None:
                if clipTo is not None:
                    prMxToFill[i] = _np.clip(sp0, clipTo[0], clipTo[1])
                else:
                    prMxToFill[i] = sp0
                
            for j in range(Np):
                p_plus_dp = p.copy()
                p_plus_dp[j] += eps
                self.from_vector(p_plus_dp)
                dsp = (self._success_prob(c) - sp0) / eps
                mxToFill[2*i,j] = dsp
                mxToFill[2*i+1,j] = -dsp
            self.from_vector(p)

    def __str__(self):
        s = "Error Rates model with error rates: \n" + \
            "\n".join(["%s = %g" % (k,self._paramvec[i]) for k,i in self._gate_error_rate_indices.items()]) + "\n" + \
            "\n".join(["%s = %g" % (k,self._paramvec[i]) for k,i in self._readout_error_rate_indices.items()])
        return s

    

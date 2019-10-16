""" ... """
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


class ErrorRatesModel(object):
    """
    todo
    """
    def __init__(self, error_rates, model_type='GlobalDep'):
        """

        model_type: {'FiE', 'FiE+Uni', 'GlobalDep'}
        """
        self.error_rates = error_rates
        assert(model_type in ('FiE', 'FiE+U', 'GlobalDep'))
        self.model_type = model_type

    def success_prob(self, circuit):
        """
        todo
        """
        depth = circuit.depth()
        width = circuit.width()

        if self.model_type in ('FiE', 'FiE+U'):

            twoQgates = []
            for i in range(depth):
                layer = circuit.get_layer(i)
                twoQgates += [q.qubits for q in layer if len(q.qubits) > 1]

            sp = 1
            oneqs = {q: depth for q in circuit.line_labels}

            for qs in twoQgates:
                sp = sp * (1 - self.error_rates['gates'][frozenset(qs)])
                oneqs[qs[0]] += -1
                oneqs[qs[1]] += -1

            sp = sp * _np.prod([(1 - self.error_rates['gates'][q])**oneqs[q]
                                * (1 - self.error_rates['readout'][q]) for q in circuit.line_labels])

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
                        sp_layer = sp_layer * (1 - self.error_rates['gates'][frozenset(qs)])

                for q in circuit.line_labels:
                    if q not in usedQs:
                        sp_layer = sp_layer * (1 - self.error_rates['gates'][q])

                p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
                p = p * p_layer

            p = p * _np.prod([(1 - self.error_rates['readout'][q]) for q in circuit.line_labels])
            sp = p + (1 - p) * (1 / 2**width)

            return sp

    # todo: remove this.
    def get_model_type(self):
        """

        """
        return self.model_type

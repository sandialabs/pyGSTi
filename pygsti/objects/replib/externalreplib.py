"""Defines Python-version calculation "representation" objects for external simulators"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


# TODO: Shift to base class (either ExternalOpRep or pushed up for all OpReps)
# Also maybe check we don't see a performance hit with the inheritance
# Representations for external simulators do not need to have acton methods
# as we do not always have access to the internal state of simulator
class ExternalOpRep(object):
    def acton(self, state):
        raise NotImplementedError()

    def adjoint_acton(self, state):
        raise NotImplementedError()


class CHPOpRep(ExternalOpRep):
    def __init__(self, ops, nqubits):
        self.chp_ops = ops
        self.nqubits = nqubits
        self.dim = 2**nqubits

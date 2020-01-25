""" RPE Protocol objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from . import protocol as _proto


class RobustPhaseEstimationDesign(_proto.CircuitListsDesign):
    """ Experimental design for robust phase estimation """

    def __init__(self, rpe_experiment_inputs, qubit_labels=None):
        circuit_lists = None  # TODO
        super().__init__(circuit_lists, qubit_labels=qubit_labels)


class RobustPhaseEstimation(_proto.Protocol):
    """ Robust phase estimation (RPE) protocol """

    def __init__(self, rpe_params, name=None):

        super().__init__(name)
        # ...
        self.rpeparams = rpe_params
        self.auxfile_types['rpe_params'] = 'pickle'

    def run(self, data):
        design = data.edesign  # experiment design (specifies circuits)
        ds = data.dataset  # dataset
        # ...
        ret = RobustPhaseEstimationResults(data, self, self.more_args)
        # ...
        return ret


class RobustPhaseEstimationResults(_proto.ProtocolResults):
    """ Results from the RPE protocol """

    def __init__(self, data, protocol_instance, more_args):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        self.rpe_result_values = 0

        self.auxfile_types['rpe_result_values'] = 'pickle'  # if rep_result_values can't be json'd


#shorthands
RPEDesign = RobustPhaseEstimationDesign
RPE = RobustPhaseEstimation
RPEResults = RobustPhaseEstimationResults

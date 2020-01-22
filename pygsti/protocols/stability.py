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


class StabilityAnalysisDesign(_proto.ExperimentDesign):
    """ Experimental design for stability analysis """
    def __init__(self, circuit_list, qubit_labels=None):
        self.needs_timestamps = True
        super().__init__(circuit_list, qubit_labels=qubit_labels)


class StabilityAnalysis(_proto.Protocol):
    """ Robust phase estimation (RPE) protocol """

    def __init__(self, params_from_do_stability_analysis, transform, name=None):

        super().__init__(name)
        self.transform = transform
        # ...

        #self.auxfile_types['big_thing'] = 'pickle'

    def run(self, data):
        design = data.edesign  # experiment design (specifies circuits)
        ds = data.dataset  # dataset
        # ... do analysis
        return StabilityAnalysisResults(data, self, self.more_args)  # put results in here


class StabilityAnalysisResults(_proto.ProtocolResults):
    """ Results from the RPE protocol """
    def __init__(self, data, protocol_instance, more_args):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        self.stored_results = more_args

        #self.auxfile_types['rpe_result_values'] = 'pickle'  # if rep_result_values can't be json'd


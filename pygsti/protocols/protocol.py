""" Protocol object """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class Protocol(object):
    def __init__(self):
        pass

    def run():
        raise NotImplementedError("Derived classes should implement this!")

    def run_on_data():
        raise NotImplementedError("Derived classes should implement this!")


class ProtocolInput(object):
    """ Serialize-able input data for a protocol """
    def __init__(self, default_protocol_name=None, basedata=None, typestring=None):
        self.typestring = self.__class__.__name__
        self.basedata = {} if basedata is None else basedata
        self.default_protocol_info = {} if default_protocol_name is None \
            else {'protocol_name': default_protocol_name}

    def create_circuit_list(self, verbosity=0):
        return self.basedata['circuitList']

    def create_circuit_lists(self, verbosity=0):  # Needed?? / Helpful?
        return [self.create_circuit_list()]

    def read(self, dirname):
        pass

    def write(self, dirname):
        pass


class SerialProtocolInputs(ProtocolInput):
    def __init__(self, pinputs):
        metadata = {'default_protocol': 'multi', 'subinputs': pinputs}
        all_circuits = []
        for inp in pinputs:
            all_circuits.extend(inp.circuits)
        _lt.remove_duplicates_inplace(all_circuits)
        super(metadata, all_circuits)


class ParallelProtocolInputs(ProtocolInput):
    """ TODO - need to be given sub-inputs whose circuits all act on the same set of
        qubits and are disjoint with the sets of all other sub-inputs.
    """
    def __init__(self, pinputs):
        #TODO
        metadata = {'default_protocol': 'multi-parallel', 'subinputs': pinputs}
        all_circuits = []
        for inp in pinputs:
            all_circuits.extend(inp.circuits)
        _lt.remove_duplicates_inplace(all_circuits)
        super(metadata, all_circuits)


class ProtocolData(object):
    def __init__(self, pinput, dataset=None):
        self.input = pinput
        self.dataset = dataset  # MultiDataSet allowed for multi-pass data?


class ProtocolResults(ProtocolData):
    def __init__(self, pdata, result_qtys):
        super(pdata.input, pdata.dataset)
        self.qtys = result_qtys if (result_qtys is not None) else {}


#Need way to specify *where* the data for a protocol input comes from that
# isn't the data itself - maybe an object within a ProtocolDirectory?
# e.g. create a

#Operations we'd like to have - maybe in creating MultiInput?
# - merge circuits to perform protocols in parallel: Inputs => MultiInput
# - interleave protcols so data is taken together: Inputs => MultiInput
# - add a protocol whose data will be taken along with (and maybe overlaps) an existing
#    protocol's data: Input.add(Input) => MultiInput containing both?
# - don't nest MultiInputs, i.e. MultiInput + Input => MultiInput only one level deep
# - Directory holds multinputs separately - a type of some kind of link between datasets and inputs...
#    any other type needed?

# - Protocol.run_on_data methods can take a ProtocolResults object and try to extract cached qtys to speed up calc
# - possible to create inputs from a ProcessorSpec, protocol name, and target qubits?

# Directory structure:
# root/inputs/NAME/SUBinputNAME...   - same as saving a collection of inputs in named dirs
# root/datasets/NAME - datasets - same names as top-level inputs (maybe multi-inputs) - just saved DataSets
# root/results/NAME/SUBinputNAME...  -- but may want protocols to specify how results should be
#  organized separately, e.g. datasets of success counts for nQ RB?  But could we add MultiInput types that
#  know to store e.g. marginalized counts, in a higher level directory that any existing or added sub-protocols
#  can utilize?
# root/reports/REPORTNAME - reports generated separately?  Maybe Directory has create_report and add_report
#  methods?  Is it possible to allow reports to pull from a cache of results somewhere?


class ProtocolDirectory(object):
    """ Holds multiple ProtocolData objects
    - and maybe can add an object with a protocol input and a data name (or not?)?
    - issue is, how to allow same data to be used for different protocols...
    Could hold reports too?
    """
    def __init__(self, inputs, datas, reports):
        self.inputs = inputs  # should be a dict of inputs; otherwise make into a dict
        self.datas  # pull datas apart into datasets and inputs; collect unique DataSets -> all_datasets_in_datas
        self.datasets = all_datasets_in_datas
        self.reports = reports
        self.results = ???

    def read(self, dirname):
        pass

    def write(self, dirname):
        pass

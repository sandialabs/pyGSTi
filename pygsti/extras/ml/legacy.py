
"""Legacy neural network / model-fitting utilities.

This module contains older helper functions for:
  * Extracting success/fail datasets from tabular data
  * Building a simple `TwirledLayersModel` from a ProcessorSpec
  * Fitting an MLE model via pyGSTi GST protocols

Notes
-----
These functions predate the newer QPANN workflows and are kept for backwards compatibility.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools

import numpy as _np

from pygsti.models.oplessmodel import TwirledLayersModel
from pygsti.baseobjs.label import Label
from pygsti.data import DataSet
from pygsti.circuits import Circuit
from pygsti.protocols import GSTDesign
from pygsti.protocols import ProtocolData
from pygsti.protocols import GST
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas
    from pygsti.processors import ProcessorSpec

def batch_prediction(model: Any, circuits: list[Circuit], prediction_key: str = 'success') -> _np.ndarray:
    """Compute a vector of outcome probabilities from a model over a list of circuits.

    Parameters
    ----------
    model : object
        Model object exposing `model.probabilities(circuit)` which returns a dict-like
        mapping from outcome labels to probabilities.
    circuits : list[pygsti.circuits.Circuit]
        Circuits to evaluate.
    prediction_key : str, default 'success'
        Which outcome probability to extract.

    Returns
    -------
    numpy.ndarray
        Array of shape `(len(circuits),)` with the requested probabilities.
    """
    return _np.array([model.probabilities(circuit)[prediction_key] for circuit in circuits])

def extract_success_fail(df: "pandas.DataFrame") -> tuple[DataSet, list[Circuit]]:
    """Build a two-outcome ('success','fail') pyGSTi DataSet from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame expected to contain columns:
          * 'Circuit' : circuit string representation
          * 'D:SC' : success counts
          * 'D:Counts' : total counts

    Returns
    -------
    sfds : pygsti.data.DataSet
        Dataset with outcomes ['success','fail'].
    circuits : list[pygsti.circuits.Circuit]
        Circuits in the same order as the dataframe rows.
    """
    sfds = DataSet(outcome_labels = ['success','fail'])
    circuits = []
    for _, row in df.iterrows():
        circuit = Circuit(None, stringrep = row['Circuit'])
        sfds.add_count_dict(circuit, {'success': row['D:SC'] , 'fail': row['D:Counts'] - row['D:SC']})
        circuits.append(circuit)
    sfds.done_adding_data()
    return sfds, circuits

def create_spec_model(pspec: "ProcessorSpec") -> tuple[TwirledLayersModel, dict]:
    """Create a simple `TwirledLayersModel` and corresponding error dictionary from a ProcessorSpec.

    Parameters
    ----------
    pspec : ProcessorSpec
        Processor specification that supplies qubit labels, gate names, and availability.

    Returns
    -------
    specmodel : pygsti.models.oplessmodel.TwirledLayersModel
        A model initialized with uniform gate and readout error rates.
    error_dict : dict
        Dictionary used to construct the model (gate and readout error rates).

    Notes
    -----
    This is a heuristic initializer: it sets all 1Q gates and CNOTs to error rate 0.01 and
    all readout errors to 0.0.
    """
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(pspec, QubitProcessorSpec)
    qubit_labels = pspec.qubit_labels
    num_qubits = len(qubit_labels)
    one_qubit_gate_names = pspec.gate_names[1:]
    availability = pspec.availability
    error_dict = {'gates': {},
                  'readout': {i: .0 for i in qubit_labels}}

    for i,j in _itertools.product(one_qubit_gate_names, qubit_labels):
        error_dict['gates'][Label(i,state_space_labels = (j,))] = 0.01
    for i,j in _itertools.product(['Gcnot'],availability['Gcnot']):
        error_dict['gates'][Label('Gcnot',state_space_labels = cast(Any, j))] = 0.01
    specmodel = TwirledLayersModel(error_dict, num_qubits = num_qubits, state_space_labels = qubit_labels, 
                                   idle_name = cast(Any, None))
    
    return specmodel, error_dict

def create_mle_model(df: "pandas.DataFrame", indices: dict, pspec: "ProcessorSpec", specmodel: TwirledLayersModel) -> object:
    """Fit an MLE GST model on a subset of circuits selected from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing circuits and counts (see `extract_success_fail`).
    indices : dict
        Dictionary with key 'train' giving indices/boolean mask selecting training rows.
    pspec : ProcessorSpec
        Processor specification for GSTDesign.
    specmodel : pygsti model
        Starting model for GST.

    Returns
    -------
    mlemodel : pygsti model
        Final iteration estimate from GST results.
    """
    from pygsti.processors import QubitProcessorSpec
    assert isinstance(pspec, QubitProcessorSpec)
    sfds, train_circuits  = extract_success_fail(df.loc[indices['train']]) 
    train_circuits = list(sfds.keys())
    print('You had {} circuits. You are training on {} circuits.'.format(len(df), len(train_circuits)))
    print('Creating MLE Model')

    gst_edesign = GSTDesign(pspec, [train_circuits]) #creates gst edesign based on a specmodel. sfds.keys is the set of circuits. <---- restrict this
    gst_protocol_data = ProtocolData(gst_edesign, sfds) # No need to change sfds because the protocol only looks for circuits in its design. DOUBLE CHECK!
    gst_protocol = GST(specmodel, gaugeopt_suite=cast(Any, None),  badfit_options={}, verbosity=1) #
    gst_results = gst_protocol.run(gst_protocol_data) #fits an mle model to the gst_protocol_data
    mlemodel = gst_results.estimates['GateSetTomography'].models['final iteration estimate'] # extracts the mle model parameters
    
    return mlemodel
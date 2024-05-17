import itertools

import numpy as _np

from pygsti.models.oplessmodel import TwirledLayersModel
from pygsti.baseobjs.label import Label
from pygsti.data import DataSet
from pygsti.circuits import Circuit
from pygsti.protocols import GSTDesign
from pygsti.protocols import ProtocolData
from pygsti.protocols import GST

def batch_prediction(model, circuits, prediction_key = 'success'):
    return _np.array([model.probabilities(circuit)[prediction_key] for circuit in circuits])

def extract_success_fail(df):
    sfds = DataSet(outcome_labels = ['success','fail'])
    circuits = []
    for _, row in df.iterrows():
        circuit = Circuit(None, stringrep = row['Circuit'])
        sfds.add_count_dict(circuit, {'success': row['D:SC'] , 'fail': row['D:Counts'] - row['D:SC']})
        circuits.append(circuit)
    sfds.done_adding_data()
    return sfds, circuits

def create_spec_model(pspec):
    qubit_labels = pspec.qubit_labels
    num_qubits = len(qubit_labels)
    one_qubit_gate_names = pspec.gate_names[1:]
    availability = pspec.availability
    error_dict = {'gates': {},
                  'readout': {i: .0 for i in qubit_labels}}

    for i,j in itertools.product(one_qubit_gate_names, qubit_labels):
        error_dict['gates'][Label(i,state_space_labels = (j,))] = 0.01
    for i,j in itertools.product(['Gcnot'],availability['Gcnot']):
        error_dict['gates'][Label('Gcnot',state_space_labels = j)] = 0.01
    specmodel = TwirledLayersModel(error_dict, num_qubits = num_qubits, state_space_labels = qubit_labels, 
                                   idle_name = None)
    
    return specmodel, error_dict

def create_mle_model(df, indices, pspec, specmodel):
    sfds, train_circuits  = extract_success_fail(df.loc[indices['train']]) 
    train_circuits = list(sfds.keys())
    print('You had {} circuits. You are training on {} circuits.'.format(len(df), len(train_circuits)))
    print('Creating MLE Model')

    gst_edesign = GSTDesign(pspec, [train_circuits]) #creates gst edesign based on a specmodel. sfds.keys is the set of circuits. <---- restrict this
    gst_protocol_data = ProtocolData(gst_edesign, sfds) # No need to change sfds because the protocol only looks for circuits in its design. DOUBLE CHECK!
    gst_protocol = GST(specmodel, gaugeopt_suite=None,  badfit_options={}, verbosity=1) #
    gst_results = gst_protocol.run(gst_protocol_data) #fits an mle model to the gst_protocol_data
    mlemodel = gst_results.estimates['GateSetTomography'].models['final iteration estimate'] # extracts the mle model parameters
    
    return mlemodel
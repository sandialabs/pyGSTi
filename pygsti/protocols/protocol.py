""" Protocol object """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import numpy as _np
import itertools as _itertools
import copy as _copy
import json as _json
import pickle as _pickle
import pathlib as _pathlib
import importlib as _importlib

from . import support as _support
from .. import construction as _cnst
from .. import objects as _objs
from .. import io as _io
from ..objects import circuit as _cir
from ..tools import listtools as _lt


def load_protocol_from_dir(dirname):
    dirname = _pathlib.Path(dirname)
    return _support.obj_from_meta_json(dirname).from_dir(dirname)


class Protocol(object):
    @classmethod
    def from_dir(cls, dirname):
        ret = cls.__new__(cls)
        ret.__dict__.update(_support.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types'))
        ret._init_unserialized_attributes()
        return ret

    def __init__(self, name=None):
        super().__init__()
        self.name = name if name else self.__class__.__name__
        self.auxfile_types = {}

    def run(self, data):
        raise NotImplementedError("Derived classes should implement this!")

    def write(self, dirname):
        _support.write_obj_to_meta_based_dir(self, dirname, 'auxfile_types')

    def _init_unserialized_attributes(self):
        """Initialize anything that isn't serialized based on the things that are serialized.
           Usually this means initializing things with auxfile_type == 'none' that aren't
           separately serialized.
        """
        pass


class MultiPassProtocol(Protocol):
    # expects a MultiDataSet of passes and maybe adds data comparison (?) - probably not RB specific
    def __init__(self, protocol, name=None):
        if name is None: name = self.protocol.name + "_multipass"
        super().__init__(name)
        self.protocol = protocol
        self.auxfile_types['protocol'] = 'protocolobj'

    def run(self, data):
        results = MultiPassResults(data, self)
        for pass_name, sub_data in data.passes.items():  # a multipass DataProtocol object contains per-pass datas
            #TODO: print progress: pass X of Y, etc
            sub_results = self.protocol.run(sub_data)
            # TODO: maybe blank-out the .data and .protocol of sub_results since we don't need this info?  or call as_dict?
            results.passes[pass_name] = sub_results  # pass_name is a "ds_name" key of data.dataset (a MultiDataSet)
        return results


#class SimpleProtocol(Protocol):
#
#    def run(self, data):
#        if data.is_multipass():
#            implicit_multipassprotocol = MultiPassProtocol(self)
#            return implicit_multipassprotocol.run(data)
#
#        elif data.is_multiinput():  # ~ is this is a directory
#            # Note: we also know that self *isn't* a MultiProtocol object since
#            # MultiProtocol overrides run(...).
#            #if isinstance(data.input, SimultaneousInput):
#            #    implicit_multiprotocol = SimultaneousProtocol(self)
#            #else:
#            implicit_multiprotocol = MultiProtocol(self)
#            return implicit_multiprotocol.run(data)
#
#        else:
#            return self.simple_run(data)
#
#    def simple_run(self, data):
#        raise NotImplementedError("Derived classes should implement this!")


# MultiProtocol -> runs, on same input circuit structure & data, multiple protocols (e.g. different GST & model tests on same GST data)
#   if that one input is a MultiInput, then it must have the same number of inputs as there are protocols and each protocol is run on the corresponding input.
#   if that one input is a normal input, then the protocols can cache information in a Results object that is handed down.
class ProtocolRunner(object):
    """ - creates ProtocolResultsDir vs. Protocols, which create ProtocolResults objects"""
    def run(self, data):
        raise NotImplementedError()


class TreeRunner(ProtocolRunner):
    """Run specific protocols on specific paths"""
    def __init__(self, protocol_dict):
        self.protocols = protocol_dict

    def run(self, data):
        ret = ProtocolResultsDir(data)  # creates entire tree of nodes
        for path, protocol in self.protocols.items():

            root = ret
            for el in path:  # traverse path
                root = root[el]
            root.for_protocol[protocol.name] = protocol.run(root.data)  # run the protocol

        return ret


class SimpleRunner(ProtocolRunner):
    """ Run a single protocol on every data node that has no sub-nodes (possibly separately for each pass) """
    def __init__(self, protocol, protocol_can_handle_multipass_data=False):
        self.protocol = protocol
        self.do_passes_separately = not protocol_can_handle_multipass_data

    def run(self, data):
        ret = ProtocolResultsDir(data)  # creates entire tree of nodes

        def visit_node(node):
            if len(node.data) > 0:
                for subname, subnode in node.items():
                    visit_node(subnode)
            elif node.data.is_multipass() and self.do_passes_separately:
                implicit_multipassprotocol = MultiPassProtocol(self.protocol)
                node.for_protocol[implicit_multipassprotocol.name] = implicit_multipassprotocol.run(node.data)
            else:
                node.for_protocol[self.protocol.name] = self.protocol.run(node.data)
        visit_node(ret)
        return ret


class DefaultRunner(ProtocolRunner):
    def __init__(self):
        pass

    def run(self, data):
        ret = ProtocolResultsDir(data)  # creates entire tree of nodes

        def visit_node(node):
            for name, protocol in node.data.input.default_protocols.items():
                assert(name == protocol.name), "Protocol name inconsistency"
                node.for_protocol[name] = protocol.run(node.data)

            for subname, subnode in node.items():
                visit_node(subnode)

        visit_node(ret)
        return ret

        
#class MultiProtocol(Protocol):
#    """This class simply runs sub-protocols on corresponding sub-datas """
#    def __init__(self, protocols, name=None):
#        if name is None and protocols:
#            name = protocols.name if isinstance(protocols, Protocol) else \
#                '-'.join([p.name for p in protocols])
#        super().__init__(name)
#        self.protocols = protocols
#        #self.root_qty_name = 'Protocol'
#        self.auxfile_types['protocols'] = 'list-of-protocolobjs'
#
#    def run(self, data):
#        protocols = self.protocols
#
#        assert(data.is_multiinput())
#        #root_qty_name = data.input.root_qty_name  # the multi-input should know what category name corresponds to it's keys
#        if isinstance(protocols, Protocol):  # allow a single Protocol to be given as 'self.protocols'
#            protocols = [protocols] * len(data)
#
#        assert(len(data) == len(protocols))
#
#        subresults = {}
#        for (sub_name, sub_data), protocol in zip(data.items(), protocols):
#            protocol.name = self.name  # override protocol's name so load/save works
#            sub_results = protocol.run(sub_data)
#            subresults[sub_name] = sub_results  # something like this?
#
#        return MultiProtocolResults(data, subresults) #, root_qty_name)

#for a given set of circuits, there's structure to the circuits (ProtocolInput),
# but there can also be sub-structure to the circuits, i.e. how they're composed
# of other circuit structures if you have both, then 
            
        #else:
        #    assert(False), "We don't really want to do this..."
        #    for protocol in protocols:
        #        sub_results = protocol.run(data)
        #        results.qtys[protocol.name] = sub_results  # something like this? - or just merge qtys?
        #
        #return results


# SimultaneousProtocol -> runs multiple protocols on the same data, but "trims" circuits and data before running sub-protocols
#  (e.g. Volumetric or randomized benchmarks on different subsets of qubits) -- only accepts SimultaneousInputs.
#class SimultaneousProtocol(MultiProtocol):
#    def __init__(self, protocols, name=None):
#        super().__init__(protocols, name)
#        self.root_qty_name = 'Qubits'


def load_input_from_dir(dirname):
    dirname = _pathlib.Path(dirname)
    return _support.obj_from_meta_json(dirname / 'input').from_dir(dirname)


class ProtocolInput(_support.TreeNode):
    """ Serialize-able input data for a protocol """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None):
        dirname = _pathlib.Path(dirname)
        ret = cls.__new__(cls)
        ret.__dict__.update(_support.load_meta_based_dir(dirname / 'input', 'auxfile_types'))
        ret._init_children(dirname, 'input')
        return ret

    def __init__(self, circuits=None, qubit_labels=None,
                 children=None, children_dirs=None, child_category=None):

        self.all_circuits_needing_data = circuits if (circuits is not None) else []
        self.default_protocols = {}

        #Instructions for saving/loading certain members - if a __dict__ member
        # *isn't* listed in this dict, then it's assumed to be json-able and included
        # in the main 'meta.json' file.  Allowed values are:
        # 'text-circuit-list' - a text circuit list file
        # 'json' - a json file
        # 'pickle' - a python pickle file (use only if really needed!)
        self.auxfile_types = {'all_circuits_needing_data': 'text-circuit-list',
                              'default_protocols': 'dict-of-protocolobjs'}

        # because TreeNode takes care of its own serialization:
        self.auxfile_types.update({'_dirs': 'none', '_vals': 'none', '_childcategory': 'none'})

        if qubit_labels is None:
            if len(circuits) > 0:
                self.qubit_labels = circuits[0].line_labels
            elif children:
                self.qubit_labels = tuple(_itertools.chain(*[inp.qubit_labels for inp in children.values()]))
            else:
                self.qubit_labels = ('*',)  # default "qubit labels"
        else:
            self.qubit_labels = tuple(qubit_labels)

        if children is None: children = {} 
        children_dirs = children_dirs.copy() if (children_dirs is not None) else \
            {subname.replace(' ', '_'): subname for subname in children}

        assert(set(children.keys()) == set(children_dirs.keys()))
        super().__init__(children_dirs, children, child_category)

    def add_default_protocol(self, default_protocol_instance):
        instance_name = default_protocol_instance.name
        self.default_protocols[instance_name] = default_protocol_instance

    def write(self, dirname, parent=None):
        _support.write_obj_to_meta_based_dir(self, _pathlib.Path(dirname) / 'input', 'auxfile_types')
        self.write_children(dirname)

    def create_subdata(self, subdata_name, dataset):
        raise NotImplementedError("This protocol input cannot create any subdata!")


class CircuitListsInput(ProtocolInput):
    def __init__(self, circuit_lists, all_circuits_needing_data=None, qubit_labels=None, nested=False):

        if all_circuits_needing_data is not None:
            all_circuits = all_circuits_needing_data
        elif nested and len(circuit_lists) > 0:
            all_circuits = circuit_lists[-1]
        else:
            all_circuits = []
            for lst in circuit_lists:
                all_circuits.extend(lst)
            _lt.remove_duplicates_in_place(all_circuits)

        self.circuit_lists = circuit_lists
        self.nested = nested

        super().__init__(all_circuits, qubit_labels)
        self.auxfile_types['circuit_lists'] = 'text-circuit-lists'


class CircuitStructuresInput(CircuitListsInput):
    def __init__(self, circuit_structs, qubit_labels=None, nested=False):
        """ TODO: docstring - note that a *single* structure can be given as circuit_structs """

        #Convert a single LsGermsStruct to a list if needed:
        validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
        if isinstance(circuit_structs, validStructTypes):
            master = circuit_structs
            circuit_structs = [master.truncate(Ls=master.Ls[0:i + 1])
                               for i in range(len(master.Ls))]
            nested = True  # (by this construction)

        super().__init__([s.allstrs for s in circuit_structs], None, qubit_labels, nested)
        self.circuit_structs = circuit_structs
        self.auxfile_types['circuit_structs'] = 'pickle'


# MultiInput -> specifies multiple circuit structures on (possibly subsets of) the same data (e.g. collecting into one large dataset the data for multiple protocols)
class CombinedInput(ProtocolInput):  # for multiple inputs on the same dataset

    def __init__(self, sub_inputs, all_circuits=None, qubit_labels=None, sub_input_dirs=None,
                 interleave=False, category='Protocol'):

        if not isinstance(sub_inputs, dict):
            sub_inputs = {("**%d" % i): inp for i, inp in enumerate(sub_inputs)}

        if all_circuits is None:
            all_circuits = []
            if not interleave:
                for inp in sub_inputs.values():
                    all_circuits.extend(inp.all_circuits_needing_data)
            else:
                raise NotImplementedError("Interleaving not implemented yet")
            _lt.remove_duplicates_in_place(all_circuits)  # Maybe don't always do this?

        super().__init__(all_circuits, qubit_labels, sub_inputs, sub_input_dirs, category)

    def create_subdata(self, sub_name, dataset):
        """TODO: docstring - used to create sub-ProtocolData objects (by ProtocolData) """
        return ProtocolData(self[sub_name], dataset)


# SimultaneousInput -- specifies a "qubit structure" for each sub-input
class SimultaneousInput(ProtocolInput):
    """ TODO - need to be given sub-inputs whose circuits all act on the same set of
        qubits and are disjoint with the sets of all other sub-inputs.
    """

    @classmethod
    def from_tensored_circuits(cls, circuits, template_input, qubit_labels_per_input):
        pass #Useful??? - need to break each circuit into different parts
    # based on qubits, then copy (?) template input and just replace itself
    # all_circuits_needing_data member?

    def __init__(self, inputs, tensored_circuits=None, qubit_labels=None, category='Qubits'):
        #TODO: check that inputs don't have overlapping qubit_labels

        if qubit_labels is None:
            qubit_labels = tuple(_itertools.chain(*[inp.qubit_labels for inp in inputs]))

        if tensored_circuits is None:
            #Build tensor product of circuits
            tensored_circuits = []
            circuits_per_input = [inp.all_circuits_needing_data[:] for inp in inputs]

            #Pad shorter lists with None values
            maxLen = max(map(len, circuits_per_input))
            for lst in circuits_per_input:
                if len(lst) < maxLen: lst.extend([None] * (maxLen - len(lst)))
                
            for subcircuits in zip(*circuits_per_input):
                c = _cir.Circuit(num_lines=0, editable=True)  # Creates a empty circuit over no wires
                for subc in subcircuits:
                    if subc is not None:
                        c.tensor_circuit(subc)
                c.line_labels = qubit_labels
                c.done_editing()
                tensored_circuits.append(c)

        sub_inputs = {inp.qubit_labels: inp for inp in inputs}
        sub_input_dirs = {qlbls: '_'.join(map(str, qlbls)) for qlbls in sub_inputs}
        super().__init__(tensored_circuits, qubit_labels, sub_inputs, sub_input_dirs, category)

    def create_subdata(self, qubit_labels, dataset):
        if isinstance(dataset, _objs.MultiDataSet):
            raise NotImplementedError("SimultaneousInputs don't work with multi-pass data yet.")
        qubit_ordering = list(dataset.keys())[0].line_labels
        qubit_index = {qlabel: i for i, qlabel in enumerate(qubit_ordering)}
        sub_input = self[qubit_labels]
        qubit_indices = [qubit_index[ql] for ql in qubit_labels]  # TODO: better way to connect qubit indices in dataset with labels??
        filtered_ds = _cnst.filter_dataset(dataset, qubit_labels, qubit_indices, idle=None)  # Marginalize dataset
        return ProtocolData(sub_input, filtered_ds)


def load_data_from_dir(dirname):
    dirname = _pathlib.Path(dirname)
    return _support.obj_from_meta_json(dirname / 'data').from_dir(dirname)


class ProtocolData(_support.TreeNode):

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None):
        p = _pathlib.Path(dirname)
        inpt = parent.input[name] if parent and name else \
            load_input_from_dir(dirname)

        data_dir = p / 'data'
        #with open(data_dir / 'meta.json', 'r') as f:
        #    meta = _json.load(f)

        #Load dataset or multidataset based on what files exist
        dataset_files = sorted(list(data_dir.glob('*.txt')))
        if len(dataset_files) == 0:  # assume same dataset as parent
            if parent is None: parent = ProtocolData.from_dir(dirname / '..')
            dataset = parent.dataset
        elif len(dataset_files) == 1 and dataset_files[0].name == 'dataset.txt':  # a single dataset.txt file
            dataset = _io.load_dataset(dataset_files[0])
        else:
            raise NotImplementedError("Need to implement MultiDataSet.init_from_dict!")
            dataset = _objs.MultiDataSet.init_from_dict({pth.name: _io.load_dataset(pth) for pth in dataset_files})

        cache = _support.read_json_or_pkl_files_to_dict(data_dir / 'cache')

        ret = cls(inpt, dataset, cache)
        ret._init_children(dirname, 'data')  # loads child nodes
        return ret

    def __init__(self, protocol_input, dataset=None, cache=None):
        self.input = protocol_input
        self.dataset = dataset  # MultiDataSet allowed for multi-pass data
        self.cache = cache if (cache is not None) else {}

        if isinstance(self.dataset, _objs.MultiDataSet):
            for dsname in self.dataset:
                if dsname not in self.cache: self.cache[dsname] = {}  # create separate caches for each pass
            self._passdatas = {dsname: ProtocolData(self.input, ds, self.cache[dsname])
                               for dsname, ds in self.dataset.items()}
        else:
            self._passdatas = {None: self}
            
        super().__init__(self.input._dirs, {}, self.input._childcategory)  # children created on-demand

    def _create_childval(self, key):  # (this is how children are created on-demand)
        return self.input.create_subdata(key, self.dataset)

    @property
    def passes(self):
        return self._passdatas

    #def is_multiinput(self):
    #    return isinstance(self.input, MultiInput)
    #

    def is_multipass(self):
        return isinstance(self.dataset, _objs.MultiDataSet)

    #def get_tree_paths(self):
    #    return self.input.get_tree_paths()

    def filter_paths(self, paths, paths_are_sorted=False):
        def build_data(inp, src_data):
            """ Uses a template (filtered) input to selectively
                copy the non-input parts of a 'src_data' ProtocolData """
            ret = ProtocolData(inp, src_data.dataset, src_data.cache)
            for subname, subinput in inp.items():
                if subname in src_data._vals:  # if we've actually created this sub-data...
                    ret._vals[subname] = build_data(subinput, src_data._vals[subname])
            return ret
        filtered_input = self.input.filter_paths(paths, paths_are_sorted)
        return build_data(filtered_input, self)

    def write(self, dirname, parent=None):
        dirname = _pathlib.Path(dirname)
        data_dir = dirname / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        _support.obj_to_meta_json(self, data_dir)

        if parent is None:
            self.input.write(dirname)  # assume parent has already written input
        
        if parent and (self.dataset is parent.dataset):  # then no need to write any data
            assert(len(list(data_dir.glob('*.txt'))) == 0), "There shouldn't be *.txt files in %s!" % str(data_dir)
        else:
            data_dir.mkdir(exist_ok=True)
            if isinstance(self.dataset, _objs.MultiDataSet):
                for dsname, ds in self.dataset.items():
                    _io.write_dataset(data_dir / (dsname + '.txt'), ds)
            else:
                _io.write_dataset(data_dir / 'dataset.txt', self.dataset)

        if self.cache:
            _support.write_dict_to_json_or_pkl_files(self.cache, data_dir / 'cache')

        self.write_children(dirname, write_subdir_json=False)  # writes sub-datas


def load_results_from_dir(dirname, name=None, preloaded_data=None):
    dirname = _pathlib.Path(dirname)
    results_dir = dirname / 'results'
    if name is None:  # then it's a directory object
        return _support.obj_from_meta_json(results_dir).from_dir(dirname)
    else:  # it's a ProtocolResults object
        return _support.obj_from_meta_json(results_dir / name).from_dir(dirname, name, preloaded_data)


class ProtocolResults(object):
    @classmethod
    def from_dir(cls, dirname, name, preloaded_data=None):
        dirname = _pathlib.Path(dirname)
        ret = cls.__new__(cls)
        ret.data = preloaded_data if (preloaded_data is not None) else \
            load_data_from_dir(dirname)
        ret.__dict__.update(_support.load_meta_based_dir(dirname / 'results' / name, 'auxfile_types'))
        assert(ret.name == name), "ProtocolResults name inconsistency!"
        return ret

#        #Initialize input and data
#        if parent is not None:
#            data = preloaded_data
#        else:
#            data = load_data_from_dir(dirname)
#
#        #Initialize results
#        p = _pathlib.Path(dirname)
#        results_dir = p / 'results' / name
#
#        protocol = load_protocol_from_dir(results_dir / 'protocol')
#
#        #load qtys from results directory
#        qtys, auxfile_types = _support.load_meta_based_dir(results_dir, 'qty_auxfile_types', separate_auxfiletypes=True)
#        #TODO - how to preserve NamedDict objects??
#
#        ret = cls(name, data, protocol, qtys)
#        ret.qty_auxfile_types.update(auxfile_types)
#        return ret

    def __init__(self, data, protocol_instance):
        """root quantity seriestypes, 'root_qty_stypes' can be a (key-type, val-type) tuple or a single key-type """
        #if isinstance(root_qty_stypes, tuple):
        #    ktype, vtype = root_qty_stypes
        #else:
        #    ktype = root_qty_stypes; vtype = None
        self.name = protocol_instance.name  # just for convenience in JSON dir
        self.protocol = protocol_instance
        self.data = data
        self.auxfile_types = {'data': 'none', 'protocol': 'protocolobj'}

# self.qtys = {}
#    def items(self):
#        return self.qtys.items()
#
#    def keys(self):
#        return self.qtys.keys()
#
#    def __getitem__(self, key):
#        return self.qtys[key]
#
#    def __contains__(self, key):
#        return key in self.qtys
#
#    def __len__(self):
#        return len(self.qtys)

    def write(self, dirname, data_already_written=False):
        p = _pathlib.Path(dirname)
        results_dir = p / 'results' / self.name
        results_dir.mkdir(parents=True, exist_ok=True)

        #write input and data
        if not data_already_written:
            self.data.write(dirname)

        #write qtys to results dir
        _support.write_obj_to_meta_based_dir(self, results_dir, 'auxfile_types')

    #TODO - revamp functions below here now that _subresults is added
    def asdict(self):
        ret = {}
        for k, v in self.qtys.items():
            if isinstance(v, ProtocolResults):
                ret[k] = v.asdict()
            else:
                ret[k] = v
        return ret

    def asdataframe(self):
        return self.qtys.asdataframe()

    def __str__(self):
        import pprint
        P = pprint.PrettyPrinter()
        return P.pformat(self.asdict())


class MultiPassResults(ProtocolResults):
    def __init__(self, data, protocol_instance):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        self.passes = {}
        self.auxfile_types['passes'] = 'dict-of-resultsobjs'


class ProtocolResultsDir(_support.TreeNode):

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None):
        dirname = _pathlib.Path(dirname)
        data = parent.data[name] if (parent and name) else \
            load_data_from_dir(dirname)

        #Load results in results_dir
        results = {}
        results_dir = dirname / 'results'
        if results_dir.is_dir():  # if results_dir doesn't exist that's ok (just no results to load)
            for pth in results_dir.iterdir():
                if pth.is_dir() and (pth / 'meta.json').is_file():
                    results[pth.name] = _support.obj_from_meta_json(pth).from_dir(
                        dirname, pth.name, preloaded_data=data)

        ret = cls(data, results, {})  # don't initialize children now
        ret._init_children(dirname, meta_subdir='results')
        return ret

    def __init__(self, data, protocol_results=None, children=None):
        """ protocol_results should be a dict of ProtocolResults object whose keys are the names of the contained results """
        self.data = data  # input and data
        self.for_protocol = protocol_results.copy() if protocol_results else {}
        assert(all([r.data is self.data for r in self.for_protocol.values()]))

        #self._children = children if (children is not None) else {}
        if children is None:
            #automatically create tree based on data to hold whatever results we'll need
            # otherwise need to be able to create these lazily like ProtocolData.
            children = {}
            for subname, subdata in self.data.items():
                children[subname] = ProtocolResultsDir(subdata)
        else:
            children = children.copy()

        super().__init__(self.data.input._dirs, children, self.data.input._childcategory)

    def write(self, dirname, parent=None):
        if parent is None: self.data.write(dirname)  # assume parent has already written data
        dirname = _pathlib.Path(dirname)

        results_dir = dirname / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        _support.obj_to_meta_json(self, results_dir)

        #write the results
        for name, results in self.for_protocol.items():
            assert(results.name == name)
            results.write(dirname, data_already_written=True)

        self.write_children(dirname, write_subdir_json=False)  # writes sub-nodes
        
#     # Dictionary access into _children
#     def items(self):
#         return self._children.items()
# 
#     def keys(self):
#         return self._children.keys()
# 
#     def __getitem__(self, key):
#         return self._children[key]
# 
#     def __contains__(self, key):
#         return key in self._children
# 
#     def __len__(self):
#         return len(self._children)


    ##TODO - revamp functions below here now that _subresults is added
    #def asdict(self):
    #    ret = {}
    #    for k, v in self.qtys.items():
    #        if isinstance(v, ProtocolResults):
    #            ret[k] = v.asdict()
    #        else:
    #            ret[k] = v
    #    return ret
    #
    #def asdataframe(self):
    #    return self.qtys.asdataframe()
    #
    #def __str__(self):
    #    import pprint
    #    P = pprint.PrettyPrinter()
    #    return P.pformat(self.asdict())


#class ProtocolResultsTree(object):
#    @classmethod
#    def from_dir(cls, dirname, preloaded_data=None):
#        ret = cls.__new__(cls)
#        ret.name = name
#    
#        #Initialize input and data
#        if preloaded_data is not None:
#            ret.data = preloaded_data
#        else:
#            ret.data = load_data_from_dir(dirname)
#
#        #Traverse input tree, loading nodes at each location
#        root = ProtocolResultsTreeNode.from_dir(dirname)
#            
#        #load _subresults from other directories
#        p = _pathlib.Path(dirname)
#        ret._subresults = {}
#        if ret.data.is_multipass():
#            raise NotImplementedError()  # TODO - load in per-pass results as subresults??
#        elif ret.data.is_multiinput():
#            subname_to_subdir = {subname: subdir for subdir, subname in ret.data.input._directories.items()}
#            for sub_name, sub_data in ret.data.items():
#                subdir = subname_to_subdir[sub_name]
#                if (p / subdir / 'results' / 'meta.json').is_file():
#                    # A results object doesn't *need* to have all possible sub-results
#                    ret._subresults[sub_name] = load_results_from_dir(p / subdir, name, sub_data)
#    
#        return ret
#
#    def __init__(self, data, sub_results=None):
#        assert(data.is_multiinput()), "MultiProtocolResults is meant to hold sub-results corresponding to sub-inputs!"
#        self.data = data
#        self._subresults = sub_results if (sub_results is not None) else {}
#
#
#    def write(self, dirname, data_already_written=False):
#        p = _pathlib.Path(dirname)
#
#        #write input and data
#        if not data_already_written:
#            self.data.write(dirname)
#
#        #write _subresults to other directories
#        subname_to_subdir = {subname: subdir for subdir, subname in self.data.input._directories.items()}
#        for sub_name, sub_results in self._subresults.items():
#            subdir = subname_to_subdir[sub_name]
#            sub_results.write(p / subdir, data_already_written=True)  # avoid re-writing the data



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
        self.results = TODO

    def read(self, dirname):
        pass

    def write(self, dirname):
        pass


def run_default_protocols(data):
    return DefaultRunner().run(data)


def write_empty_protocol_data(inpt, dirname, sparse="auto"):
    dirname = _pathlib.Path(dirname)
    data_dir = dirname / 'data'
    circuits = inpt.all_circuits_needing_data
    nQubits = len(inpt.qubit_labels)
    if sparse == "auto":
        sparse = bool(nQubits > 3)  # HARDCODED

    if sparse:
        header_str = "# Note: on each line, put comma-separated <outcome:count> items, i.e. 00110:23"
        nZeroCols = 0
    else:
        fstr = '{0:0%db} count' % nQubits
        nZeroCols = 2**nQubits
        header_str = "## Columns = " + ", ".join([fstr.format(i) for i in range(nZeroCols)])

    pth = data_dir / 'dataset.txt'
    if pth.exists():
        raise ValueError("Template data file would clobber %s, which already exists!" % pth)
    data_dir.mkdir(parents=True, exist_ok=True)
    inpt.write(dirname)
    _io.write_empty_dataset(pth, circuits, header_str, nZeroCols)

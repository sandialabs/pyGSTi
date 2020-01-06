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

from .. import construction as _cnst
from .. import objects as _objs
from .. import io as _io
from ..objects import circuit as _cir
from ..tools import listtools as _lt


def full_class_name(x):
    return x.__class__.__module__ + '.' + x.__class__.__name__


class Protocol(object):
    def __init__(self):
        pass

    def run(self, data):
        if data.is_multipass():
            implicit_multipassprotocol = MultiPassProtocol(self)
            return implicit_multipassprotocol.run(data)

        elif data.is_multiinput():  # ~ is this is a directory
            # Note: we also know that self *isn't* a MultiProtocol object since
            # MultiProtocol overrides run(...).
            if isinstance(data.input, SimultaneousInput):
                implicit_multiprotocol = SimultaneousProtocol(self)
            else:
                implicit_multiprotocol = MultiProtocol(self)
            return implicit_multiprotocol.run(data)

        else:
            return self._run(data)

    def _run(self, data):
        raise NotImplementedError("Derived classes should implement this!")


# MultiProtocol -> runs, on same input circuit structure & data, multiple protocols (e.g. different GST & model tests on same GST data)
#   if that one input is a MultiInput, then it must have the same number of inputs as there are protocols and each protocol is run on the corresponding input.
#   if that one input is a normal input, then the protocols can cache information in a Results object that is handed down.
class MultiProtocol(Protocol):
    def __init__(self, protocols):
        super().__init__()
        self.protocols = protocols
        self.root_qty_name = 'Protocol'

    def run(self, data):
        assert(data.is_multiinput()), "`MultiProtocol` can only be run on ProtocolData objects with multi-input data"
        protocols = self.protocols
        if isinstance(protocols, Protocol):  # allow a single Protocol to be given as 'self.protocols'
            protocols = [protocols] * len(data)

        assert(len(data) == len(protocols))

        results = ProtocolResults(data, self.root_qty_name, 'category')
        for (sub_name, sub_data), protocol in zip(data.items(), protocols):
            sub_results = protocol.run(sub_data)
            results.qtys[sub_name] = sub_results  # something like this?
        return results


# SimultaneousProtocol -> runs multiple protocols on the same data, but "trims" circuits and data before running sub-protocols
#  (e.g. Volumetric or randomized benchmarks on different subsets of qubits) -- only accepts SimultaneousInputs.
class SimultaneousProtocol(MultiProtocol):
    def __init__(self, protocols):
        super().__init__(protocols)
        self.root_qty_name = 'Qubits'


class MultiPassProtocol(Protocol):
    # expects a MultiDataSet of passes and maybe adds data comparison (?) - probably not RB specific
    def __init__(self, protocol):
        super().__init__()
        self.protocol = protocol

    def run(self, data):
        assert(data.is_multipass()), "`MultiPassProtocol` can only be run on ProtocolData objects with multi-pass data"

        results = ProtocolResults(data, 'Pass', 'category')
        for pass_name, sub_data in data.items():  # a multipass DataProtocol object contains per-pass datas
            #TODO: print progress: pass X of Y, etc
            sub_results = self.protocol.run(sub_data)
            results.qtys[pass_name] = sub_results  # pass_name is a "ds_name" key of data.dataset (a MultiDataSet)
        return results


import importlib as _importlib
def class_for_name(module_and_class_name):
    parts = module_and_class_name.split('.')
    module_name = '.'.join(parts[0:-1])
    class_name = parts[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = _importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def load_input_from_dir(dirname):
    p = _pathlib.Path(dirname)
    with open(p / 'input' / 'meta.json', 'r') as f:
        meta = _json.load(f)
    typestr = meta['type']
    cls = class_for_name(typestr)  # class of object to create
    return cls.from_dir(dirname)


class ProtocolInput(object):
    """ Serialize-able input data for a protocol """

    #def __init__(self, default_protocol_name=None, default_protocol_info=None, circuits=None, typestring=None):
    #    if default_protocol_info is None: default_protocol_info = {}
    #    self.typestring = self.__class__.__name__
    #    self.all_circuits_needing_data = all_circuits if (all_circuits is not None) else []
    #    self.default_protocol_infos = {} if default_protocol_name is None \
    #        else {default_protocol_name: default_protocol_info}

    @classmethod
    def _get_auxfile_ext(cls, typ):
        #get expected extension
        if typ == 'text-circuit-list': ext = '.txt'
        elif typ == 'text-circuit-lists': ext = '.txt'
        elif typ == 'json': ext = '.json'
        elif typ == 'pickle': ext = '.pkl'
        elif typ == 'none': ext = '.NONE'
        else: raise ValueError("Invalid aux-file type: %s" % typ)
        return ext

    @classmethod
    def from_dir(cls, dirname):
        p = _pathlib.Path(dirname)
        input_dir = p / 'input'

        with open(input_dir / 'meta.json', 'r') as f:
            meta = _json.load(f)

        ret = cls.__new__(cls)
        for key, val in meta.items():
            if key == 'type': continue  # ignore - this is just the class name
            ret.__dict__[key] = val

        for key, typ in meta['auxfile_types'].items():
            if typ == 'none': continue  # no file exists for this member
            ext = cls._get_auxfile_ext(typ)

            #Process cases with non-standard expected path(s)
            if typ == 'text-circuit-lists':
                i = 0; val = []
                while True:
                    pth = input_dir / (key + str(i) + ext)
                    if not pth.exists(): break
                    val.append(_io.load_circuit_list(pth)); i += 1

            else:

                # cases with 'standard' expected path exists
                pth = input_dir / (key + ext)
                if not (pth.exists() and not pth.is_dir()):
                    raise ValueError("Expected path: %s does not exist or is a dir!" % pth)

                #load value into object
                if typ == 'text-circuit-list':
                    val = _io.load_circuit_list(pth)
                elif typ == 'json':
                    with open(pth, 'r') as f:
                        val = _json.load(f)
                elif typ == 'pickle':
                    with open(pth, 'rb') as f:
                        val = _pickle.load(f)
                else:
                    raise ValueError("Invalid aux-file type: %s" % typ)

            ret.__dict__[key] = val
        return ret

    def __init__(self, circuits=None, qubit_labels=None):
        self.all_circuits_needing_data = circuits if (circuits is not None) else []
        self.default_protocol_infos = {}

        #Instructions for saving/loading certain members - if a __dict__ member
        # *isn't* listed in this dict, then it's assumed to be json-able and included
        # in the main 'meta.json' file.  Allowed values are:
        # 'text-circuit-list' - a text circuit list file
        # 'json' - a json file
        # 'pickle' - a python pickle file (use only if really needed!)
        self.auxfile_types = {'all_circuits_needing_data': 'text-circuit-list'}

        if qubit_labels is None:
            if len(circuits) > 0:
                self.qubit_labels = circuits[0].line_labels
            else:
                self.qubit_labels = ('*',)  # default "qubit labels"
        else:
            self.qubit_labels = tuple(qubit_labels)

    def add_default_protocol(self, default_protocol_name, default_protocol_info=None):
        if default_protocol_info is None: default_protocol_info = {}
        self.default_protocol_infos[default_protocol_name] = default_protocol_info

    #TODO REMOVE
    #def create_circuit_list(self, verbosity=0):
    #    return self.basedata['circuitList']
    #
    #def create_circuit_lists(self, verbosity=0):  # Needed?? / Helpful?
    #    return [self.create_circuit_list()]        

    def write(self, dirname):
        p = _pathlib.Path(dirname)
        input_dir = p / 'input'

        p.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(exist_ok=True)

        meta = {'type': full_class_name(self)}
        for key, val in self.__dict__.items():
            if key in self.auxfile_types: continue  # member is serialized to a separate file (see below)
            meta[key] = val
        with open(input_dir / 'meta.json', 'w') as f:
            _json.dump(meta, f)

        for auxnm, typ in self.auxfile_types.items():
            val = self.__dict__[auxnm]
            ext = self._get_auxfile_ext(typ)

            if typ == 'text-circuit-lists':
                for i, circuit_list in enumerate(val):
                    pth = input_dir / (auxnm + str(i) + ext)
                    _io.write_circuit_list(pth, circuit_list)

            else:
                # standard path cases
                pth = input_dir / (auxnm + ext)

                if typ == 'text-circuit-list':
                    _io.write_circuit_list(pth, val)
                elif typ == 'json':
                    with open(pth, 'w') as f:
                        _json.dump(val, f)
                elif typ == 'pickle':
                    with open(pth, 'wb') as f:
                        _pickle.dump(val, f)
                elif typ == 'none':
                    pass
                else:
                    raise ValueError("Invalid aux-file type: %s" % typ)

    def get_tree_paths(self):
        """Dictionary paths leading to "terminal"/"leaf" input objects beneath this one"""
        return [()]  # a single 0-length path, pointing to this terminal input itself

    def view(self, keys_to_keep):
        #Can only take views of "simple" (non-multi) inputs when there are no keys to keep
        if len(keys_to_keep) == 0:
            return self
        else:
            raise ValueError("Cannot take view w/keys=%s (I have no sub-inputs!)" % str(keys_to_keep))

    def filter_paths(self, paths, paths_are_sorted=False):
        if len(paths) == 1 and len(paths[0]) == 0:
            return self
        else:
            raise ValueError("Cannot filter paths: %s (I have no sub-inputs!)" % str(paths))

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
class MultiInput(ProtocolInput):  # for multiple inputs on the same dataset

    @classmethod
    def from_dir(cls, dirname):
        ret = super().from_dir(dirname)  # Note: super() != ProtocolInput - this passes *`cls`* to base class method
        for subdir in ret._directories:  # convert subname lists->tuples (json does opposite!)
            if isinstance(ret._directories[subdir], list):  # (we never want list-type subnames b/c we must hash them)
                ret._directories[subdir] = tuple(ret._directories[subdir])

        p = _pathlib.Path(dirname)
        ret._subinputs = {subname: load_input_from_dir(p / subdir)
                          for subdir, subname in ret._directories.items()}
        return ret

    def __init__(self, sub_inputs, all_circuits=None, qubit_labels=None, sub_input_dirs=None):
        if not isinstance(sub_inputs, dict):
            sub_inputs = {("**%d" % i): inp for i, inp in enumerate(sub_inputs)}

        if all_circuits is None:
            all_circuits = []
            for inp in sub_inputs.values():
                all_circuits.extend(inp.all_circuits_needing_data)
            _lt.remove_duplicates_in_place(all_circuits)  #Maybe don't always do this?

        if qubit_labels is None:
            qubit_labels = tuple(_itertools.chain(*[inp.qubit_labels for inp in sub_inputs.values()]))

        super().__init__(all_circuits, qubit_labels)
        self._subinputs = sub_inputs
        if sub_input_dirs is None:
            self._directories = {subname.replace(' ','_'): subname for subname in sub_inputs.keys()}
        else:
            self._directories = sub_input_dirs
        self.auxfile_types['_subinputs'] = "none"

    def create_subdata(self, sub_name, dataset):
        """TODO: docstring - used to create sub-ProtocolData objects (by ProtocolData) """
        sub_input = self[sub_name]
        return ProtocolData(sub_input, dataset)

    def items(self):
        return self._subinputs.items()

    def keys(self):
        return self._subinputs.keys()

    def __iter__(self):
        return self._subinputs.__iter__()

    def __getitem__(self, key):
        return self._subinputs[key]

    def __contains__(self, key):
        return key in self._subinputs

    def __len__(self):
        return len(self._subinputs)

    def write(self, dirname):
        p = _pathlib.Path(dirname)
        super().write(dirname)
        for subdir, subname in self._directories.items():
            outdir = p / subdir
            outdir.mkdir(exist_ok=True)
            self._subinputs[subname].write(outdir)

    def get_tree_paths(self):
        paths = []
        for subname, subinput in self.items():
            paths.extend([(subname,) + pth for pth in subinput.get_tree_paths()])
        return paths

    def view(self, keys_to_keep):
        input_view = _copy.deepcopy(self)  # copies type of multi-input
        input_view._subinputs = {k: input_view[k] for k in keys_to_keep}
        return input_view

    def filter_paths(self, paths, paths_are_sorted=False):
        sorted_paths = paths if paths_are_sorted else sorted(paths)
        nPaths = len(sorted_paths)

        if nPaths == 1 and len(sorted_paths[0]) == 0:
            return self  # special case when this MultiInput itself is selected

        i = 0
        subinputs = {}
        while i < nPaths:
            assert(len(sorted_paths[i]) > 0), \
                "Cannot select a MultiInput *and* some/all of its elements using filter_paths!"
            ky = sorted_paths[i][0]

            paths_starting_with_ky = []
            while i < nPaths and sorted_paths[i][0] == ky:
                paths_starting_with_ky.append(sorted_paths[i][1:])
                i += 1
            subinputs[ky] = self[ky].filter_paths(paths_starting_with_ky, True)

        assert(len(subinputs) > 0)  # Multi-Inputs should always have at least one sub-input

        input_view = _copy.deepcopy(self)  # copies type of multi-input
        input_view._subinputs = subinputs
        return input_view


# SimultaneousInput -- specifies a "qubit structure" for each sub-input
class SimultaneousInput(MultiInput):
    """ TODO - need to be given sub-inputs whose circuits all act on the same set of
        qubits and are disjoint with the sets of all other sub-inputs.
    """

    @classmethod
    def from_tensored_circuits(cls, circuits, template_input, qubit_labels_per_input):
        pass #Useful??? - need to break each circuit into different parts
    # based on qubits, then copy (?) template input and just replace itself
    # all_circuits_needing_data member?
    
    def __init__(self, inputs, tensored_circuits=None, qubit_labels=None):
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
        sub_input_dirs = {'_'.join(map(str, qlbls)): qlbls for qlbls in sub_inputs}
        super().__init__(sub_inputs, tensored_circuits, qubit_labels, sub_input_dirs)

    def get_structure(self):  #TODO: USED??
        return list(self.keys())  # a list of qubit-label tuples

    def create_subdata(self, qubit_labels, dataset):
        qubit_ordering = list(dataset.keys())[0].line_labels
        qubit_index = {qlabel: i for i, qlabel in enumerate(qubit_ordering)}
        sub_input = self[qubit_labels]
        qubit_indices = [qubit_index[ql] for ql in qubit_labels]  # TODO: better way to connect qubit indices in dataset with labels??
        filtered_ds = _cnst.filter_dataset(dataset, qubit_labels, qubit_indices, idle=None)  # Marginalize dataset
        return ProtocolData(sub_input, filtered_ds)


def load_data_from_dir(dirname):
    p = _pathlib.Path(dirname)
    with open(p / 'data' / 'meta.json', 'r') as f:
        meta = _json.load(f)
    typestr = meta['type']
    cls = class_for_name(typestr)  # class of object to create
    return cls.from_dir(dirname)


class ProtocolData(object):

    @classmethod
    def from_dir(cls, dirname, parent_data=None, load_subdatas=True):
        p = _pathlib.Path(dirname)
        data_dir = p / 'data'

        with open(data_dir / 'meta.json', 'r') as f:
            meta = _json.load(f)
        dtype = meta['dataset_type']

        ret = cls.__new__(cls)
        ret.input = load_input_from_dir(dirname)

        if dtype == 'normal':
            ret.dataset = _io.load_dataset(data_dir / 'dataset.txt')
        elif dtype == 'multi-single-file':
            ret.dataset = _io.load_multidataset(data_dir / 'dataset.txt')
        elif dtype == 'multi-multiple-files':
            raise NotImplementedError()  # should load in several datasetX.txt files as elements of a MultiDataSet
        elif dtype == 'same-as-parent':
            if parent_data is None:
                parent_data = ProtocolData.from_dir(dirname / '..', load_subdatas=False)
            ret.dataset = parent_data.dataset
        else:
            raise ValueError("Invalid dataset type: %s" % dtype)

        ret.cache = {}
        cache_dir = data_dir / 'cache'
        if cache_dir.is_dir():
            for pth in cache_dir.iterdir():
                if pth.suffix == '.json':
                    with open(pth, 'r') as f:
                        val = _json.load(f)
                elif pth.suffix == '.pkl':
                    with open(pth, 'r') as f:
                        val = _pickle.load(f)
                else:
                    continue  # ignore cache file times we don't understand
                
                ret.cache[pth.name] = val

        ret._subdatas = {}
        if load_subdatas:
            if isinstance(ret.dataset, _objs.MultiDataSet):
                for subname, sub_ds in ret.dataset.items():
                    ret._subdatas[subname] = cls.from_dir(p / subname, parent_data=ret)
            elif isinstance(ret.input, MultiInput):
                for subdir, subname in ret.input._directories.items():
                    if (p / subdir / 'data').exists():
                        ret._subdatas[subname] = cls.from_dir(p / subdir, parent_data=ret)
                    #It's ok if not all data exists - it is created on-demand
            else: # no subdata to load
                ret._subdatas = None
        return ret

    def __init__(self, protocol_input, dataset=None, cache=None):
        self.input = protocol_input
        self.dataset = dataset  # MultiDataSet allowed for multi-pass data
        self.cache = cache if (cache is not None) else {}
        if isinstance(dataset, _objs.MultiDataSet):
            self._subdatas = {dsname: ProtocolData(self.input, ds) for dsname, ds in dataset.items()}
        elif isinstance(protocol_input, MultiInput):
            self._subdatas = {}
        else:
            self._subdatas = None
        #Note: if a ProtocolData is given a multi-Input and multi-DataSet, then the sub-datas are
        # for the different *passes* and not the inputs - i.e. multi-data takes precedence over multi-input.

    def is_multiinput(self):
        return isinstance(self.input, MultiInput) and not isinstance(self.dataset, _objs.MultiDataSet)

    def is_multipass(self):
        return isinstance(self.dataset, _objs.MultiDataSet)

    def get_tree_paths(self):
        return self.input.get_tree_paths()

    def filter_paths(self, paths, paths_are_sorted=False):
        assert(not self.is_multipass()), "Can't take views of multi-pass data yet."

        def build_data(inp, src_data):
            """ Uses a template (filtered) input to selectively
                copy the non-input parts of a 'src_data' ProtoolData """
            ret = ProtocolData(inp, src_data.dataset, src_data.cache)
            if ret.is_multiinput():
                for subname, subinput in inp.items():
                    ret._subdatas[subname] = build_data(subinput, src_data._subdatas[subname])
            return ret

        filtered_input = self.input.filter_paths(paths, paths_are_sorted)
        return build_data(filtered_input, self)

    #Support lazy evaluation of sub_datas
    def keys(self):
        if self.is_multipass():
            return self._subdatas.keys()
        elif self.is_multiinput():
            return self.input.keys()
        else:
            return []

    def items(self):
        if self.is_multipass():
            for passname, val in self._subdatas.items():
                yield passname, val
        elif self.is_multiinput():
            for name in self.input.keys():
                yield name, self[name]

    def __iter__(self):
        if self.is_multipass():
            return self._subdatas.__iter__()
        elif self.is_multiinput():
            return self.input.__iter__()
        else:
            return [].__iter__()

    def __getitem__(self, subdata_name):
        if self.is_multipass():
            return self._subdatas[subdata_name]
        elif self.is_multiinput():  #can compute lazily
            if subdata_name not in self._subdatas:
                if subdata_name not in self.input.keys():
                    raise KeyError("Missing sub-data name: %s" % subdata_name)
                self._subdatas[subdata_name] = self.input.create_subdata(subdata_name, self.dataset)
            return self._subdatas[subdata_name]
        else:
            raise ValueError("This ProtocolData object has no sub-datas.")

    #def __setitem__(self, subdata_name, val):
    #    self._subdatas[subdata_name] = val

    def __contains__(self, subdata_name):
        if self.is_multipass():
            return subdata_name in self._subdatas
        elif self.is_multiinput():
            return subdata_name in self.input.keys()
        else:
            return False

    def __len__(self):
        if self.is_multipass():
            return len(self._subdatas)
        elif self.is_multiinput():
            return len(self.input)
        else:
            return 0

    def write(self, dirname, parent_data=None):
        p = _pathlib.Path(dirname)
        data_dir = p / 'data'

        p.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        self.input.write(dirname)

        if parent_data and self.dataset is parent_data.dataset:
            dtype = 'same-as-parent'
        elif isinstance(self.dataset, _objs.MultiDataSet):
            dtype = 'multi-single-file'
            _io.write_multidataset(data_dir / 'dataset.txt', self.dataset)
            #dtype = 'multi-multiple-files'  # TODO: how to decide between these?
        else:
            dtype = 'normal'
            _io.write_dataset(data_dir / 'dataset.txt', self.dataset)

        meta = {'type': full_class_name(self), 'dataset_type': dtype}
        with open(data_dir / 'meta.json', 'w') as f:
            _json.dump(meta, f)

        if self.cache:
            cache_dir = data_dir / 'cache'
            cache_dir.mkdir(exist_ok=True)
            for key, val in self.cache.items():
                try:
                    with open(cache_dir / (key + '.json'), 'w') as f:
                        _json.dump(val, f)
                except:
                    #remove json file??
                    with open(cache_dir / (key + '.pkl'), 'wb') as f:
                        _pickle.dump(val, f)

        if isinstance(self.dataset, _objs.MultiDataSet):
            for subname, sub_ds in self.dataset.items():
                self._subdatas[subname].write(p / subname, parent_data=self)
        elif isinstance(self.input, MultiInput):
            name_to_dir = {subname: subdir for subdir, subname in self.input._directories.items()}
            for subname, subdata in self._subdatas.items():
                subdir = name_to_dir[subname]
                subdata.write(p / subdir, parent_data=self)


class NamedDict(dict):
    def __init__(self, name=None, keytype=None, valtype=None, items=()):
        super().__init__(items)
        self.name = name
        self.keytype = keytype
        self.valtype = valtype

    def __reduce__(self):
        return (NamedDict, (self.name, self.keytype, self.valtype, list(self.items())), None)

    def asdataframe(self):
        import pandas as _pandas

        columns = {'value': []}
        seriestypes = {'value': "unknown"}
        self._add_to_columns(columns, seriestypes, {})

        columns_as_series = {}
        for colname, lst in columns.items():
            seriestype = seriestypes[colname]
            if seriestype == 'float':
                s = _np.array(lst, dtype='d')
            elif seriestype == 'int':
                s = _np.array(lst, dtype=int)  # or pd.Series w/dtype?
            elif seriestype == 'category':
                s = _pandas.Categorical(lst)
            else:
                s = lst  # will infer an object array?

            columns_as_series[colname] = s

        df = _pandas.DataFrame(columns_as_series)
        return df

    def _add_to_columns(self, columns, seriestypes, row_prefix):
        nm = self.name
        if nm not in columns:
            #add column; assume 'value' is always a column
            columns[nm] = [None] * len(columns['value'])
            seriestypes[nm] = self.keytype
        elif seriestypes[nm] != self.keytype:
            seriestypes[nm] = None  # conflicting types, so set to None

        row = row_prefix.copy()
        for k, v in self.items():
            row[nm] = k
            if isinstance(v, NamedDict):
                v._add_to_columns(columns, seriestypes, row)
            elif isinstance(v, ProtocolResults):
                v.qtys._add_to_columns(columns, seriestypes, row)
            else:
                #Add row
                complete_row = row.copy()
                complete_row['value'] = v
                
                if seriestypes['value'] == "unknown":
                    seriestypes['value'] = self.valtype
                elif seriestypes['value'] != self.valtype:
                    seriestypes['value'] = None  # conflicting type, so set to None
                    
                for rk, rv in complete_row.items():
                    columns[rk].append(rv)


class ProtocolResults(ProtocolData):
    def __init__(self, data, root_qty_name='ROOT', root_qty_stypes=None):
        super().__init__(data.input, data.dataset)
        if isinstance(root_qty_stypes, tuple):
            ktype, vtype = root_qty_stypes
        else:
            ktype = root_qty_stypes; vtype = None
        self.qtys = NamedDict(root_qty_name, ktype, vtype)

    def items(self):
        return self.qtys.items()

    def keys(self):
        return self.qtys.keys()
    
    def __getitem__(self, key):
        return self.qtys[key]

    def __contains__(self, key):
        return key in self.qtys

    def __len__(self):
        return len(self.qtys)

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

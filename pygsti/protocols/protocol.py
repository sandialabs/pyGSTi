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
    """
    Load a :class:`Protocol` from a directory on disk.

    Parameters
    ----------
    dirname : string
        Directory name.

    Returns
    -------
    Protocol
    """
    dirname = _pathlib.Path(dirname)
    return _support.obj_from_meta_json(dirname).from_dir(dirname)


class Protocol(object):
    """
    A Protocol object represents things like, but not strictly limited to, QCVV protocols.
    This class is essentially a serializable `run` function that takes as input a
    :class:`ProtocolData` object and returns a :class:`ProtocolResults` object.  This
    function describes the working of the "protocol".
    """

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
    """
    A simple protocol that runs a given contained :class:`Protocol` on
    all the passes within a :class:`ProtocolData` object that contains
    a :class:`MultiDataSet`.  Instances of this class essentially act as
    wrappers around other protocols enabling them to handle multi-pass
    data.
    """

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


class ProtocolRunner(object):
    """
    A a :class:`ProtocolRunner` object is used to combine multiple calls to :method:`Protocol.run`,
    that is, to run potentially multiple protocols on potentially different data.  From the outside,
    a :class:`ProtocolRunner` object behaves similarly, and can often be used interchangably,
    with a Protocol object.  It posesses a `run` method that takes a :class:`ProtocolData`
    as input and returns a :class:`ProtocolResultsDir` that can contain multiple :class:`ProtocolResults`
    objects within it.
    """
    
    def run(self, data):
        raise NotImplementedError()


class TreeRunner(ProtocolRunner):
    """
    Runs specific protocols on specific data-tree paths.
    """
    
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
    """ 
    Runs a single protocol on every data node that has no sub-nodes (possibly separately for each pass).
    """
    def __init__(self, protocol, protocol_can_handle_multipass_data=False, input_type='all'):
        self.protocol = protocol
        self.input_type = input_type
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
            elif self.input_type == 'all' or isinstance(node.data.input, self.input_type):
                node.for_protocol[self.protocol.name] = self.protocol.run(node.data)
            else:
                pass  # don't run on this node, since the input is the wrong type
        visit_node(ret)
        return ret


class DefaultRunner(ProtocolRunner):
    """
    Run the default protocol at each data-tree node.  (Default protocols
    are given within :class:`ProtocolInput` objects.)
    """

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


def load_input_from_dir(dirname):
    """
    Load a :class:`ProtocolInput` from a directory on disk.

    Parameters
    ----------
    dirname : string
        Directory name.

    Returns
    -------
    ProtocolInput
    """
    dirname = _pathlib.Path(dirname)
    return _support.obj_from_meta_json(dirname / 'input').from_dir(dirname)


class ProtocolInput(_support.TreeNode):
    """
    An input-data specification for one or more QCVV protocols.

    The "input" needed to collect data to run a :class:`Protocol`.  Minimally,
    a :class:`ProtocolInput`  object holds a list of :class:`Circuit`s that need
    to be run.  Typically, a :class:`ProtocolInput` object also contains
    information used to interpret these circuits, either by describing how they
    are constructed from smaller pieces or how they are drawn from a distribution.

    It's important to note that a :class:`ProtocolInput` does *not* contain all the
    inputs needed to run any particular QCVV protocol (e.g. there may be additional
    parameters specified when creating a :class:`Protocol` object, and it may be the
    case that the data described by a single :class:`ProtocolInput` can be used by
    muliple protocols).  Rather, a :class:`ProtocolInput` specifies what is necessary
    to acquire and interpret the *data* needed for one or more QCVV protocols.
    """

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
    """
    Protocol input-data specification that is comprised of multiple circuit lists.
    """
    
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
    """
    Protocol input-data specification that is comprised of multiple circuit
    structures (:class:`CircuitStructure` objects).
    """

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
    """
    Protocol input-data specification that combines the input specifications of
    one or more "sub-inputs".  The sub-inputs are preserved as children under
    the :class:`CombinedInput` instance, creating a "data-tree" structure.  The
    :class:`CombinedInput` object itself simply merges all of the circuit lists.
    """

    def __init__(self, sub_inputs, all_circuits=None, qubit_labels=None, sub_input_dirs=None,
                 interleave=False, category='InputBranch'):

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
    """
    A protocol input-data specification whose circuits are the tensor-products
    of the circuits from one or more  :class:`ProtocolInput` objects that
    act on disjoint sets of qubits.  The sub-inputs are preserved as children under
    the :class:`SimultaneousInput` instance, creating a "data-tree" structure.
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
    """
    Load a :class:`ProtocolData` from a directory on disk.

    Parameters
    ----------
    dirname : string
        Directory name.

    Returns
    -------
    ProtocolData
    """
    dirname = _pathlib.Path(dirname)
    return _support.obj_from_meta_json(dirname / 'data').from_dir(dirname)


class ProtocolData(_support.TreeNode):
    """
    A :class:`ProtocolData` object represents the experimental data needed to
    run one or more QCVV protocols.  This class contains a :class:`ProtocolIput`,
    which describes a set of circuits, and a :class:`DataSet` (or :class:`MultiDataSet`)
    that holds data for these circuits.  These members correspond to the `.input`
    and `.dataset` attributes.
    """

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
    """
    Load a :class:`ProtocolResults` or :class:`ProtocolsResultsDir` from a
    directory on disk (depending on whether `name` is given).

    Parameters
    ----------
    dirname : string
        Directory name.  This should be a "base" directory, containing
        subdirectories like "input", "data", and "results"


    name : string or None
        The 'name' of a particular :class:`ProtocolResults` object, which
        is a sub-directory beneath `dirname/results/`.  If None, then *all*
        the results (all names) at the given base-directory are loaded and
        returned as a :class:`ProtocolResultsDir` object.

    Returns
    -------
    ProtocolResults or ProtocolResultsDir
    """
    dirname = _pathlib.Path(dirname)
    results_dir = dirname / 'results'
    if name is None:  # then it's a directory object
        cls = _support.obj_from_meta_json(results_dir) if (results_dir / 'meta.json').exists() \
            else ProtocolResultsDir  # default if no meta.json (if only a results obj has been written inside dir)
        return cls.from_dir(dirname)
    else:  # it's a ProtocolResults object
        return _support.obj_from_meta_json(results_dir / name).from_dir(dirname, name, preloaded_data)


class ProtocolResults(object):
    """
    A :class:`ProtocolResults` object contains a :class:`ProtocolData` object
    and stores the results from running a QCVV protocol (a :class:`Protcocol`)
    on this data.
    """
    
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

    def as_nameddict(self):
        #This function can be overridden by derived classes - this just
        # tries to give a decent default implementation
        ret = _support.NamedDict('Qty', 'category')
        ignore_members = ('name', 'protocol', 'data', 'auxfile_types')
        for k, v in self.__dict__.items():
            if k.startswith('_') or k in ignore_members: continue
            if isinstance(v, ProtocolResults):
                ret[k] = v.as_nameddict()
            elif isinstance(v, _support.NamedDict):
                ret[k] = v
            elif isinstance(v, dict):
                pass  # don't know how to make a dict into a (nested) NamedDict
            else:  # non-dicts are ok to just store
                ret[k] = v
        return ret

    def as_dataframe(self):
        return self.as_nameddict().as_dataframe()

    def __str__(self):
        import pprint
        P = pprint.PrettyPrinter()
        return P.pformat(self.as_nameddict())


class MultiPassResults(ProtocolResults):
    """
    Holds the results of a single protocol on multiple "passes"
    (sets of data, typically taken at different times).  The results
    of each pass are held as a separate :class:`ProtcolResults` object
    within the `.passes` attribute.
    """

    def __init__(self, data, protocol_instance):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        self.passes = {}
        self.auxfile_types['passes'] = 'dict-of-resultsobjs'


class ProtocolResultsDir(_support.TreeNode):
    """
    A :class:`ProtocolResultsDir` holds a dictionary of :class:`ProtocolResults`
    objects.  It contains a :class:`ProtocolData` object and is rooted at the_model
    corresponding node of the data-tree.  It contains links to child-:class:`ProtocolResultsDir`
    objects representing sub-directories.
    """

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

    def as_nameddict(self):
        sub_results = {k: v.as_nameddict() for k, v in self.items()}
        results_on_this_node = _support.NamedDict('Protocol Instance', 'category',
                                                  items={k: v.as_nameddict() for k, v in self.for_protocol.items()})
        if sub_results:
            category = self.child_category if self.child_category else 'nocategory'
            ret = _support.NamedDict(category, 'category')
            if results_on_this_node:
                #Results in this (self's) dir don't have a value for the sub-category, so put None
                ret[None] = results_on_this_node
            ret.update(sub_results)
            return ret
        else:  # no sub-results, so can just return a dict of results on this node
            return results_on_this_node

    def as_dataframe(self):
        return self.as_nameddict().as_dataframe()

    def __str__(self):
        import pprint
        P = pprint.PrettyPrinter()
        return P.pformat(self.as_nameddict())


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


class ProtocolPostProcessor(object):
    """
    A :class:`ProtocolPostProcessor` is similar to a protocol, but runs on an
    *existing* results object, and produces a new (updated?) Results object.
    """

    #Note: this is essentially a duplicate of the Protocol class (except run takes a results object)
    # but it's conceptually a different thing...  Should we derive it from Protocol?
    
    @classmethod
    def from_dir(cls, dirname):  # same I/O pattern as Protocol
        ret = cls.__new__(cls)
        ret.__dict__.update(_support.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types'))
        ret._init_unserialized_attributes()
        return ret

    def __init__(self, name):
        super().__init__()
        self.name = name if name else self.__class__.__name__
        self.auxfile_types = {}

    def _init_unserialized_attributes(self):
        pass

    def run(self, results):
        #Maybe these could also take data objects and run protocols on them automatically?
        #Returned Results object should be rooted at place of given results/resultsdir
        raise NotImplementedError("Derived classes should implement this!")

    def write(self, dirname):
        _support.write_obj_to_meta_based_dir(self, dirname, 'auxfile_types')


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
import collections as _collections

from .treenode import TreeNode as _TreeNode
from .. import construction as _cnst
from .. import objects as _objs
from .. import io as _io
from ..objects import circuit as _cir
from ..tools import listtools as _lt
from ..tools import NamedDict as _NamedDict


class Protocol(object):
    """
    A Protocol object represents things like, but not strictly limited to, QCVV protocols.
    This class is essentially a serializable `run` function that takes as input a
    :class:`ProtocolData` object and returns a :class:`ProtocolResults` object.  This
    function describes the working of the "protocol".
    """

    @classmethod
    def from_dir(cls, dirname, quick_load=False):
        """
        Initialize a new Protocol object from `dirname`.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Parameters
        ----------
        dirname : str
            The directory name.

        Returns
        -------
        Protocol
        """
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types', quick_load=quick_load))
        ret._init_unserialized_attributes()
        return ret

    def __init__(self, name=None):
        """
        Create a new Protocol object.

        Parameters
        ----------
        name : str, optional
            The name of this protocol, also used to (by default) name the
            results produced by this protocol.  If None, the class name will
            be used.

        Returns
        -------
        Protocol
        """
        super().__init__()
        self.name = name if name else self.__class__.__name__
        self.tags = {}  # string-values (key,val) pairs that serve to label this protocol instance
        self.auxfile_types = {}
        self._nameddict_attributes = ()  # (('name', 'ProtocolName', 'category'),) implied in setup_nameddict

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        ProtocolResults
        """
        raise NotImplementedError("Derived classes should implement this!")

    def write(self, dirname):
        """
        Write this protocol to a directory.

        Parameters
        ----------
        dirname : str
            The directory name to write.  This directory will be created
            if needed, and the files in an existing directory will be
            overwritten.

        Returns
        -------
        None
        """
        _io.write_obj_to_meta_based_dir(self, dirname, 'auxfile_types')

    def setup_nameddict(self, final_dict):
        """
        Initializes a set of nested :class:`NamedDict` dictionaries describing this protocol.

        This function is used by :class:`ProtocolResults` objects when they're creating
        nested dictionaries of their contents.  This function returns a set of nested,
        single (key,val)-pair named-dictionaries which describe the particular attributes
        of this :class:`Protocol` object named within its `self._nameddict_attributes` tuple.
        The final nested dictionary is set to be `final_dict`, which allows additional result
        quantities to easily be added.

        Parameters
        ----------
        final_dict : NamedDict
            the final-level (innermost-nested) NamedDict in the returned nested dictionary.

        Returns
        -------
        NamedDict
        """
        keys_vals_types = [('ProtocolName', self.name, 'category'),
                           ('ProtocolType', self.__class__.__name__, 'category')]
        keys_vals_types.extend(_convert_nameddict_attributes(self))
        keys_vals_types.extend([(k, v, 'category') for k, v in self.tags.items()])
        return _NamedDict.create_nested(keys_vals_types, final_dict)

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
        """
        Create a new MultiPassProtocol object.

        Parameters
        ----------
        protocol : Protocol
            The protocol to run on each pass.

        name : str, optional
            The name of this protocol, also used to (by default) name the
            results produced by this protocol.  If None, the class name will
            be used.

        Returns
        -------
        MultiPassProtocol
        """
        if name is None: name = protocol.name + "_multipass"
        super().__init__(name)
        self.protocol = protocol
        self.auxfile_types['protocol'] = 'protocolobj'

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        MultiPassResults
        """
        results = MultiPassResults(data, self)
        for pass_name, sub_data in data.passes.items():  # a multipass DataProtocol object contains per-pass datas
            #TODO: print progress: pass X of Y, etc
            sub_results = self.protocol.run(sub_data, memlimit, comm)
            # TODO: maybe blank-out the .data and .protocol of sub_results since we don't need this info?
            #  or call as_dict?
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

    def run(self, data, memlimit=None, comm=None):
        """
        Run all the protocols specified by this protocol-runner on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this
            protocol-runner in parallel.

        Returns
        -------
        ProtocolResultsDir
        """
        raise NotImplementedError()


class TreeRunner(ProtocolRunner):
    """
    Runs specific protocols on specific data-tree paths.
    """

    def __init__(self, protocol_dict):
        """
        Create a new TreeRunner object, which runs specific protocols on
        specific data-tree paths.

        Parameters
        ----------
        protocol_dict : dict
            A dictionary of :class:`Protocol` objects whose keys are paths
            (tuples of strings) specifying where in the data-tree that
            protocol should be run.

        Returns
        -------
        TreeRunner
        """
        self.protocols = protocol_dict

    def run(self, data, memlimit=None, comm=None):
        """
        Run all the protocols specified by this protocol-runner on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this
            protocol-runner in parallel.

        Returns
        -------
        ProtocolResultsDir
        """
        ret = ProtocolResultsDir(data)  # creates entire tree of nodes
        for path, protocol in self.protocols.items():

            root = ret
            for el in path:  # traverse path
                root = root[el]
            root.for_protocol[protocol.name] = protocol.run(root.data, memlimit, comm)  # run the protocol

        return ret


class SimpleRunner(ProtocolRunner):
    """
    Runs a single protocol on every data node that has no sub-nodes (possibly separately for each pass).
    """

    def __init__(self, protocol, protocol_can_handle_multipass_data=False, edesign_type='all'):
        """
        Create a new SimpleRunner object, which runs a single protocol on every
        'leaf' of the data-tree.

        Parameters
        ----------
        protocol : Protocol
            The protocol to run.

        protocol_can_handle_multipass_data : bool, optional
            Whether `protocol` is able to process multi-pass data, or
            if :class:`MultiPassProtocol` objects should be created
            implicitly.

        edesign_type : type or 'all'
            Only run `protocol` on leaves with this type.  (If 'all', then
            no filtering is performed.)

        Returns
        -------
        SimpleRunner
        """
        self.protocol = protocol
        self.edesign_type = edesign_type
        self.do_passes_separately = not protocol_can_handle_multipass_data

    def run(self, data, memlimit=None, comm=None):
        """
        Run all the protocols specified by this protocol-runner on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this
            protocol-runner in parallel.

        Returns
        -------
        ProtocolResultsDir
        """
        ret = ProtocolResultsDir(data)  # creates entire tree of nodes

        def visit_node(node):
            if len(node.data) > 0:
                for subname, subnode in node.items():
                    visit_node(subnode)
            elif node.data.is_multipass() and self.do_passes_separately:
                implicit_multipassprotocol = MultiPassProtocol(self.protocol)
                node.for_protocol[implicit_multipassprotocol.name] = \
                    implicit_multipassprotocol.run(node.data, memlimit, comm)
            elif self.edesign_type == 'all' or isinstance(node.data.edesign, self.edesign_type):
                node.for_protocol[self.protocol.name] = self.protocol.run(node.data, memlimit, comm)
            else:
                pass  # don't run on this node, since the experiment design has the wrong type
        visit_node(ret)
        return ret


class DefaultRunner(ProtocolRunner):
    """
    Run the default protocol at each data-tree node.  (Default protocols
    are given within :class:`ExperimentDesign` objects.)
    """

    def __init__(self, run_passes_separately=False):
        """
        Create a new DefaultRunner object, which runs the default protocol at
        each data-tree node.  (Default protocols are given within
        :class:`ExperimentDesign` objects.)

        Parameters
        ----------
        run_passes_separately : bool, optional
            If `True`, then when multi-pass data is encountered it is split into passes
            before handing it off to the protocols.  Set this to `True` when the default
            protocols being run expect single-pass data.

        Returns
        -------
        DefaultRunner
        """
        self.run_passes_separately = run_passes_separately

    def run(self, data, memlimit=None, comm=None):
        """
        Run all the protocols specified by this protocol-runner on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this
            protocol-runner in parallel.

        Returns
        -------
        ProtocolResultsDir
        """
        ret = ProtocolResultsDir(data)  # creates entire tree of nodes

        def visit_node(node, breadcrumb):
            for name, protocol in node.data.edesign.default_protocols.items():
                assert(name == protocol.name), "Protocol name inconsistency"
                print("Running protocol %s at %s" % (name, breadcrumb))
                if node.data.is_multipass() and self.run_passes_separately:
                    implicit_multipassprotocol = MultiPassProtocol(protocol)
                    node.for_protocol[implicit_multipassprotocol.name] = \
                        implicit_multipassprotocol.run(node.data, memlimit, comm)
                else:
                    node.for_protocol[name] = protocol.run(node.data, memlimit, comm)

            for subname, subnode in node.items():
                visit_node(subnode, breadcrumb + '/' + str(subname))

        visit_node(ret, '.')
        return ret


class ExperimentDesign(_TreeNode):
    """
    An experimental-design specification for one or more QCVV protocols.

    The quantities needed to define the experiments required to run a
    :class:`Protocol`.  Minimally,  a :class:`ExperimentDesign`
    object holds a list of :class:`Circuit`s that need to be run.  Typically,
    a :class:`ExperimentDesign` object also contains information used to
    interpret these circuits, either by describing how they are constructed from
    smaller pieces or how they are drawn from a distribution.

    It's important to note that a :class:`ExperimentDesign` does *not*
    contain all the inputs needed to run any particular QCVV protocol (e.g. there
    may be additional parameters specified when creating a :class:`Protocol` object,
    and it may be the case that the data described by a single :class:`ExperimentDesign`
    can be used by muliple protocols).  Rather, a :class:`ExperimentDesign`
    specifies what is necessary to acquire and interpret the *data* needed for
    one or more QCVV protocols.
    """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None, quick_load=False):
        """
        Initialize a new ExperimentDesign object from `dirname`.

        Parameters
        ----------
        dirname : str
            The *root* directory name (under which there is a 'edesign'
            subdirectory).

        parent : ExperimentDesign, optional
            The parent design object, if there is one.  Primarily used
            internally - if in doubt, leave this as `None`.

        name : str, optional
            The sub-name of the design object being loaded, i.e. the
            key of this data object beneath `parent`.  Only used when
            `parent` is not None.

        quick_load : bool, optional
            Setting this to True skips the loading of the potentially long
            circuit lists.  This can be useful when loading takes a long time
            and all the information of interest lies elsewhere, e.g. in an
            encompassing results object.

        Returns
        -------
        ExperimentDesign
        """
        dirname = _pathlib.Path(dirname)
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(dirname / 'edesign', 'auxfile_types', quick_load=quick_load))
        ret._init_children(dirname, 'edesign', quick_load=quick_load)
        ret._loaded_from = str(dirname.absolute())
        return ret

    def __init__(self, circuits=None, qubit_labels=None,
                 children=None, children_dirs=None, child_category=None):
        """
        Create a new ExperimentDesign object, which holds a set of circuits (needing data).

        Parameters
        ----------
        circuits : list of Circuits, optional
            A list of the circuits needing data.  If None, then the list is empty.

        qubit_labels : tuple or "multiple", optional
            The qubits that this experiment design applies to.  These should also
            be the line labels of `circuits`.  If None,  the concatenation
            of the qubit labels of any child experiment designs is used, or, if
            there are no child designs, the line labels of the first circuit is used.
            The special "multiple" value means that different circuits act on different
            qubit lines.

        children : dict, optional
            A dictionary of whose values are child
            :class:`ExperimentDesign` objects and whose keys are the
            names used to identify them in a "path".

        children_dirs : dict, optional
            A dictionary whose values are directory names and keys are child
            names (the same as the keys of `children`).  If None, then the
            keys of `children` must be strings and are used as directory
            names.  Directory names are used when saving the object (via
            :method:`write`).

        child_category : str, optional
            The category that describes the children of this object.  This
            is used as a heading for the keys of `children`.

        Returns
        -------
        ExperimentDesign
        """

        self.all_circuits_needing_data = circuits if (circuits is not None) else []
        self.alt_actual_circuits_executed = None  # None means == all_circuits_needing_data
        self.default_protocols = {}
        self.tags = {}
        self._nameddict_attributes = ()
        self._loaded_from = None

        #Instructions for saving/loading certain members - if a __dict__ member
        # *isn't* listed in this dict, then it's assumed to be json-able and included
        # in the main 'meta.json' file.  Allowed values are:
        # 'text-circuit-list' - a text circuit list file
        # 'json' - a json file
        # 'pickle' - a python pickle file (use only if really needed!)
        typ = 'pickle' if isinstance(self.all_circuits_needing_data, _objs.BulkCircuitList) else 'text-circuit-list'
        self.auxfile_types = {'all_circuits_needing_data': typ,
                              'alt_actual_circuits_executed': 'text-circuit-list',
                              'default_protocols': 'dict-of-protocolobjs'}

        # because TreeNode takes care of its own serialization:
        self.auxfile_types.update({'_dirs': 'none', '_vals': 'none', '_childcategory': 'none', '_loaded_from': 'none'})

        if qubit_labels is None:
            if children:
                if any([des.qubit_labels == "multiple" for des in children.values()]):
                    self.qubit_labels = "multiple"
                else:
                    self.qubit_labels = tuple(_itertools.chain(*[design.qubit_labels for design in children.values()]))

            elif len(circuits) > 0:
                self.qubit_labels = circuits[0].line_labels

            else:
                self.qubit_labels = ('*',)  # default "qubit labels"

        elif qubit_labels == "multiple":
            self.qubit_labels = "multiple"
        else:
            self.qubit_labels = tuple(qubit_labels)

        def auto_dirname(child_key):
            if isinstance(child_key, (list, tuple)):
                child_key = '_'.join(map(str, child_key))
            return child_key.replace(' ', '_')

        if children is None: children = {}
        children_dirs = children_dirs.copy() if (children_dirs is not None) else \
            {subname: auto_dirname(subname) for subname in children}

        assert(set(children.keys()) == set(children_dirs.keys()))
        super().__init__(children_dirs, children, child_category)

    def set_actual_circuits_executed(self, actual_circuits):
        """
        Sets a list of circuits equivalent to those in
        self.all_circuits_needing_data that will actually be
        executed.  For example, when the circuits in this design
        are run simultaneously with other circuits, the circuits
        in this design may need to be padded with idles.

        Parameters
        ----------
        actual_circuits : list
            A list of :class:`Circuit` objects that must be the same
            length as self.all_circuits_needing_data.

        Returns
        -------
        None
        """
        assert(len(actual_circuits) == len(self.all_circuits_needing_data))
        self.alt_actual_circuits_executed = actual_circuits

    def add_default_protocol(self, default_protocol_instance):
        """
        Add a "default" protocol to this experiment design.

        Default protocols are a way of designating protocols you mean to run
        on the the data corresponding to an experiment design *before* that
        data has been taken.  Use a :class:`DefaultRunner` object to run
        (all) the default protocols of the experiment designs within a
        :class:`ProtocolData` object.

        Note that default protocols are indexed by their names, and so
        when adding multiple default protocols they need to have distinct
        names (usually given to the protocol when it is constructed).

        Parameters
        ----------
        default_protocol_instance : Protocol
            The protocol to add.  This protocol's name is used to index it.

        Returns
        -------
        None
        """
        instance_name = default_protocol_instance.name
        self.default_protocols[instance_name] = default_protocol_instance

    def write(self, dirname=None, parent=None):
        """
        Write this experiment design to a directory.

        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have
            an 'edesign' subdirectory, which will be created if needed and
            overwritten if present.  If None, then the path this object
            was loaded from is used (if this object wasn't loaded from disk,
            an error is raised).

        parent : ExperimentDesign, optional
            The parent experiment design, when a parent is writing this
            design as a sub-experiment-design.  Otherwise leave as None.

        Returns
        -------
        None
        """
        if dirname is None:
            dirname = self._loaded_from
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")

        _io.write_obj_to_meta_based_dir(self, _pathlib.Path(dirname) / 'edesign', 'auxfile_types')
        self.write_children(dirname)
        self._loaded_from = str(_pathlib.Path(dirname).absolute())  # for future writes

    def setup_nameddict(self, final_dict):
        """
        Initializes a set of nested :class:`NamedDict` dictionaries describing this design.

        This function is used by :class:`ProtocolResults` objects when they're creating
        nested dictionaries of their contents.  This function returns a set of nested,
        single (key,val)-pair named-dictionaries which describe the particular attributes
        of this :class:`ExperimentDesign` object named within its `self._nameddict_attributes`
        tuple.  The final nested dictionary is set to be `final_dict`, which allows additional
        result quantities to easily be added.

        Parameters
        ----------
        final_dict : NamedDict
            the final-level (innermost-nested) NamedDict in the returned nested dictionary.

        Returns
        -------
        NamedDict
        """
        keys_vals_types = _convert_nameddict_attributes(self)
        keys_vals_types.extend([(k, v, 'category') for k, v in self.tags.items()])
        return _NamedDict.create_nested(keys_vals_types, final_dict)

    def create_subdata(self, subdata_name, dataset):
        """
        Creates a :class:`ProtocolData` object for the sub-experiment-design
        given by `subdata_name` starting from `dataset` as the data for *this*
        experiment design.  This is used internally by :class:`ProtocolData`
        objects, and shouldn't need to be used by external users.
        """
        raise NotImplementedError("This protocol edesign cannot create any subdata!")


class CircuitListsDesign(ExperimentDesign):
    """
    Experiment deisgn specification that is comprised of multiple circuit lists.
    """

    def __init__(self, circuit_lists, all_circuits_needing_data=None, qubit_labels=None,
                 nested=False, remove_duplicates=True):
        """
        Create a new CircuitListsDesign object.

        Parameters
        ----------
        circuit_lists : list
            A list whose elements are themselves lists of :class:`Circuit`
            objects, specifying the data that needs to be taken.

        all_circuits_needing_data : list, optional
            A list of all the circuits needing data.  By default, This is just
            the concatenation of the elements of `circuit_lists` with duplicates
            removed.  The only reason to specify this separately is if you
            happen to have this list lying around.

        qubit_labels : tuple, optional
            The qubits that this experiment design applies to. If None, the
            line labels of the first circuit is used.

        nested : bool, optional
            Whether the elements of `circuit_lists` are nested, e.g. whether
            `circuit_lists[i]` is a subset of `circuit_lists[i+1]`.  This
            is useful to know because certain operations can be more efficient
            when it is known that the lists are nested.

        remove_duplicates : bool, optional
            Whether to remove duplicates when automatically creating
            all the circuits that need data (this argument isn't used
            when `all_circuits_needing_data` is given).

        Returns
        -------
        CircuitListsDesign
        """

        if isinstance(circuit_lists, _objs.BulkCircuitList):
            master = circuit_lists
            assert(master.circuit_structure is not None), \
                "When specifying a set of lists using a single BulkCircuitList it must contain a circuit structure."
            master_struct = master.circuit_structure
            circuit_lists = [_objs.BulkCircuitList(master_struct.truncate(max_lengths=master.Ls[0:i + 1]),
                                                   master.op_label_aliases, master.circuit_weights)
                             for i in range(len(master.Ls))]
            nested = True  # (by this construction)

        if all_circuits_needing_data is not None:
            all_circuits = all_circuits_needing_data  # (ok if this is a BulkCircuitList)
        elif nested and len(circuit_lists) > 0:
            all_circuits = circuit_lists[-1]  # (ok if this is a BulkCircuitList)
        else:
            all_circuits = []
            for lst in circuit_lists:
                all_circuits.extend(lst)  # Note: this should work even for type(lst) == BulkCircuitList
            if remove_duplicates:
                _lt.remove_duplicates_in_place(all_circuits)

        self.circuit_lists = circuit_lists
        self.nested = nested

        super().__init__(all_circuits, qubit_labels)
        self.auxfile_types['circuit_lists'] = 'pickle' \
            if any([isinstance(lst, _objs.BulkCircuitList) for lst in circuit_lists]) else 'text-circuit-lists'


class CombinedExperimentDesign(ExperimentDesign):  # for multiple designs on the same dataset
    """
    An experiment design that combines the specifications of
    one or more "sub-designs".  The sub-designs are preserved as children under
    the :class:`CombinedExperimentDesign` instance, creating a "data-tree" structure.  The
    :class:`CombinedExperimentDesign` object itself simply merges all of the circuit lists.
    """

    def __init__(self, sub_designs, all_circuits=None, qubit_labels=None, sub_design_dirs=None,
                 interleave=False, category='EdesignBranch'):
        """
        Create a new CombinedExperimentDesign object.

        Parameters
        ----------
        sub_designs : dict or list
            A dictionary of other :class:`ExperimentDesign` objects whose keys
            are names for each sub-edesign (used for directories and to index
            the sub-edesigns from this experiment design).  If a list is given instead,
            a default names of the form "**<number>" are used.

        all_circuits : list, optional
            A list of :class:`Circuit`s, specifying all the circuits needing
            data.  This can include additional circuits that are not in any
            of `sub_designs`.  By default, the union of all the circuits in
            the sub-designs is used.

        qubit_labels : tuple, optional
            The qubits that this experiment design applies to. If None, the line labels
            of the first circuit is used.

        sub_design_dirs : dict, optional
            A dictionary whose values are directory names and keys are sub-edesign
            names (the same as the keys of `sub_designs`).  If None, then the
            keys of `sub_designs` must be strings and are used as directory
            names.  Directory names are used when saving the object (via
            :method:`write`).

        category : str, optional
            The category that describes the sub-edesigns of this object.  This
            is used as a heading for the keys of `sub_designs`.

        Returns
        -------
        CombinedExperimentDesign
        """

        if not isinstance(sub_designs, dict):
            sub_designs = {("**%d" % i): des for i, des in enumerate(sub_designs)}

        if all_circuits is None:
            all_circuits = []
            if not interleave:
                for des in sub_designs.values():
                    all_circuits.extend(des.all_circuits_needing_data)
            else:
                raise NotImplementedError("Interleaving not implemented yet")
            _lt.remove_duplicates_in_place(all_circuits)  # Maybe don't always do this?

        if qubit_labels is None and len(sub_designs) > 0:
            first = sub_designs[list(sub_designs.keys())[0]].qubit_labels
            if any([des.qubit_labels != first for des in sub_designs.values()]):
                qubit_labels = "multiple"
            else:
                qubit_labels = first

        super().__init__(all_circuits, qubit_labels, sub_designs, sub_design_dirs, category)

    def create_subdata(self, sub_name, dataset):
        """
        Creates a :class:`ProtocolData` object for the sub-experiment-design
        given by `subdata_name` starting from `dataset` as the data for *this*
        experiment design.  This is used internally by :class:`ProtocolData`
        objects, and shouldn't need to be used by external users.
        """
        sub_circuits = self[sub_name].all_circuits_needing_data
        if isinstance(dataset, dict):  # then do truncation "element-wise"
            truncated_ds = {k: ds.truncate(sub_circuits) for k, ds in dataset.items()}
            for tds in truncated_ds.values(): tds.add_std_nqubit_outcome_labels(len(self[sub_name].qubit_labels))
        else:
            truncated_ds = dataset.truncate(sub_circuits)  # maybe have filter_dataset also do this?
            #truncated_ds.add_outcome_labels(dataset.get_outcome_labels())  # make sure truncated ds has all outcomes
            truncated_ds.add_std_nqubit_outcome_labels(len(self[sub_name].qubit_labels))
        return ProtocolData(self[sub_name], truncated_ds)


class SimultaneousExperimentDesign(ExperimentDesign):
    """
    An experiment design whose circuits are the tensor-products
    of the circuits from one or more  :class:`ExperimentDesign` objects that
    act on disjoint sets of qubits.  The sub-designs are preserved as children under
    the :class:`SimultaneousExperimentDesign` instance, creating a "data-tree" structure.
    """

    #@classmethod
    #def from_tensored_circuits(cls, circuits, template_edesign, qubit_labels_per_edesign):
    #    pass #Useful??? - need to break each circuit into different parts
    # based on qubits, then copy (?) template edesign and just replace itself
    # all_circuits_needing_data member?

    def __init__(self, edesigns, tensored_circuits=None, qubit_labels=None, category='Qubits'):
        """
        Create a new SimultaneousExperimentDesign object.

        Parameters
        ----------
        edesigns : list
            A list of :class:`ExperimentDesign` objects  whose circuits
            are to occur simultaneously.

        tensored_circuits : list, optional
            A list of all the circuits for this experiment design.  By default,
            these are the circuits of those in `edesigns` tensored together.
            Typically this is left as the default.

        qubit_labels : tuple, optional
            The qubits that this experiment design applies to. If None, the
            concatenated qubit labels of `edesigns` are used (this is usually
            what you want).

        category : str, optional
            The category name for the qubit-label-tuples correspoding to the
            elements of `edesigns`.

        Returns
        -------
        SimultaneousExperimentDesign
        """
        #TODO: check that sub-designs don't have overlapping qubit_labels
        assert(not any([des.qubit_labels == "multiple" for des in edesigns])), \
            "SimultaneousExperimentDesign requires sub-designs with definite qubit_labels, not 'multiple'"

        if qubit_labels is None:
            qubit_labels = tuple(_itertools.chain(*[des.qubit_labels for des in edesigns]))

        if tensored_circuits is None:
            #Build tensor product of circuits
            tensored_circuits = []
            circuits_per_edesign = [des.all_circuits_needing_data[:] for des in edesigns]

            #Pad shorter lists with None values
            maxLen = max(map(len, circuits_per_edesign))
            for lst in circuits_per_edesign:
                if len(lst) < maxLen: lst.extend([None] * (maxLen - len(lst)))

            def pad(subcs):
                maxLen = max([len(c) if (c is not None) else 0 for c in subcs])
                padded = []
                for c in subcs:
                    if c is not None and len(c) < maxLen:
                        cpy = c.copy(editable=True)
                        cpy.insert_idling_layers(None, maxLen - len(cpy))
                        cpy.done_editing()
                        padded.append(cpy)
                    else:
                        padded.append(c)
                assert(all([len(c) == maxLen for c in padded if c is not None]))
                return padded

            padded_circuit_lists = [list() for des in edesigns]
            for subcircuits in zip(*circuits_per_edesign):
                c = _cir.Circuit(num_lines=0, editable=True)  # Creates a empty circuit over no wires
                padded_subcircuits = pad(subcircuits)
                for subc in padded_subcircuits:
                    if subc is not None:
                        c.tensor_circuit(subc)
                c.line_labels = qubit_labels
                c.done_editing()
                tensored_circuits.append(c)
                for lst, subc in zip(padded_circuit_lists, padded_subcircuits):
                    if subc is not None: lst.append(subc)

            for des, padded_circuits in zip(edesigns, padded_circuit_lists):
                des.set_actual_circuits_executed(padded_circuits)

        sub_designs = {des.qubit_labels: des for des in edesigns}
        sub_design_dirs = {qlbls: '_'.join(map(str, qlbls)) for qlbls in sub_designs}
        super().__init__(tensored_circuits, qubit_labels, sub_designs, sub_design_dirs, category)

    def create_subdata(self, qubit_labels, dataset):
        """
        Creates a :class:`ProtocolData` object for the sub-experiment-design
        given by `subdata_name` starting from `dataset` as the data for *this*
        experiment design.  This is used internally by :class:`ProtocolData`
        objects, and shouldn't need to be used by external users.
        """
        if isinstance(dataset, _objs.MultiDataSet):
            raise NotImplementedError("SimultaneousExperimentDesigns don't work with multi-pass data yet.")

        all_circuits = self.all_circuits_needing_data
        qubit_ordering = all_circuits[0].line_labels  # first circuit in *this* edesign determines qubit order
        qubit_index = {qlabel: i for i, qlabel in enumerate(qubit_ordering)}
        sub_design = self[qubit_labels]
        qubit_indices = [qubit_index[ql] for ql in qubit_labels]  # order determined by first circuit (see above)

        if isinstance(dataset, dict):  # then do filtration "element-wise"
            filtered_ds = {k: _cnst.filter_dataset(ds, qubit_labels, qubit_indices) for k, ds in dataset.items()}
            for fds in filtered_ds.values(): fds.add_std_nqubit_outcome_labels(len(qubit_labels))
        else:
            filtered_ds = _cnst.filter_dataset(dataset, qubit_labels, qubit_indices)  # Marginalize dataset
            filtered_ds.add_std_nqubit_outcome_labels(len(qubit_labels))  # ensure filtered_ds has appropriate outcomes

        if sub_design.alt_actual_circuits_executed:
            actual_to_desired = _collections.defaultdict(lambda: None)
            actual_to_desired.update({actual: desired for actual, desired in
                                      zip(sub_design.alt_actual_circuits_executed,
                                          sub_design.all_circuits_needing_data)})
            if isinstance(dataset, dict):  # then do circuit processing "element-wise"
                for k in filtered_ds:
                    fds = filtered_ds[k].copy_nonstatic()
                    fds.process_circuits(lambda c: actual_to_desired[c], aggregate=False)
                    fds.done_adding_data()
                    filtered_ds[k] = fds
            else:
                filtered_ds = filtered_ds.copy_nonstatic()
                filtered_ds.process_circuits(lambda c: actual_to_desired[c], aggregate=False)
                filtered_ds.done_adding_data()
        return ProtocolData(sub_design, filtered_ds)


class ProtocolData(_TreeNode):
    """
    A :class:`ProtocolData` object represents the experimental data needed to
    run one or more QCVV protocols.  This class contains a :class:`ProtocolIput`,
    which describes a set of circuits, and a :class:`DataSet` (or :class:`MultiDataSet`)
    that holds data for these circuits.  These members correspond to the `.edesign`
    and `.dataset` attributes.
    """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None, quick_load=False):
        """
        Initialize a new ProtocolData object from `dirname`.

        Parameters
        ----------
        dirname : str
            The *root* directory name (under which there are 'edesign'
            and 'data' subdirectories).

        parent : ProtocolData, optional
            The parent data object, if there is one.  This is needed for
            sub-data objects which reference/inherit their parent's dataset.
            Primarily used internally - if in doubt, leave this as `None`.

        name : str, optional
            The sub-name of the design object being loaded, i.e. the
            key of this data object beneath `parent`.  Only used when
            `parent` is not None.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time, e.g. the actual raw data set(s). This can be useful
            when loading takes a long time and all the information of interest
            lies elsewhere, e.g. in an encompassing results object.

        Returns
        -------
        ProtocolData
        """
        p = _pathlib.Path(dirname)
        edesign = parent.edesign[name] if parent and name else \
            _io.load_edesign_from_dir(dirname, quick_load=quick_load)

        data_dir = p / 'data'
        #with open(data_dir / 'meta.json', 'r') as f:
        #    meta = _json.load(f)

        if quick_load:
            dataset = None  # don't load any dataset - just the cache (usually b/c loading is slow)
            # Note: could also use (path.stat().st_size >= max_size) to condition on size of data files
        else:
            #Load dataset or multidataset based on what files exist
            dataset_files = sorted(list(data_dir.glob('*.txt')))
            if len(dataset_files) == 0:  # assume same dataset as parent
                if parent is None: parent = ProtocolData.from_dir(dirname / '..')
                dataset = parent.dataset
            elif len(dataset_files) == 1 and dataset_files[0].name == 'dataset.txt':  # a single dataset.txt file
                dataset = _io.load_dataset(dataset_files[0], with_times=False, ignore_zero_count_lines=False,
                                           verbosity=0)
            else:
                dataset = {pth.stem: _io.load_dataset(pth, with_times=False, ignore_zero_count_lines=False, verbosity=0)
                           for pth in dataset_files}
                #FUTURE: use MultiDataSet, BUT in addition to init_from_dict we'll need to add truncate, filter, and
                # process_circuits support for MultiDataSet objects -- for now (above) we just use dicts of DataSets.
                #raise NotImplementedError("Need to implement MultiDataSet.init_from_dict!")
                #dataset = _objs.MultiDataSet.init_from_dict(
                #    {pth.name: _io.load_dataset(pth, verbosity=0) for pth in dataset_files})

        cache = _io.read_json_or_pkl_files_to_dict(data_dir / 'cache')

        ret = cls(edesign, dataset, cache)
        ret._init_children(dirname, 'data', quick_load=quick_load)  # loads child nodes
        return ret

    def __init__(self, edesign, dataset=None, cache=None):
        """
        Create a new ProtocolData object.

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design describing what circuits this object
            contains data for.  If None, then an unstructured
            :class:`ExperimentDesign` is created containing the circuits
            present in `dataset`.

        dataset : DataSet or MultiDataSet, optional
            The data counts themselves.

        cache : dict, optional
            A cache of values which holds values derived *only* from
            the experiment design and data in this object.

        Returns
        -------
        ProtocolData
        """
        self.edesign = edesign
        self.dataset = dataset  # MultiDataSet allowed for multi-pass data; None also allowed.
        self.cache = cache if (cache is not None) else {}
        self.tags = {}

        if isinstance(self.dataset, (_objs.MultiDataSet, dict)):  # can be a dict of DataSets instead of a multidataset
            for dsname in self.dataset:
                if dsname not in self.cache: self.cache[dsname] = {}  # create separate caches for each pass
            self._passdatas = {dsname: ProtocolData(self.edesign, ds, self.cache[dsname])
                               for dsname, ds in self.dataset.items()}
            ds_to_get_circuits_from = self.dataset[list(self.dataset.keys())[0]]
        else:
            self._passdatas = {None: self}
            ds_to_get_circuits_from = dataset

        if self.edesign is None:
            self.edesign = ExperimentDesign(list(ds_to_get_circuits_from.keys()))
        super().__init__(self.edesign._dirs, {}, self.edesign._childcategory)  # children created on-demand

    def __getstate__(self):
        # don't pickle ourself recursively if self._passdatas contains just ourself
        to_pickle = self.__dict__.copy()
        if list(to_pickle['_passdatas'].keys()) == [None]:
            to_pickle['_passdatas'] = None
        return to_pickle

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        if self._passdatas is None:
            self._passdatas = {None: self}

    def _create_childval(self, key):  # (this is how children are created on-demand)
        """ Create the value for `key` on demand. """
        return self.edesign.create_subdata(key, self.dataset)

    @property
    def passes(self):
        """A dictionary of the per-pass sub-results."""
        return self._passdatas

    def is_multipass(self):
        """
        Whether this protocol data contains multiple passes (more
        accurately, whether the `.dataset` of this object is a
        :class:`MultiDataSet`).

        Returns
        -------
        bool
        """
        return isinstance(self.dataset, (_objs.MultiDataSet, dict))

    #def get_tree_paths(self):
    #    return self.edesign.get_tree_paths()

    def filter_paths(self, paths, paths_are_sorted=False):
        """
        Returns a new :class:`ProtocolData` object with a subset of the
        data-tree paths contained under this one.

        Parameters
        ----------
        paths : list
            A list of the paths to keep.  Each path is a tuple of keys,
            delineating a path in the data-tree.

        paths_are_sorted : bool, optional
            Whether `paths` has already been sorted lexographically.

        Returns
        -------
        ProtocolData
        """
        def build_data(des, src_data):
            """ Uses a template (filtered) edesign to selectively
                copy the non-edesign parts of a 'src_data' ProtocolData """
            ret = ProtocolData(des, src_data.dataset, src_data.cache)
            for subname, subedesign in des.items():
                if subname in src_data._vals:  # if we've actually created this sub-data...
                    ret._vals[subname] = build_data(subedesign, src_data._vals[subname])
            return ret
        filtered_edesign = self.edesign.filter_paths(paths, paths_are_sorted)
        return build_data(filtered_edesign, self)

    def write(self, dirname=None, parent=None):
        """
        Write this protocol data to a directory.

        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have
            'edesign' and 'data' subdirectories, which will be created if
            needed and overwritten if present.  If None, then the path this object
            was loaded from is used (if this object wasn't loaded from disk,
            an error is raised).

        parent : ProtocolData, optional
            The parent protocol data, when a parent is writing this
            data as a sub-protocol-data object.  Otherwise leave as None.

        Returns
        -------
        None
        """
        if dirname is None:
            dirname = self.edesign._loaded_from
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")
        dirname = _pathlib.Path(dirname)
        data_dir = dirname / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        _io.obj_to_meta_json(self, data_dir)

        if parent is None:
            self.edesign.write(dirname)  # otherwise assume parent has already written edesign

        if self.dataset is not None:  # otherwise don't write any dataset
            if parent and (self.dataset is parent.dataset):  # then no need to write any data
                assert(len(list(data_dir.glob('*.txt'))) == 0), "There shouldn't be *.txt files in %s!" % str(data_dir)
            else:
                data_dir.mkdir(exist_ok=True)
                if isinstance(self.dataset, (_objs.MultiDataSet, dict)):
                    for dsname, ds in self.dataset.items():
                        _io.write_dataset(data_dir / (dsname + '.txt'), ds)
                else:
                    _io.write_dataset(data_dir / 'dataset.txt', self.dataset)

        if self.cache:
            _io.write_dict_to_json_or_pkl_files(self.cache, data_dir / 'cache')

        self.write_children(dirname, write_subdir_json=False)  # writes sub-datas

    def setup_nameddict(self, final_dict):
        """
        Initializes a set of nested :class:`NamedDict` dictionaries describing this data.

        This function is used by :class:`ProtocolResults` objects when they're creating
        nested dictionaries of their contents.  The final nested dictionary is set to be
        `final_dict`, which allows additional result quantities to easily be added.

        Parameters
        ----------
        final_dict : NamedDict
            the final-level (innermost-nested) NamedDict in the returned nested dictionary.

        Returns
        -------
        NamedDict
        """
        keys_vals_types = [(k, v, 'category') for k, v in self.tags.items()]
        return self.edesign.setup_nameddict(_NamedDict.create_nested(keys_vals_types, final_dict))


class ProtocolResults(object):
    """
    A :class:`ProtocolResults` object contains a :class:`ProtocolData` object
    and stores the results from running a QCVV protocol (a :class:`Protcocol`)
    on this data.
    """

    @classmethod
    def from_dir(cls, dirname, name, preloaded_data=None, quick_load=False):
        """
        Initialize a new ProtocolResults object from `dirname` / results / `name`.

        Parameters
        ----------
        dirname : str
            The *root* directory name (under which there is are 'edesign',
            'data', and 'results' subdirectories).

        name : str
            The sub-directory name of the particular results object to load
            (there can be multiple under a given root `dirname`).  This is the
            name of a subdirectory of `dirname` / results.

        preloaded_data : ProtocolData, optional
            In the case that the :class:`ProtocolData` object for `dirname`
            is already loaded, it can be passed in here.  Otherwise leave this
            as None and it will be loaded.

        quick_load : bool, optional
            Setting this to True skips the loading of data and experiment-design
            components that may take a long time to load. This can be useful
            all the information of interest lies only within the results object.

        Returns
        -------
        ProtocolResults
        """
        dirname = _pathlib.Path(dirname)
        ret = cls._from_dir_partial(dirname / 'results' / name, quick_load, load_protocol=True)
        ret.data = preloaded_data if (preloaded_data is not None) else \
            _io.load_data_from_dir(dirname, quick_load=quick_load)
        assert(ret.name == name), "ProtocolResults name inconsistency!"
        return ret

    @classmethod
    def _from_dir_partial(cls, dirname, quick_load=False, load_protocol=False):
        """
        Internal method for loading only the results-specific data, and not the `data` member.
        This method may be used independently by derived ProtocolResults objecsts which contain
        multiple sub-results (e.g. MultiPassResults)
        """
        ignore = ('type',) if load_protocol else ('type', 'protocol')
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(dirname, 'auxfile_types', ignore, quick_load=quick_load))
        return ret

    def __init__(self, data, protocol_instance):
        """
        Create a new ProtocolResults object.

        Parameters
        ----------
        data : ProtocolData
            The input data from which these results are derived.

        protocol_instance : Protocol
            The protocol that created these results.

        Returns
        -------
        ProtocolResults
        """
        self.name = protocol_instance.name  # just for convenience in JSON dir
        self.protocol = protocol_instance
        self.data = data
        self.auxfile_types = {'data': 'none', 'protocol': 'protocolobj'}

    def write(self, dirname=None, data_already_written=False):
        """
        Write these protocol results to a directory.

        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have
            'edesign', 'data', and 'results/<myname>' subdirectories, which will
            path be created if needed and overwritten if present.  If None, then
            the this object was loaded from is used (if this object wasn't
            loaded from disk, an error is raised).

        data_already_written : bool, optional
            Set this to True if you're sure the `.data` :class:`ProtocolData` object
            within this results object has already been written to `dirname`.  Leaving
            this as the default is a safe option.

        Returns
        -------
        None
        """
        if dirname is None:
            dirname = self.data.edesign._loaded_from
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")

        p = _pathlib.Path(dirname)
        results_dir = p / 'results' / self.name
        results_dir.mkdir(parents=True, exist_ok=True)

        #write edesign and data
        if not data_already_written:
            self.data.write(dirname)

        #write qtys to results dir
        self._write_partial(results_dir, write_protocol=True)

    def _write_partial(self, results_dir, write_protocol=False):
        """
        Internal method used to write the results-specific data to a directory.
        This method does not write the object's `data` member, which must be
        serialized separately.
        """
        _io.write_obj_to_meta_based_dir(self, results_dir, 'auxfile_types',
                                        omit_attributes=() if write_protocol else ('protocol',))

    def as_nameddict(self):
        """
        Convert these results into nested :class:`NamedDict` objects.

        Returns
        -------
        NamedDict
        """
        return self.protocol.setup_nameddict(
            self.data.setup_nameddict(
                self._my_attributes_as_nameddict()
            ))

    def _my_attributes_as_nameddict(self):
        #This function can be overridden by derived classes - this just
        # tries to give a decent default implementation.  Ideally derived
        # implementatons would use ValueName and Value columns so results
        # can be aggregated easily.
        vals = _NamedDict('ValueName', 'category')
        ignore_members = ('name', 'protocol', 'data', 'auxfile_types')
        for k, v in self.__dict__.items():
            if k.startswith('_') or k in ignore_members: continue
            if isinstance(v, ProtocolResults):
                vals[k] = v.as_nameddict()
            elif isinstance(v, _NamedDict):
                vals[k] = v
            elif isinstance(v, dict):
                pass  # don't know how to make a dict into a (nested) NamedDict
            else:  # non-dicts are ok to just store
                vals[k] = v
        return vals

    def as_dataframe(self):
        """
        Convert these results into Pandas dataframe.

        Returns
        -------
        DataFrame
        """
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

    @classmethod
    def from_dir(cls, dirname, name, preloaded_data=None, quick_load=False):
        #Because 'dict-of-resultsobjs' only does *partial* loading/writing of the given results
        # objects, we need to finish the loading manually.  Only partial loading is performed so
        # because it is assumed that whatever object has a 'dict-of-resultsobjs' and isn't a
        # ProtocolResultsDir must separately have access to the protocol and data used by these
        # results (as they should be derivative of the protocol and data of the object).
        ret = super(MultiPassResults, cls).from_dir(dirname, name, preloaded_data, quick_load)  # call base class
        for pass_name, partially_loaded_results in ret.passes.items():
            partially_loaded_results.data = ret.data.passes[pass_name]  # assumes data and ret.passes use *same* keys
            partially_loaded_results.protocol = ret.protocol.protocol  # assumes ret.protocol is MultiPassProtocol-like
        return ret

    def __init__(self, data, protocol_instance):
        """
        Initialize an empty MultiPassResults object, which contain a dictionary
        of sub-results one per "pass".  Usually these sub-results are obtained
        by running `protocol_instance` on each data set within `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data from which these results are derived.

        protocol_instance : Protocol
            The protocol that created these results.

        Returns
        -------
        MultiPassResults
        """
        super().__init__(data, protocol_instance)

        self.passes = {}  # _NamedDict('Pass', 'category') - as_nameddict takes care of this
        self.auxfile_types['passes'] = 'dict-of-resultsobjs'

    def as_nameddict(self):
        # essentially inject a 'Pass' dict right beneath the outer-most Protocol Name dict
        ret = _NamedDict('Protocol Name', 'category')
        for pass_name, r in self.passes.items():
            sub = r.as_nameddict()  # should have outer-most 'Protocol Name' dict
            assert(sub.name == 'Protocol Name' and len(sub) == 1)
            pname = r.protocol.name  # also list(sub.keys())[0]
            if pname not in ret:
                ret[pname] = _NamedDict('Pass', 'category')
            ret[pname][pass_name] = sub[pname]

        return ret


class ProtocolResultsDir(_TreeNode):
    """
    A :class:`ProtocolResultsDir` holds a dictionary of :class:`ProtocolResults`
    objects.  It contains a :class:`ProtocolData` object and is rooted at the_model
    corresponding node of the data-tree.  It contains links to child-:class:`ProtocolResultsDir`
    objects representing sub-directories.
    """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None, quick_load=False):
        """
        Initialize a new ProtocolResultsDir object from `dirname`.

        Parameters
        ----------
        dirname : str
            The *root* directory name (under which there are 'edesign'
            and 'data', and possibly 'results', subdirectories).

        parent : ProtocolResultsDir, optional
            The parent results-directory object that is loading the
            returned object as a sub-results.  This is used internally
            when loading a :class:`ProtocolResultsDir` that represents
            a node of the data-tree with children.

        name : str, optional
            The name of this result within `parent`.  This is only
            used when `parent` is not None.

        quick_load : bool, optional
            Setting this to True skips the loading of data and experiment-design
            components that may take a long time to load. This can be useful
            all the information of interest lies only within the contained results objects.

        Returns
        -------
        ProtcolResultsDir
        """
        dirname = _pathlib.Path(dirname)
        data = parent.data[name] if (parent and name) else \
            _io.load_data_from_dir(dirname, quick_load=quick_load)

        #Load results in results_dir
        results = {}
        results_dir = dirname / 'results'
        if results_dir.is_dir():  # if results_dir doesn't exist that's ok (just no results to load)
            for pth in results_dir.iterdir():
                if pth.is_dir() and (pth / 'meta.json').is_file():
                    results[pth.name] = _io.cls_from_meta_json(pth).from_dir(
                        dirname, pth.name, preloaded_data=data, quick_load=quick_load)

        ret = cls(data, results, {})  # don't initialize children now
        ret._init_children(dirname, meta_subdir='results', quick_load=quick_load)
        return ret

    def __init__(self, data, protocol_results=None, children=None):
        """
        Create a new ProtocolResultsDir object.

        This container object holds two things:
        1. A `.for_protocol` dictionary of :class:`ProtocolResults` corresponding
           to different protocols (keys are protocol names).

        2. Child :class:`ProtocolResultsDir` objects, obtained by indexing this
           object directly using the name of the sub-directory.

        Parameters
        ----------
        data : ProtocolData
            The data from which *all* the Results objects in this
            ProtocolResultsDir are derived.

        protocol_results : ProtocolResults, optional
            An initial (single) results object to add.  The name of the
            results object is used as its key within the `.for_protocol`
            dictionary.  If None, then an empty results directory is created.

        children : dict, optional
            A dictionary of the :class:`ProtocolResultsDir` objects that are
            sub-directories beneath this one.  If None, then children are
            automatically created based upon the tree given by `data`.  (To
            avoid creating any children, you can pass an empty dict here.)

        Returns
        -------
        ProtocolResultsDir
        """
        self.data = data  # edesign and data
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

        super().__init__(self.data.edesign._dirs, children, self.data.edesign._childcategory)

    def write(self, dirname=None, parent=None):
        """
        Write this "protocol results directory" to a directory.

        Parameters
        ----------
        dirname : str
            The *root* directory to write into.  This directory will have
            'edesign', 'data', and 'results' subdirectories, which will be
            created if needed and overwritten if present.    If None, then
            the path this object was loaded from is used (if this object
            wasn't loaded from disk, an error is raised).

        parent : ProtocolResultsDir, optional
            The parent protocol results directory, when a parent is writing this
            results dir as a sub-results-dir.  Otherwise leave as None.

        Returns
        -------
        None
        """
        if dirname is None:
            dirname = self.data.edesign._loaded_from
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")

        if parent is None: self.data.write(dirname)  # assume parent has already written data
        dirname = _pathlib.Path(dirname)

        results_dir = dirname / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        _io.obj_to_meta_json(self, results_dir)

        #write the results
        for name, results in self.for_protocol.items():
            assert(results.name == name)
            results.write(dirname, data_already_written=True)

        self.write_children(dirname, write_subdir_json=False)  # writes sub-nodes

    def as_nameddict(self):
        """
        Convert the results in this object into nested :class:`NamedDict` objects.

        Returns
        -------
        NamedDict
        """
        sub_results = {k: v.as_nameddict() for k, v in self.items()}
        nds = [v.as_nameddict() for v in self.for_protocol.values()]
        if len(nds) > 0:
            assert(all([nd.keyname == nds[0].keyname for nd in nds])), \
                "All protocols on a given node must return a NamedDict with the *same* root name!"
            results_on_this_node = nds[0]
            for nd in nds[1:]:
                results_on_this_node.update(nd)
        else:
            results_on_this_node = None

        if sub_results:
            category = self.child_category if self.child_category else 'nocategory'
            ret = _NamedDict(category, 'category')
            if results_on_this_node:
                #Results in this (self's) dir don't have a value for the sub-category, so put None
                ret[None] = results_on_this_node
            ret.update(sub_results)
            return ret
        else:  # no sub-results, so can just return a dict of results on this node
            return results_on_this_node

    def as_dataframe(self):
        """
        Convert these results into Pandas dataframe.

        Returns
        -------
        DataFrame
        """
        return self.as_nameddict().as_dataframe()

    def __str__(self):
        import pprint
        P = pprint.PrettyPrinter()
        return P.pformat(self.as_nameddict())


def run_default_protocols(data, memlimit=None, comm=None):
    """
    Run the default protocols for the data-tree rooted at `data`.

    Parameters
    ----------
    data : ProtocolData
        the data to run default protocols on.

    memlimit : int, optional
        A rough per-processor memory limit in bytes.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator used to run the protocols
        in parallel.

    Returns
    -------
    ProtocolResultsDir
    """
    return DefaultRunner().run(data, memlimit, comm)


class ProtocolPostProcessor(object):
    """
    A :class:`ProtocolPostProcessor` is similar to a protocol, but runs on an
    *existing* results object, and produces a new (updated?) Results object.
    """

    #Note: this is essentially a duplicate of the Protocol class (except run takes a results object)
    # but it's conceptually a different thing...  Should we derive it from Protocol?

    @classmethod
    def from_dir(cls, dirname, quick_load=False):  # same I/O pattern as Protocol
        """
        Initialize a new ProtocolPostProcessor object from `dirname`.

        Parameters
        ----------
        dirname : str
            The directory name.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Returns
        -------
        ProtocolPostProcessor
        """
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types', quick_load=quick_load))
        ret._init_unserialized_attributes()
        return ret

    def __init__(self, name):
        """
        Create a new ProtocolPostProcessor object.

        Parameters
        ----------
        name : str
            The name of this post-processor.

        Returns
        -------
        ProtocolPostProcessor
        """
        super().__init__()
        self.name = name if name else self.__class__.__name__
        self.auxfile_types = {}

    def _init_unserialized_attributes(self):
        pass

    def run(self, results, memlimit=None, comm=None):
        """
        Run this post-processor on `results`.

        Parameters
        ----------
        results : ProtocolResults
            The input results.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this
            post-processor in parallel.

        Returns
        -------
        ProtocolResults
        """
        #Maybe these could also take data objects and run protocols on them automatically?
        #Returned Results object should be rooted at place of given results/resultsdir
        raise NotImplementedError("Derived classes should implement this!")

    def write(self, dirname):
        """
        Write this protocol post-processor to a directory.

        Parameters
        ----------
        dirname : str
            The directory to write into.

        Returns
        -------
        None
        """
        _io.write_obj_to_meta_based_dir(self, dirname, 'auxfile_types')


#In the future, we could put this function into a base class for
# the classes that utilize it above, so it would become a proper method.
def _convert_nameddict_attributes(obj):
    """
    A helper function that converts the elements of the 
    "_nameddict_attributes" attribute of several classes to
    the (key, value, type) array expected by 
    :method:`NamedDict.create_nested`.
    """
    keys_vals_types = []
    for tup in obj._nameddict_attributes:
        if len(tup) == 1: attr, key, typ = tup[0], tup[0], None
        elif len(tup) == 2: attr, key, typ = tup[0], tup[1], None
        elif len(tup) == 3: attr, key, typ = tup
        keys_vals_types.append((key, getattr(obj, attr), typ))
    return keys_vals_types


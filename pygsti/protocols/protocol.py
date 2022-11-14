"""
Protocol object
"""
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
import collections as _collections
import copy as _copy
import numpy as _np
import itertools as _itertools
import pathlib as _pathlib

from pygsti.protocols.treenode import TreeNode as _TreeNode
from pygsti import io as _io
from pygsti import circuits as _circuits
from pygsti import data as _data
from pygsti.tools import NamedDict as _NamedDict
from pygsti.tools import listtools as _lt
from pygsti.tools.dataframetools import _process_dataframe


class Protocol(object):
    """
    An analysis routine that is run on experimental data.  A generalized notion of a  QCVV protocol.

    A Protocol object represents things like, but not strictly limited to, QCVV protocols.
    This class is essentially a serializable `run` function that takes as input a
    :class:`ProtocolData` object and returns a :class:`ProtocolResults` object.  This
    function describes the working of the "protocol".

    Parameters
    ----------
    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
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

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Returns
        -------
        Protocol
        """
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(_pathlib.Path(dirname), 'auxfile_types', quick_load=quick_load))
        ret._init_unserialized_attributes()
        return ret

    @classmethod
    def from_mongodb(cls, mongodb_collection, doc_id, quick_load=False):
        """
        Initialize a new Protocol object from a Mongo database.

        Parameters
        ----------
        mongodb_collection : pymongo.collection.Collection
            The MongoDB collection to load data from.

        doc_id : str
            The user-defined identifier of the protocol object to load.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time to load.

        Returns
        -------
        Protocol
        """
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.read_auxtree_from_mongodb(mongodb_collection, doc_id,
                                                          'auxfile_types', quick_load=quick_load))
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

    def write_to_mongodb(self, mongodb_collection, doc_id=None, session=None, overwrite_existing=False):
        """
        Write this Protocol to a MongoDB database.

        Parameters
        ----------
        mongodb_collection : pymongo.collection.Collection
            The MongoDB collection to write to.

        doc_id : str, optional
            The user-defined identifier of the Protocol to write.  Can be
            left as `None` to generate a random identifier.

        session : pymongo.client_session.ClientSession, optional
            MongoDB session object to use when interacting with the MongoDB
            database. This can be used to implement transactions
            among other things.

        overwrite_existing : bool, optional
            Whether existing documents should be overwritten.  The default of `False` causes
            a ValueError to be raised if a document with the given `doc_id` already exists.
            Setting this to `True` mimics the behaviour of a typical filesystem, where writing
            to a path can be done regardless of whether it already exists.

        Returns
        -------
        None
        """
        _io.write_obj_to_mongodb_auxtree(self, mongodb_collection, doc_id, 'auxfile_types',
                                         session=session, overwrite_existing=overwrite_existing)

    @classmethod
    def remove_from_mongodb(cls, mongodb_collection, doc_id, custom_collection_names=None, session=None):
        """
        Remove a Protocol from a MongoDB database.

        Returns
        -------
        bool
            `True` if the specified experiment design was removed, `False` if it didn't exist.
        """
        delcnt = _io.remove_auxtree_from_mongodb(mongodb_collection, doc_id, 'auxfile_types',
                                                 session=session)
        return bool(delcnt == 1)

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
    Runs a (contained) :class:`Protocol` on all the passes of a multi-pass :class:`ProtocolData`.

    A simple protocol that runs a "sub-protocol" on the passes of a :class:`ProtocolData`
    containing a :class:`MultiDataSet`.  The sub-protocol therefore doesn't need to know
    how to deal with multiple data passes. Instances of this class essentially act as
    wrappers around other protocols enabling them to handle multi-pass data.

    Parameters
    ----------
    protocol : Protocol
        The protocol to run on each pass.

    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
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
        self.auxfile_types['protocol'] = 'dir-serialized-object'

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
            #  or call to_dict?
            results.passes[pass_name] = sub_results  # pass_name is a "ds_name" key of data.dataset (a MultiDataSet)
        return results


class ProtocolRunner(object):
    """
    Used to run :class:`Protocol`(s) on an entire *tree* of data

    This class provides a way of combining multiple calls to :method:`Protocol.run`,
    potentially running multiple protocols on different data.  From the outside, a
    :class:`ProtocolRunner` object behaves similarly, and can often be used
    interchangably, with a Protocol object.  It posesses a `run` method that takes a
    :class:`ProtocolData` as input and returns a :class:`ProtocolResultsDir` that can
    contain multiple :class:`ProtocolResults` objects within it.
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

    Parameters
    ----------
    protocol_dict : dict
        A dictionary of :class:`Protocol` objects whose keys are paths
        (tuples of strings) specifying where in the data-tree that
        protocol should be run.
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
    Run the default protocol at each data-tree node.

    (Default protocols are given within :class:`ExperimentDesign` objects.)

    Parameters
    ----------
    run_passes_separately : bool, optional
        If `True`, then when multi-pass data is encountered it is split into passes
        before handing it off to the protocols.  Set this to `True` when the default
        protocols being run expect single-pass data.
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

        #Fixes to JSON codec's conversion of tuples => lists
        ret.qubit_labels = tuple(ret.qubit_labels) if isinstance(ret.qubit_labels, list) else ret.qubit_labels

        return ret

    @classmethod
    def from_mongodb(cls, mongodb, doc_id, parent=None, name=None, quick_load=False, custom_collection_names=None):
        """
        Initialize a new ExperimentDesign object from a Mongo database.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to load data from.

        doc_id : str
            The user-defined identifier of the experiment design to load.

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

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different
            types of pyGSTi objects.  In this case, only the `"edesigns"` key of this dictionary
            is relevant.  Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        Returns
        -------
        ExperimentDesign
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.read_auxtree_from_mongodb(mongodb[collection_names['edesigns']], doc_id,
                                                          'auxfile_types', quick_load=quick_load))
        ret._init_children_from_mongodb(mongodb, 'edesigns', doc_id, custom_collection_names, quick_load=quick_load)
        ret._loaded_from = (mongodb, doc_id, custom_collection_names.copy()
                            if (custom_collection_names is not None) else None)

        #Fixes to JSON codec's conversion of tuples => lists
        ret.qubit_labels = tuple(ret.qubit_labels) if isinstance(ret.qubit_labels, list) else ret.qubit_labels

        return ret

    @classmethod
    def from_edesign(cls, edesign):
        """
        Create an ExperimentDesign out of an existing experiment design.

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design to convert (use as a base).

        Returns
        -------
        ExperimentDesign
        """
        if cls != ExperimentDesign:
            raise NotImplementedError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))
        return cls(edesign.all_circuits_needing_data, edesign.qubit_labels)

    def __init__(self, circuits=None, qubit_labels=None,
                 children=None, children_dirs=None):
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

        Returns
        -------
        ExperimentDesign
        """

        self.all_circuits_needing_data = circuits if (circuits is not None) else []
        self.alt_actual_circuits_executed = None  # None means == all_circuits_needing_data
        self.default_protocols = {}
        self.tags = {}
        self._nameddict_attributes = (('qubit_labels', 'Qubits', 'category'),)
        self._loaded_from = None

        #Instructions for saving/loading certain members - if a __dict__ member
        # *isn't* listed in this dict, then it's assumed to be json-able and included
        # in the main 'meta.json' file.  Allowed values are:
        # 'text-circuit-list' - a text circuit list file
        # 'json' - a json file
        # 'pickle' - a python pickle file (use only if really needed!)
        typ = 'serialized-object' if isinstance(self.all_circuits_needing_data, _circuits.CircuitList) \
            else 'text-circuit-list'
        self.auxfile_types = {'all_circuits_needing_data': typ,
                              'alt_actual_circuits_executed': 'text-circuit-list',
                              'default_protocols': 'dict:dir-serialized-object'}

        # because TreeNode takes care of its own serialization:
        self.auxfile_types.update({'_dirs': 'none', '_vals': 'none', '_loaded_from': 'none'})

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

        if children is None: children = {}
        children_dirs = children_dirs.copy() if (children_dirs is not None) else \
            {subname: self._auto_dirname(subname) for subname in children}

        assert(set(children.keys()) == set(children_dirs.keys()))
        super().__init__(children_dirs, children)

    def _auto_dirname(self, child_key):
        """ A helper function to generate a default directory name base off of a sub-name key """
        if isinstance(child_key, (list, tuple)):
            child_key = '_'.join(map(str, child_key))
        return child_key.replace(' ', '_')

    def set_actual_circuits_executed(self, actual_circuits):
        """
        Sets a list of circuits that will actually be executed.

        This list must be parallel, and corresponding circuits must be *logically
        equivalent*, to those in `self.all_circuits_needing_data`.  For example,
        when the circuits in this design are run simultaneously with other circuits,
        the circuits in this design may need to be padded with idles.

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

    def truncate_to_circuits(self, circuits_to_keep):
        """
        Builds a new experiment design containing only the specified circuits.

        Parameters
        ----------
        circuits_to_keep : list
            A list of the circuits to keep.

        Returns
        -------
        ExperimentDesign
        """
        base = _copy.deepcopy(self)  # so this works for derived classes tools
        base._truncate_to_circuits_inplace(circuits_to_keep)
        return base

    def truncate_to_available_data(self, dataset):
        """
        Builds a new experiment design containing only those circuits present in `dataset`.

        Parameters
        ----------
        dataset : DataSet
            The dataset to filter based upon.

        Returns
        -------
        ExperimentDesign
        """
        base = _copy.deepcopy(self)  # so this works for derived classes tools
        base._truncate_to_available_data_inplace(dataset)
        return base

    def truncate_to_design(self, other_design):
        """
        Truncates this experiment design by only keeping the circuits also in `other_design`

        Parameters
        ----------
        other_design : ExperimentDesign
            The experiment design to compare with.

        Returns
        -------
        ExperimentDesign
            The truncated experiment design.
        """
        base = _copy.deepcopy(self)  # so this works for derived classes tools
        base._truncate_to_design_inplace(other_design)
        return base

    def _truncate_to_circuits_inplace(self, circuits_to_keep):
        self.all_circuits_needing_data = _circuits.CircuitList.cast(self.all_circuits_needing_data)
        if self.alt_actual_circuits_executed is not None:
            self.alt_actual_circuits_executed = _circuits.CircuitList.cast(self.alt_actual_circuits_executed)

            allc = []; actualc = []
            if isinstance(circuits_to_keep, set):
                for c, actual_c in zip(self.all_circuits_needing_data, self.alt_actual_circuits_executed):
                    if c in circuits_to_keep:
                        allc.append(c)
                        actualc.append(c)
            else:
                actual_lookup = {c: actual_c for c, actual_c in zip(self.all_circuits_needing_data,
                                                                    self.alt_actual_circuits_executed)}
                allc[:] = circuits_to_keep
                actualc[:] = [actual_lookup[c] for c in circuits_to_keep]
            self.all_circuits_needing_data.truncate(allc)
            self.alt_actual_circuits_executed.truncate(actualc)
        else:
            self.all_circuits_needing_data = self.all_circuits_needing_data.truncate(circuits_to_keep)

    def _truncate_to_design_inplace(self, other_design):
        self._truncate_to_circuits_inplace(other_design.all_circuits_needing_data)
        for _, sub_design in self._vals.items():
            sub_design._truncate_to_design_inplace(other_design)

    def _truncate_to_available_data_inplace(self, dataset):
        self.all_circuits_needing_data = _circuits.CircuitList.cast(self.all_circuits_needing_data)
        ds_circuits = self.all_circuits_needing_data.apply_aliases()
        circuits_to_keep = [c for c, ds_c in zip(self.all_circuits_needing_data, ds_circuits) if ds_c in dataset]
        self._truncate_to_circuits_inplace(circuits_to_keep)

        for _, sub_design in self._vals.items():
            sub_design._truncate_to_available_data_inplace(dataset)

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
            dirname = self._loaded_from if isinstance(self._loaded_from, str) else None
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")

        _io.write_obj_to_meta_based_dir(self, _pathlib.Path(dirname) / 'edesign', 'auxfile_types')

        self._write_children(dirname)
        self._loaded_from = str(_pathlib.Path(dirname).absolute())  # for future writes

    def write_to_mongodb(self, mongodb=None, doc_id=None, parent=None, custom_collection_names=None,
                         session=None, overwrite_existing=False):
        """
        Write this experiment design to a MongoDB database.

        Parameters
        ----------
        mongodb : pymongo.database.Database, optional
            The MongoDB instance to write data to.  Can be left as `None` if this
            experiment design was initialized from a MongoDB, e.g. with
            :function:`pygsti.io.read_edesign_from_mongodb`, in which case the same
            database used for the initial read-in is used.

        doc_id : str, optional
            The user-defined identifier of the experiment design to write.  Can be
            left as `None` if this experiment design was initialized from a MongoDB,
            in which case the same document identifier that was used at read-in is used.

        parent : ExperimentDesign, optional
            The parent experiment design, when a parent is writing this
            design as a sub-experiment-design.  Otherwise leave as None.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects.  In this case, only the `"edesigns"` key of this dictionary
            is relevant.  Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        session : pymongo.client_session.ClientSession, optional
            MongoDB session object to use when interacting with the MongoDB
            database. This can be used to implement transactions
            among other things.

        overwrite_existing : bool, optional
            Whether existing documents should be overwritten.  The default of `False` causes
            a ValueError to be raised if a document with the given `doc_id` already exists.
            Setting this to `True` mimics the behaviour of a typical filesystem, where writing
            to a path can be done regardless of whether it already exists.


        Returns
        -------
        None
        """
        loaded_from = self._loaded_from if isinstance(self._loaded_from, tuple) else None, None, None

        if mongodb is None:
            mongodb = loaded_from[0]
            if mongodb is None: raise ValueError("`mongodb` must be given because there's no default!")

        if doc_id is None:
            doc_id = loaded_from[1]
            if doc_id is None: raise ValueError("`doc_id` must be given because there's no default!")

        if custom_collection_names is None and loaded_from[2] is not None and doc_id is None:
            custom_collection_names = loaded_from[2]  # override if inferring document id

        collection_names = _io.mongodb_collection_names(custom_collection_names)
        _io.write_obj_to_mongodb_auxtree(self, mongodb[collection_names['edesigns']], doc_id, 'auxfile_types',
                                         session=session, overwrite_existing=overwrite_existing)
        self._write_children_to_mongodb(mongodb, doc_id, update_children_in_edesign=True,
                                        custom_collection_names=custom_collection_names, session=session)

    @classmethod
    def remove_from_mongodb(cls, mongodb, doc_id, custom_collection_names=None, session=None):
        """
        Remove an experiment design from a MongoDB database.

        Returns
        -------
        bool
            `True` if the specified experiment design was removed, `False` if it didn't exist.
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        cls._remove_children_from_mongodb(mongodb, 'edesigns', doc_id, custom_collection_names, session)
        delcnt = _io.remove_auxtree_from_mongodb(mongodb[collection_names['edesigns']], doc_id, 'auxfile_types',
                                                 session=session)
        return bool(delcnt is not None and delcnt.deleted_count == 1)

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

    def _create_subdata(self, subdata_name, dataset):
        """
        Creates a :class:`ProtocolData` object for a sub-experiment-design.

        Specifically, this creates the object for the sub-experiment-design
        given by `subdata_name` starting from `dataset` as the data for *this*
        experiment design.  This is used internally by :class:`ProtocolData`
        objects, and shouldn't need to be used by external users.

        Parameters
        ----------
        subdata_name : immutable
            The child (node) name of the sub-experiment design to create data for.

        dataset : DataSet
            The data for *this* experiment design.

        Returns
        -------
        ProtocolData
        """
        raise NotImplementedError("This protocol edesign cannot create any subdata!")

    def promote_to_combined(self, name="**0"):
        """
        Promote this experiment design to be a combined experiment design.

        Wraps this experiment design in a new :class:`CombinedExperimentDesign`
        whose only sub-design is this one, and returns the combined design.

        Parameters
        ----------
        name : str, optional
            The sub-design-name of this experiment design within the created
            combined experiment design.

        Returns
        -------
        CombinedExperimentDesign
        """
        return CombinedExperimentDesign.from_edesign(self, name)

    def promote_to_simultaneous(self):
        """
        Promote this experiment design to be a simultaneous experiment design.

        Wraps this experiment design in a new :class:`SimultaneousExperimentDesign`
        whose only sub-design is this one, and returns the simultaneous design.

        Returns
        -------
        SimultaneousExperimentDesign
        """
        return SimultaneousExperimentDesign.from_edesign(self)


class CircuitListsDesign(ExperimentDesign):
    """
    Experiment design specification that is comprised of multiple circuit lists.

    Parameters
    ----------
    circuit_lists : list or PlaquetteGridCircuitStructure
        A list whose elements are themselves lists of :class:`Circuit`
        objects, specifying the data that needs to be taken.  Alternatively,
        a single :class:`PlaquetteGridCircuitStructure` object containing
        a sequence of circuits lists, each at a different "x" value (usually
        the maximum circuit depth).

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
    """

    @classmethod
    def from_edesign(cls, edesign):
        """
        Create a CircuitListsDesign out of an existing experiment design.

        If `edesign` already is a circuit lists experiment design, it will
        just be returned (not a copy of it).

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design to convert (use as a base).

        Returns
        -------
        CircuitListsDesign
        """
        if cls != CircuitListsDesign:
            raise NotImplementedError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

        if isinstance(edesign, CircuitListsDesign):
            return edesign
        elif isinstance(edesign, (CombinedExperimentDesign, SimultaneousExperimentDesign)):
            circuit_lists = [subd.all_circuits_needing_data for k, subd in edesign.items()]
            return cls(circuit_lists, edesign.all_circuits_needing_data, edesign.qubit_labels, remove_duplicates=False)
        elif isinstance(edesign, ExperimentDesign):
            return cls([edesign.all_circuits_needing_data], edesign.all_circuits_needing_data,
                       edesign.qubit_labels, remove_duplicates=False)
        else:
            raise ValueError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

    def __init__(self, circuit_lists, all_circuits_needing_data=None, qubit_labels=None,
                 nested=False, remove_duplicates=True):
        """
        Create a new CircuitListsDesign object.

        Parameters
        ----------
        circuit_lists : list or PlaquetteGridCircuitStructure
            A list whose elements are themselves lists of :class:`Circuit`
            objects, specifying the data that needs to be taken.  Alternatively,
            a single :class:`PlaquetteGridCircuitStructure` object containing
            a sequence of circuits lists, each at a different "x" value (usually
            the maximum circuit depth).

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

        if isinstance(circuit_lists, _circuits.PlaquetteGridCircuitStructure):
            master = circuit_lists
            circuit_lists = [master.truncate(xs_to_keep=master.xs[0:i + 1]) for i in range(len(master.xs))]
            nested = True  # (by this construction)

        if all_circuits_needing_data is not None:
            all_circuits = all_circuits_needing_data  # (ok if this is a CircuitList)
        elif nested and len(circuit_lists) > 0:
            all_circuits = circuit_lists[-1]  # (ok if this is a CircuitList)
        else:
            all_circuits = []
            for lst in circuit_lists:
                all_circuits.extend(lst)  # Note: this should work even for type(lst) == CircuitList
            if remove_duplicates:
                _lt.remove_duplicates_in_place(all_circuits)

        self.circuit_lists = circuit_lists
        self.nested = nested

        super().__init__(all_circuits, qubit_labels)
        self.auxfile_types['circuit_lists'] = 'list:serialized-object' \
            if any([isinstance(lst, _circuits.CircuitList) for lst in circuit_lists]) else 'list:text-circuit-list'

    def truncate_to_lists(self, list_indices_to_keep):
        """
        Truncates this experiment design by only keeping a subset of its circuit lists.

        Parameters
        ----------
        list_indices_to_keep : iterable
            A list of the (integer) list indices to keep.

        Returns
        -------
        CircuitListsDesign
            The truncated experiment design.
        """
        return CircuitListsDesign([self.circuit_lists[i] for i in list_indices_to_keep],
                                  qubit_labels=self.qubit_labels, nested=self.nested)

    def _truncate_to_circuits_inplace(self, circuits_to_keep):
        truncated_circuit_lists = [_circuits.CircuitList.cast(lst).truncate(circuits_to_keep)
                                   for lst in self.circuit_lists]
        self.circuit_lists = truncated_circuit_lists
        self.nested = False  # we're not sure whether the truncated lists are nested
        super()._truncate_to_circuits_inplace(circuits_to_keep)

    def _truncate_to_design_inplace(self, other_design):
        self.circuit_lists = [my_circuit_list.truncate(other_circuit_list) for my_circuit_list, other_circuit_list
                              in zip(self.circuit_lists, other_design.circuit_lists)]
        super()._truncate_to_design_inplace(other_design)

    def _truncate_to_available_data_inplace(self, dataset):
        truncated_lists = [_circuits.CircuitList.cast(clist).truncate_to_dataset(dataset)
                           for clist in self.circuit_lists]
        self.circuit_lists = truncated_lists
        #self.nested = False
        super()._truncate_to_available_data_inplace(dataset)


class CombinedExperimentDesign(ExperimentDesign):  # for multiple designs on the same dataset
    """
    An experiment design that combines the specifications of one or more "sub-designs".

    The sub-designs are preserved as children under the
    :class:`CombinedExperimentDesign` instance, creating a "data-tree" structure.
    The :class:`CombinedExperimentDesign` object itself simply merges all of the
    circuit lists.

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

    interleave : bool, optional
        Whether the circuits of the `sub_designs` should be interleaved to
        form the circuit ordering of this experiment design.
    """

    @classmethod
    def from_edesign(cls, edesign, name):
        """
        Create a combined experiment design out of an existing experiment design.

        This makes `edesign` the one and only member of a new combined experiment design,
        even in `edesign` is already a `CombinedExperimentDesign`.

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design to convert (use as a base).

        name : str
            The sub-name of `edesign` within the returned combined experiment design.

        Returns
        -------
        CombinedExperimentDesign
        """
        if cls != CombinedExperimentDesign:
            raise NotImplementedError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

        if isinstance(edesign, ExperimentDesign):
            return cls({name: edesign}, edesign.all_circuits_needing_data, edesign.qubit_labels)
        else:
            raise ValueError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

    def __init__(self, sub_designs, all_circuits=None, qubit_labels=None, sub_design_dirs=None,
                 interleave=False):
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

        interleave : bool, optional
            Whether the circuits of the `sub_designs` should be interleaved to
            form the circuit ordering of this experiment design.

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

        super().__init__(all_circuits, qubit_labels, sub_designs, sub_design_dirs)

    def _create_subdata(self, sub_name, dataset):
        """
        Creates a :class:`ProtocolData` object for a sub-experiment-design.

        Specifically, this creates the object for the sub-experiment-design
        given by `subdata_name` starting from `dataset` as the data for *this*
        experiment design.  This is used internally by :class:`ProtocolData`
        objects, and shouldn't need to be used by external users.

        Parameters
        ----------
        sub_name : immutable
            The child (node) name of the sub-experiment design to create data for.

        dataset : DataSet
            The data for *this* experiment design.

        Returns
        -------
        ProtocolData
        """
        sub_circuits = self[sub_name].all_circuits_needing_data
        if isinstance(dataset, dict):  # then do truncation "element-wise"
            truncated_ds = {k: ds.truncate(sub_circuits) for k, ds in dataset.items()}
            for tds in truncated_ds.values(): tds.add_std_nqubit_outcome_labels(len(self[sub_name].qubit_labels))
        else:
            truncated_ds = dataset.truncate(sub_circuits)  # maybe have filter_dataset also do this?
            sub_nqubits = len(self[sub_name].qubit_labels)
            outcome_labels = tuple(filter(lambda ol: (len(ol) != 1
                                                      or any([letter not in ('0', '1') for letter in ol[0]])
                                                      or len(ol[0]) == sub_nqubits), dataset.outcome_labels))
            truncated_ds.add_outcome_labels(outcome_labels)  # make sure truncated ds has all outcomes
            #truncated_ds.add_std_nqubit_outcome_labels(len(self[sub_name].qubit_labels))  # can be very SLOW
        return ProtocolData(self[sub_name], truncated_ds)

    def __setitem__(self, key, val):
        # must set base class self._vals and self._dirs (see treenode.py)
        if not isinstance(val, ExperimentDesign):
            raise ValueError("Only experiment designs can be set as sub-designs of a CombinedExperimentDesign!")

        #Check whether the new design adds any more circuits (it's not allowed to
        # because other objects, e.g. a ProtocolData object, may hold a reference to
        # this combined experiment design (e.g., to index data) and expect that it will
        # not change.
        current_circuits = set(self.all_circuits_needing_data)
        new_circuits = [c for c in val.all_circuits_needing_data if c not in current_circuits]
        if len(new_circuits) > 0:
            raise ValueError((("Adding this experiment design would add %d new circuits.  Adding circuits is not"
                               " allowed because an experiment design's circuit list may be used to index data.  The"
                               " circuits that would be added are:\n") % len(new_circuits))
                             + "\n".join([c.str for c in new_circuits[0:10]])
                             + ("\n..." if len(new_circuits) > 10 else ""))

        self._dirs[key] = self._auto_dirname(key)
        self._vals[key] = val


class SimultaneousExperimentDesign(ExperimentDesign):
    """
    An experiment design whose circuits are the tensor-products of the circuits from one or more sub-designs.

    The sub-:class:`ExperimentDesign` objects must act on disjoint sets of qubits.  The sub-designs
    are preserved as children under the :class:`SimultaneousExperimentDesign` instance, creating
    a "data-tree" structure.

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
    """

    @classmethod
    def from_edesign(cls, edesign):
        """
        Create a simultaneous experiment design out of an existing experiment design.

        This makes `edesign` the one and only member of a new simultanieous experiment
        design, even in `edesign` is already a `SimultaneousExperimentDesign`.

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design to convert (use as a base).

        Returns
        -------
        SimultaneousExperimentDesign
        """
        if cls != SimultaneousExperimentDesign:
            raise NotImplementedError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

        if isinstance(edesign, ExperimentDesign):
            return cls([edesign], None, edesign.qubit_labels)
        else:
            raise ValueError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

    #@classmethod
    #def from_tensored_circuits(cls, circuits, template_edesign, qubit_labels_per_edesign):
    #    pass #Useful??? - need to break each circuit into different parts
    # based on qubits, then copy (?) template edesign and just replace itself
    # all_circuits_needing_data member?

    def __init__(self, edesigns, tensored_circuits=None, qubit_labels=None):
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

            def pad(subcs, actually_padded_mask):
                maxLen = max([len(c) if (c is not None) else 0 for c in subcs])
                padded = []
                for i, c in enumerate(subcs):
                    if c is not None and len(c) < maxLen:
                        padded.append(c.insert_idling_layers(None, maxLen - len(c)))
                        actually_padded_mask[i] = True
                    else:
                        padded.append(c)
                assert(all([len(c) == maxLen for c in padded if c is not None]))
                return padded

            actually_padded_msk = _np.zeros(len(edesigns), dtype=bool)
            padded_circuit_lists = [list() for des in edesigns]
            for subcircuits in zip(*circuits_per_edesign):
                c = _circuits.Circuit(num_lines=0, editable=True)  # Creates a empty circuit over no wires
                padded_subcircuits = pad(subcircuits, actually_padded_msk)  # updates actually_padded array
                for subc in padded_subcircuits:
                    if subc is not None:
                        c.tensor_circuit_inplace(subc)
                c.line_labels = qubit_labels
                c.done_editing()
                tensored_circuits.append(c)
                for lst, subc in zip(padded_circuit_lists, padded_subcircuits):
                    if subc is not None: lst.append(subc)

            for i, (padded_circuits, actually_padded) in enumerate(zip(padded_circuit_lists, actually_padded_msk)):
                if actually_padded:
                    des = _copy.deepcopy(edesigns[i])  # since we're setting actual circuits executed.
                    des.set_actual_circuits_executed(padded_circuits)
                    edesigns[i] = des  # update edesigns list with copy

        sub_designs = {des.qubit_labels: des for des in edesigns}
        sub_design_dirs = {qlbls: '_'.join(map(str, qlbls)) for qlbls in sub_designs}
        super().__init__(tensored_circuits, qubit_labels, sub_designs, sub_design_dirs)

    def _create_subdata(self, qubit_labels, dataset):
        """
        Creates a :class:`ProtocolData` object for a sub-experiment-design.

        Specifically, this creates the object for the sub-experiment-design
        given by `subdata_name` starting from `dataset` as the data for *this*
        experiment design.  This is used internally by :class:`ProtocolData`
        objects, and shouldn't need to be used by external users.

        Parameters
        ----------
        qubit_labels : tuple
            The child (node) label of the sub-experiment design to create data for.

        dataset : DataSet
            The data for *this* experiment design.

        Returns
        -------
        ProtocolData
        """
        if isinstance(dataset, _data.MultiDataSet):
            raise NotImplementedError("SimultaneousExperimentDesigns don't work with multi-pass data yet.")

        all_circuits = self.all_circuits_needing_data
        qubit_ordering = all_circuits[0].line_labels  # first circuit in *this* edesign determines qubit order
        qubit_index = {qlabel: i for i, qlabel in enumerate(qubit_ordering)}
        sub_design = self[qubit_labels]
        qubit_indices = [qubit_index[ql] for ql in qubit_labels]  # order determined by first circuit (see above)

        if isinstance(dataset, dict):  # then do filtration "element-wise"
            filtered_ds = {k: _data.filter_dataset(ds, qubit_labels, qubit_indices) for k, ds in dataset.items()}
            for fds in filtered_ds.values(): fds.add_std_nqubit_outcome_labels(len(qubit_labels))
        else:
            filtered_ds = _data.filter_dataset(dataset, qubit_labels, qubit_indices)  # Marginalize dataset
            filtered_ds.add_std_nqubit_outcome_labels(len(qubit_labels))  # ensure filtered_ds has appropriate outcomes

        if sub_design.alt_actual_circuits_executed:
            actual_to_desired = _collections.defaultdict(lambda: None)
            actual_to_desired.update({actual: desired for actual, desired in
                                      zip(sub_design.alt_actual_circuits_executed,
                                          sub_design.all_circuits_needing_data)})
            if isinstance(dataset, dict):  # then do circuit processing "element-wise"
                for k in filtered_ds:
                    filtered_ds[k] = filtered_ds[k].process_circuits(lambda c: actual_to_desired[c], aggregate=False)
            else:
                filtered_ds = filtered_ds.process_circuits(lambda c: actual_to_desired[c], aggregate=False)
        return ProtocolData(sub_design, filtered_ds)


class FreeformDesign(ExperimentDesign):
    """
    Experiment design holding an arbitrary circuit list and meta data.

    Parameters
    ----------
    circuits : list or dict
        A list of the circuits needing data.  If None, then the list is empty.

    qubit_labels : tuple, optional
        The qubits that this experiment design applies to. If None, the
        line labels of the first circuit is used.
    """

    @classmethod
    def from_dataframe(cls, df, qubit_labels=None):
        """
        Create a FreeformDesign from a pandas dataframe.

        Parameters
        ----------
        df : pandas.Dataframe
            A dataframe containing a `"Circuit"` column and possibly others.

        qubit_labels : tuple, optional
            The qubits that this experiment design applies to. If None, the
            line labels of the first circuit is used.

        Returns
        -------
        FreeformDesign
        """
        circuits = {}
        for index, row in df.iterrows():
            data = {k: v for k, v in row.items() if k != 'Circuit'}
            circuits[_circuits.Circuit(row['Circuit'])] = data
        return cls(circuits, qubit_labels)

    @classmethod
    def from_edesign(cls, edesign):
        """
        Create a FreeformDesign out of an existing experiment design.

        If `edesign` already is a free-form experiment design, it will
        just be returned (not a copy of it).

        Parameters
        ----------
        edesign : ExperimentDesign
            The experiment design to convert (use as a base).

        Returns
        -------
        FreeformDesign
        """
        if cls != FreeformDesign:
            raise NotImplementedError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

        if isinstance(edesign, FreeformDesign):
            return edesign
        elif isinstance(edesign, ExperimentDesign):
            return cls(edesign.all_circuits_needing_data, edesign.qubit_labels)
        else:
            raise ValueError("Cannot convert a %s to a %s!" % (str(type(edesign)), str(cls)))

    def __init__(self, circuits, qubit_labels=None):
        if isinstance(circuits, dict):
            self.aux_info = circuits.copy()
            circuits = list(circuits.keys())
        else:
            self.aux_info = {c: None for c in circuits}
        super().__init__(circuits, qubit_labels)
        self.auxfile_types['aux_info'] = 'pickle'

    def _truncate_to_circuits_inplace(self, circuits_to_keep):
        truncated_aux_info = {k: v for k, v in self.aux_info.items() if k in circuits_to_keep}
        self.aux_info = truncated_aux_info
        super()._truncate_to_circuits_inplace(circuits_to_keep)

    def to_dataframe(self, pivot_valuename=None, pivot_value="Value", drop_columns=False):
        cdict = _NamedDict('Circuit', None)
        for cir, info in self.aux_info.items():
            cdict[cir.str] = _NamedDict('ValueName', 'category', items=info)
        df = cdict.to_dataframe()
        return _process_dataframe(df, pivot_valuename, pivot_value, drop_columns, preserve_order=True)


class ProtocolData(_TreeNode):
    """
    Represents the experimental data needed to run one or more QCVV protocols.

    This class contains a :class:`ProtocolIput`, which describes a set of circuits,
    and a :class:`DataSet` (or :class:`MultiDataSet`) that holds data for these
    circuits.  These members correspond to the `.edesign` and `.dataset` attributes.

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

    Attributes
    ----------
    passes : dict
        A dictionary of the data on a per-pass basis (works even it there's just one pass).
    """
    COLL_DATASET = "pygsti_datasets"
    COLL_CACHE = "pygsti_caches"

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
            The sub-name of the object being loaded, i.e. the
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
            _io.read_edesign_from_dir(dirname, quick_load=quick_load)

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
                dataset = _io.read_dataset(dataset_files[0], ignore_zero_count_lines=False, verbosity=0)
            else:
                dataset = {pth.stem: _io.read_dataset(pth, ignore_zero_count_lines=False, verbosity=0)
                           for pth in dataset_files}
                #FUTURE: use MultiDataSet, BUT in addition to init_from_dict we'll need to add truncate, filter, and
                # process_circuits support for MultiDataSet objects -- for now (above) we just use dicts of DataSets.
                #raise NotImplementedError("Need to implement MultiDataSet.init_from_dict!")
                #dataset = _data.MultiDataSet.init_from_dict(
                #    {pth.name: _io.read_dataset(pth, verbosity=0) for pth in dataset_files})

        cache = _io.metadir._read_json_or_pkl_files_to_dict(data_dir / 'cache')

        ret = cls(edesign, dataset, cache)
        ret._init_children(dirname, 'data', quick_load=quick_load)  # loads child nodes
        return ret

    @classmethod
    def from_mongodb(cls, mongodb, doc_id, parent=None, name=None, quick_load=False, custom_collection_names=None):
        """
        Initialize a new ProtocolData object from a Mongo database.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to load data from.

        doc_id : str
            The user-defined identifier of the protocol data to load.

        parent : ProtocolData, optional
            The parent data object, if there is one.  This is needed for
            sub-data objects which reference/inherit their parent's dataset.
            Primarily used internally - if in doubt, leave this as `None`.

        name : str, optional
            The sub-name of the object being loaded, i.e. the
            key of this data object beneath `parent`.  Only used when
            `parent` is not None.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time, e.g. the actual raw data set(s). This can be useful
            when loading takes a long time and all the information of interest
            lies elsewhere, e.g. in an encompassing results object.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects.  In this case, the `"edesigns"` and `"data"` keys of this dictionary
            are relevant.  Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        Returns
        -------
        ProtocolData
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        edesign = parent.edesign[name] if parent and name else \
            _io.read_edesign_from_mongodb(mongodb, doc_id, quick_load, comm=None,
                                          custom_collection_names=custom_collection_names)

        if quick_load:
            dataset = None  # don't load any dataset - just the cache (usually b/c loading is slow)
        else:
            #Load dataset or multidataset from database
            ds_names = [ds_doc['name'] for ds_doc in mongodb[collection_names['data']][cls.COLL_DATASET].find(
                {'protocoldata_parent': doc_id}, ['name'])]
            if ds_names == [None]:
                dataset = _io.read_dataset_from_mongodb(mongodb[collection_names['data']][cls.COLL_DATASET],
                                                        {'name': None,
                                                         'protocoldata_parent': doc_id})
            else:
                dataset = {}
                for dsname in ds_names:
                    dataset[dsname] = _io.read_dataset_from_mongodb(
                        mongodb[collection_names['data'][cls.COLL_DATASET]],
                        {'name': dsname,
                         'protocoldata_parent': doc_id})

        cache = _io.read_dict_from_mongodb(mongodb[collection_names['data']][cls.COLL_CACHE],
                                           {'member': 'cache',
                                            'protocoldata_parent': doc_id})

        ret = cls(edesign, dataset, cache)
        ret._init_children_from_mongodb(mongodb, 'data', doc_id, custom_collection_names, quick_load=quick_load)
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

        if isinstance(self.dataset, (_data.MultiDataSet, dict)):  # can be dict of DataSets instead of a multi-ds
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
        super().__init__(self.edesign._dirs, {})  # children created on-demand

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
        return self.edesign._create_subdata(key, self.dataset)

    def copy(self):
        """
        Make a copy of this object.

        Returns
        -------
        ProtocolData
        """
        if list(self._passdatas.keys()) == [None]:
            # Don't copy ourself recursively
            self._passdatas = {}
            cpy = _copy.deepcopy(self)
            self._passdatas = {None: self}
        else:
            cpy = _copy.deepcopy(self)
        return cpy

    @property
    def passes(self):
        """
        A dictionary of the data on a per-pass basis (works even it there's just one pass).

        Returns
        -------
        dict
        """
        return self._passdatas

    def is_multipass(self):
        """
        Whether this protocol data contains multiple passes.

        More accurately, whether the `.dataset` of this object is a
        :class:`MultiDataSet`.

        Returns
        -------
        bool
        """
        return isinstance(self.dataset, (_data.MultiDataSet, dict))

    #def underlying_tree_paths(self):
    #    return self.edesign.get_tree_paths()

    def prune_tree(self, paths, paths_are_sorted=False):
        """
        Prune the tree rooted here to include only the given paths, discarding all else.

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
        filtered_edesign = self.edesign.prune_tree(paths, paths_are_sorted)
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
            dirname = self.edesign._loaded_from if isinstance(self.edesign._loaded_from, str) else None
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")
        dirname = _pathlib.Path(dirname)
        data_dir = dirname / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        _io.metadir._obj_to_meta_json(self, data_dir)

        if parent is None:
            self.edesign.write(dirname)  # otherwise assume parent has already written edesign

        if self.dataset is not None:  # otherwise don't write any dataset
            if parent and (self.dataset is parent.dataset):  # then no need to write any data
                assert(len(list(data_dir.glob('*.txt'))) == 0), "There shouldn't be *.txt files in %s!" % str(data_dir)
            else:
                data_dir.mkdir(exist_ok=True)
                if isinstance(self.dataset, (_data.MultiDataSet, dict)):
                    for dsname, ds in self.dataset.items():
                        _io.write_dataset(data_dir / (dsname + '.txt'), ds)
                else:
                    _io.write_dataset(data_dir / 'dataset.txt', self.dataset)

        if self.cache:
            _io.write_dict_to_json_or_pkl_files(self.cache, data_dir / 'cache')

        self._write_children(dirname, write_subdir_json=False)  # writes sub-datas

    def write_to_mongodb(self, mongodb, doc_id, parent=None, custom_collection_names=None,
                         session=None, overwrite_existing=False):
        """
        Write this protocol data to a MongoDB database.

        Parameters
        ----------
        mongodb : pymongo.database.Database, optional
            The MongoDB instance to write data to.  Can be left as `None` if this
            protocol data was initialized from a MongoDB, e.g. with
            :function:`pygsti.io.read_data_from_mongodb`, in which case the same
            database used for the initial read-in is used.

        doc_id : str, optional
            The user-defined identifier of the protocol data to write.  Can be
            left as `None` if this protocol data was initialized from a MongoDB,
            in which case the same document identifier that was used at read-in is used.

        parent : ProtocolData, optional
            The parent protocol data, when a parent is writing this
            data as a sub-protocol-data object.  Otherwise leave as None.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects.  In this case, the `"edesigns"` and `"data"` keys of this dictionary
            are relevant.  Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        session : pymongo.client_session.ClientSession, optional
            MongoDB session object to use when interacting with the MongoDB
            database. This can be used to implement transactions
            among other things.

        overwrite_existing : bool, optional
            Whether existing documents should be overwritten.  The default of `False` causes
            a ValueError to be raised if a document with the given `doc_id` already exists.
            Setting this to `True` mimics the behaviour of a typical filesystem, where writing
            to a path can be done regardless of whether it already exists.


        Returns
        -------
        None
        """
        loaded_from = self.edesign._loaded_from if isinstance(self.edesign._loaded_from, tuple) else None, None, None

        if mongodb is None:
            mongodb = loaded_from[0]
            if mongodb is None: raise ValueError("`mongodb` must be given because there's no default!")

        if doc_id is None:
            doc_id = loaded_from[1]
            if doc_id is None: raise ValueError("`doc_id` must be given because there's no default!")

        if custom_collection_names is None and loaded_from[2] is not None and doc_id is None:
            custom_collection_names = loaded_from[2]  # override if inferring document id

        collection_names = _io.mongodb_collection_names(custom_collection_names)

        #Write our class information to mongodb, even though we don't currently use this when loading (FUTURE work)
        _io.write_obj_to_mongodb_auxtree(self, mongodb[collection_names['data']], doc_id,
                                         auxfile_types_member=None,
                                         omit_attributes=['edesign', 'dataset', '_passdatas', 'cache'],
                                         session=session, overwrite_existing=overwrite_existing)

        if parent is None:  # otherwise assume parent has already written edesign
            self.edesign.write_to_mongodb(mongodb, doc_id, None, custom_collection_names, session)

        if self.dataset is not None:  # otherwise don't write any dataset
            if parent and (self.dataset is parent.dataset):  # then no need to write any data
                assert(mongodb[collection_names['data']][self.COLL_DATASET].count_documents(
                    {'protocoldata_parent': doc_id}, session=session) == 0)
            else:
                if isinstance(self.dataset, (_data.MultiDataSet, dict)):
                    for dsname, ds in self.dataset.items():
                        _io.write_dataset_to_mongodb(ds, mongodb[collection_names['data']][self.COLL_DATASET],
                                                     {'name': dsname,
                                                      'protocoldata_parent': doc_id},
                                                     session=session)
                else:
                    _io.write_dataset_to_mongodb(self.dataset, mongodb[collection_names['data']][self.COLL_DATASET],
                                                 {'name': None,
                                                  'protocoldata_parent': doc_id},
                                                 session=session)

        if self.cache:
            _io.write_dict_to_mongodb(self.cache, mongodb[collection_names['data']][self.COLL_CACHE],
                                      {'member': 'cache',
                                       'protocoldata_parent': doc_id},
                                      session=session)

        self._write_children_to_mongodb(mongodb, doc_id, update_children_in_edesign=False,
                                        custom_collection_names=custom_collection_names,
                                        session=session)  # writes sub-datas

    @classmethod
    def remove_from_mongodb(cls, mongodb, doc_id, custom_collection_names=None, session=None):
        """
        Remove a protocol data object from a MongoDB database.

        Returns
        -------
        bool
            `True` if the specified data was removed, `False` if it didn't exist.
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        cls._remove_children_from_mongodb(mongodb, 'data', doc_id, custom_collection_names, session)

        _io.remove_dict_from_mongodb(mongodb[collection_names['data']][cls.COLL_CACHE],
                                     {'member': 'cache',
                                      'protocoldata_parent': doc_id},
                                     session=session)

        for ds_doc in mongodb[collection_names['data']][cls.COLL_DATASET].find(
                {'protocoldata_parent': doc_id}, ['name']):
            _io.remove_dataset_from_mongodb(mongodb[collection_names['data']][cls.COLL_DATASET],
                                            {'name': ds_doc['name'],
                                             'protocoldata_parent': doc_id},
                                            session=session)

        # Perhaps parent has already done this, but try to remove edesign anyway
        _io.remove_edesign_from_mongodb(mongodb, doc_id, custom_collection_names, session)

        # Remove ProtocolData document itself
        delcnt = _io.remove_auxtree_from_mongodb(mongodb[collection_names['data']], doc_id, 'auxfile_types',
                                                 session=session)
        return bool(delcnt is not None and delcnt.deleted_count == 1)

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

    def to_dataframe(self, pivot_valuename=None, pivot_value=None, drop_columns=False):
        """
        Create a Pandas dataframe with this data.

        Parameters
        ----------
        pivot_valuename : str, optional
            If not None, the resulting dataframe is pivoted using `pivot_valuename`
            as the column whose values name the pivoted table's column names.
            If None and `pivot_value` is not None,`"ValueName"` is used.

        pivot_value : str, optional
            If not None, the resulting dataframe is pivoted such that values of
            the `pivot_value` column are rearranged into new columns whose names
            are given by the values of the `pivot_valuename` column. If None and
            `pivot_valuename` is not None,`"Value"` is used.

        drop_columns : bool or list, optional
            A list of column names to drop (prior to performing any pivot).  If
            `True` appears in this list or is given directly, then all
            constant-valued columns are dropped as well.  No columns are dropped
            when `drop_columns == False`.

        Returns
        -------
        pandas.DataFrame
        """
        cdict = _NamedDict('Circuit', None)
        if isinstance(self.dataset, _data.FreeformDataSet):
            for cir, i in self.dataset.cirIndex.items():
                d = _NamedDict('ValueName', 'category', items=self.dataset._info[i])
                if isinstance(self.edesign, FreeformDesign):
                    edesign_aux = self.edesign.aux_info[cir]
                    if edesign_aux is not None:
                        d.update(edesign_aux)
                cdict[cir.str] = d
        else:
            raise NotImplementedError("Can only convert free-form data to dataframes currently.")

        df = cdict.to_dataframe()
        return _process_dataframe(df, pivot_valuename, pivot_value, drop_columns, preserve_order=True)


class ProtocolResults(object):
    """
    Stores the results from running a QCVV protocol on data.

    A :class:`ProtocolResults` object Contains a :class:`ProtocolData` object and stores
    the results of running a :class:`Protcocol` on this data.

    Parameters
    ----------
    data : ProtocolData
        The input data from which these results are derived.

    protocol_instance : Protocol
        The protocol that created these results.
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
            _io.read_data_from_dir(dirname, quick_load=quick_load)
        assert(ret.name == name), "ProtocolResults name inconsistency!"
        return ret

    @classmethod
    def _from_dir_partial(cls, dirname, quick_load=False, load_protocol=False):
        """
        Internal method for loading only the results-specific data, and not the `data` member.
        This method may be used independently by derived ProtocolResults objects which contain
        multiple sub-results (e.g. MultiPassResults)
        """
        ignore = ('type',) if load_protocol else ('type', 'protocol')
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.load_meta_based_dir(dirname, 'auxfile_types', ignore, quick_load=quick_load))
        return ret

    @classmethod
    def from_mongodb(cls, mongodb, doc_id, name, preloaded_data=None, quick_load=False, custom_collection_names=None):
        """
        Initialize a new ProtocolResults object from a Mongo database.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to load data from.

        doc_id : str
            The user-defined identifier of the results *directory* containing the
            results object to be loaded.

        name : str
            The name, within the directory given by `doc_id`, of the particular results
            object to load (there can be multiple under a given `doc_id`).

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time, e.g. the actual raw data set(s). This can be useful
            when loading takes a long time and all the information of interest
            lies elsewhere, e.g. in an encompassing results object.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects. Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        Returns
        -------
        ProtocolResults
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        result_id = doc_id + '/' + name  # derive_child_id_from_parent_id(doc_id, name)
        ret = cls._from_mongodb_partial(mongodb[collection_names['results']], result_id, quick_load, load_protocol=True)
        ret.data = preloaded_data if (preloaded_data is not None) else \
            _io.read_data_from_mongodb(mongodb, doc_id, quick_load=quick_load,
                                       custom_collection_names=custom_collection_names)
        assert(ret.name == name), "ProtocolResults name inconsistency!"
        return ret

    @classmethod
    def _from_mongodb_partial(cls, mongodb_collection, result_id, quick_load=False, load_protocol=False):
        """
        Internal method for loading only the results-specific data, and not the `data` member.
        This method may be used independently by derived ProtocolResults objects which contain
        multiple sub-results (e.g. MultiPassResults)
        """
        ignore = ('type',) if load_protocol else ('type', 'protocol')
        ret = cls.__new__(cls)
        ret.__dict__.update(_io.read_auxtree_from_mongodb(mongodb_collection, result_id,
                                                          'auxfile_types', ignore, quick_load=quick_load))
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
        self.auxfile_types = {'data': 'none', 'protocol': 'dir-serialized-object'}

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
            dirname = self.data.edesign._loaded_from if isinstance(self.data.edesign._loaded_from, str) else None
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

    def write_to_mongodb(self, mongodb=None, doc_id=None, data_already_written=False,
                         custom_collection_names=None, session=None):
        """
        Write these protocol results to a MongoDB database.

        Parameters
        ----------
        mongodb : pymongo.database.Database, optional
            The MongoDB instance to write data to.  Can be left as `None` if this
            protocol data was initialized from a MongoDB, e.g. with
            :function:`pygsti.io.read_results_from_mongodb`, in which case the same
            database used for the initial read-in is used.

        doc_id : str, optional
            The user-defined identifier of the results *directory*. Can be
            left as `None` if this protocol results object was initialized from a MongoDB,
            in which case the same document identifier that was used at read-in is used.

        data_already_written : bool, optional
            Set this to True if you're sure the `.data` :class:`ProtocolData` object
            within this results object has already been written to `mongodb`.  Leaving
            this as the default is a safe option.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects. Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        session : pymongo.client_session.ClientSession, optional
            MongoDB session object to use when interacting with the MongoDB
            database. This can be used to implement transactions
            among other things.

        Returns
        -------
        None
        """
        loaded_from = self.data.edesign._loaded_from if isinstance(self.data.edesign._loaded_from, tuple) \
            else None, None, None

        if mongodb is None:
            mongodb = loaded_from[0]
            if mongodb is None: raise ValueError("`mongodb` must be given because there's no default!")

        if doc_id is None:
            doc_id = loaded_from[1]
            if doc_id is None: raise ValueError("`doc_id` must be given because there's no default!")

        if custom_collection_names is None and loaded_from[2] is not None and doc_id is None:
            custom_collection_names = loaded_from[2]  # override if inferring document id

        #write edesign and data
        if not data_already_written:
            self.data.write_to_mongodb(mongodb, doc_id, custom_collection_names=custom_collection_names,
                                       session=session)

        #write qtys to results dir
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        result_id = doc_id + '/' + self.name  # derive_child_id_from_parent_id(doc_id, self.name)
        self._write_partial_to_mongodb(mongodb[collection_names['results']], result_id, write_protocol=True,
                                       additional_meta={'directory_id': doc_id},
                                       session=session)

    def _write_partial_to_mongodb(self, mongodb_collection, result_id, write_protocol=False,
                                  additional_meta=None, session=None, overwrite_existing=False):
        """
        Internal method used to write the results-specific data to a MongoDB.
        This method does not write the object's `data` member, which must be
        serialized separately.
        """
        _io.write_obj_to_mongodb_auxtree(self, mongodb_collection, result_id,
                                         'auxfile_types', omit_attributes=() if write_protocol else ('protocol',),
                                         additional_meta=additional_meta, session=session,
                                         overwrite_existing=overwrite_existing)

    @classmethod
    def remove_from_mongodb(cls, mongodb, doc_id, name, custom_collection_names=None, session=None):
        """
        Remove a :class:`ProtocolResults` object from a MongoDB database.

        Returns
        -------
        bool
            `True` if the specified data was removed, `False` if it didn't exist.
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        result_id = doc_id + '/' + name  # derive_child_id_from_parent_id(doc_id, name)
        delcnt = _io.remove_auxtree_from_mongodb(mongodb[collection_names['results']], result_id, 'auxfile_types',
                                                 session=session)
        return bool(delcnt is not None and delcnt.deleted_count == 1)

    def to_nameddict(self):
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
                vals[k] = v.to_nameddict()
            elif isinstance(v, _NamedDict):
                vals[k] = v
            elif isinstance(v, dict):
                pass  # don't know how to make a dict into a (nested) NamedDict
            else:  # non-dicts are ok to just store
                vals[k] = v
        return vals

    def to_dataframe(self, pivot_valuename=None, pivot_value=None, drop_columns=False):
        """
        Convert these results into Pandas dataframe.

        Parameters
        ----------
        pivot_valuename : str, optional
            If not None, the resulting dataframe is pivoted using `pivot_valuename`
            as the column whose values name the pivoted table's column names.
            If None and `pivot_value` is not None,`"ValueName"` is used.

        pivot_value : str, optional
            If not None, the resulting dataframe is pivoted such that values of
            the `pivot_value` column are rearranged into new columns whose names
            are given by the values of the `pivot_valuename` column. If None and
            `pivot_valuename` is not None,`"Value"` is used.

        drop_columns : bool or list, optional
            A list of column names to drop (prior to performing any pivot).  If
            `True` appears in this list or is given directly, then all
            constant-valued columns are dropped as well.  No columns are dropped
            when `drop_columns == False`.

        Returns
        -------
        DataFrame
        """
        df = self.to_nameddict().to_dataframe()
        return _process_dataframe(df, pivot_valuename, pivot_value, drop_columns)

    def __str__(self):
        import pprint
        P = pprint.PrettyPrinter()
        return P.pformat(self.to_nameddict())


class MultiPassResults(ProtocolResults):
    """
    Holds the results of a single protocol on multiple "passes" (sets of data, typically taken at different times).

    The results of each pass are held as a separate :class:`ProtcolResults`
    object within the `.passes` attribute.

    Parameters
    ----------
    data : ProtocolData
        The input data from which these results are derived.

    protocol_instance : Protocol
        The protocol that created these results.
    """

    @classmethod
    def from_dir(cls, dirname, name, preloaded_data=None, quick_load=False):
        """
        Initialize a new MultiPassResults object from `dirname` / results / `name`.

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

        self.passes = _collections.OrderedDict()  # _NamedDict('Pass', 'category') - to_nameddict takes care of this
        self.auxfile_types['passes'] = 'dict:partialdir-serialized-object'

    def to_nameddict(self):
        """
        Create a :class:`NamedDict` of the results within this object.

        Returns
        -------
        NamedDict
        """
        # essentially inject a 'Pass' dict right beneath the outer-most Protocol Name dict
        ret = _NamedDict('ProtocolName', 'category')
        for pass_name, r in self.passes.items():
            sub = r.to_nameddict()  # should have outer-most 'ProtocolName' dict
            assert(sub.keyname == 'ProtocolName' and len(sub) == 1)
            pname = r.protocol.name  # also list(sub.keys())[0]
            if pname not in ret:
                ret[pname] = _NamedDict('Pass', 'category')
            ret[pname][pass_name] = sub[pname]

        return ret

    def copy(self):
        """
        Make a copy of this object.

        Returns
        -------
        MultiPassResults
        """
        cpy = MultiPassResults(self.data.copy(), _copy.deepcopy(self.protocol))
        for k, v in self.passes.items():
            cpy.passes[k] = v.copy()
        return cpy


class ProtocolResultsDir(_TreeNode):
    """
    Holds a dictionary of :class:`ProtocolResults` objects.

    It contains a :class:`ProtocolData` object and is rooted at the_model
    corresponding node of the data-tree.  It contains links to
    child-:class:`ProtocolResultsDir` objects representing sub-directories.

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
    """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None, preloaded_data=None, quick_load=False):
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

        preloaded_data : ProtocolData, optional
            In the case that the :class:`ProtocolData` object for `dirname`
            is already loaded, it can be passed in here.  Otherwise leave this
            as None and it will be loaded.

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
            (preloaded_data if preloaded_data is not None else
             _io.read_data_from_dir(dirname, quick_load=quick_load))

        #Load results in results_dir
        results = {}
        results_dir = dirname / 'results'
        if results_dir.is_dir():  # if results_dir doesn't exist that's ok (just no results to load)
            for pth in results_dir.iterdir():
                if pth.is_dir() and (pth / 'meta.json').is_file():
                    results[pth.name] = _io.metadir._cls_from_meta_json(pth).from_dir(
                        dirname, pth.name, preloaded_data=data, quick_load=quick_load)

        ret = cls(data, results, {})  # don't initialize children now
        ret._init_children(dirname, meta_subdir='results', quick_load=quick_load)
        return ret

    @classmethod
    def from_mongodb(cls, mongodb, doc_id, parent=None, name=None, preloaded_data=None, quick_load=False,
                     custom_collection_names=None):
        """
        Initialize a new :class:`ProtocolResultsDir` object from a Mongo database.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to load data from.

        doc_id : str
            The user-defined identifier of the protocol results directory to load.

        parent : ProtocolData, optional
            The parent data object, if there is one.  This is needed for
            sub-results objects which reference/inherit their parent's dataset.
            Primarily used internally - if in doubt, leave this as `None`.

        name : str, optional
            The sub-name of the object being loaded, i.e. the
            key of this data object beneath `parent`.  Only used when
            `parent` is not None.

        preloaded_data : ProtocolData, optional
            In the case that the :class:`ProtocolData` object for `doc_id`
            is already loaded, it can be passed in here.  Otherwise leave this
            as None and it will be loaded.

        quick_load : bool, optional
            Setting this to True skips the loading of components that may take
            a long time, e.g. the actual raw data set(s). This can be useful
            when loading takes a long time and all the information of interest
            lies elsewhere, e.g. in an encompassing results object.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects.  Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        Returns
        -------
        ProtocolResultsDir
        """
        data = parent.data[name] if (parent and name) else \
            (preloaded_data if preloaded_data is not None else
             _io.read_data_from_mongodb(mongodb, doc_id, quick_load=quick_load, comm=None,
                                        custom_collection_names=custom_collection_names))

        collection_names = _io.mongodb_collection_names(custom_collection_names)

        #Load results with directory_id == doc_id
        results = {}
        for res_doc in mongodb[collection_names['results']].find({'directory_id': doc_id}, ['name', 'type']):
            results[res_doc['name']] = _io.metadir._class_for_name(res_doc['type']).from_mongodb(
                mongodb, doc_id, res_doc['name'], preloaded_data=data, quick_load=quick_load,
                custom_collection_names=custom_collection_names)

        ret = cls(data, results, {})  # don't initialize children now
        ret._init_children_from_mongodb(mongodb, None, doc_id, custom_collection_names, quick_load=quick_load)
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

        super().__init__(self.data.edesign._dirs, children)

    def _create_childval(self, key):  # (this is how children are created on-demand)
        """ Create the value for `key` on demand. """
        if self.data.edesign._loaded_from and isinstance(self.data.edesign._loaded_from, str) \
           and key in self._dirs:
            dirname = _pathlib.Path(self.data.edesign._loaded_from)
            subdir = self._dirs[key]
            subobj_dir = dirname / subdir

            if subobj_dir.exists():
                submeta_dir = subobj_dir / 'results'
                if submeta_dir.exists() and (submeta_dir / 'meta.json').exists():
                    # then use this metadata to determine the results-dir object type
                    classobj = _io.metadir._cls_from_meta_json(submeta_dir)
                else:
                    # otherwise just make the sub-resultsdir object the same type as this one
                    classobj = self.__class__
                return classobj.from_dir(subobj_dir, parent=self, name=key, preloaded_data=self.data[key])
            else:
                raise ValueError("Expected directory: '%s' doesn't exist!" % str(subobj_dir))
        else:
            raise KeyError("Invalid key: %s" % str(key))

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
            dirname = self.data.edesign._loaded_from if isinstance(self.data.edesign._loaded_from, str) else None
            if dirname is None: raise ValueError("`dirname` must be given because there's no default directory")

        if parent is None: self.data.write(dirname)  # assume parent has already written data
        dirname = _pathlib.Path(dirname)

        results_dir = dirname / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        _io.metadir._obj_to_meta_json(self, results_dir)

        #write the results
        for name, results in self.for_protocol.items():
            assert(results.name == name)
            results.write(dirname, data_already_written=True)

        self._write_children(dirname, write_subdir_json=False)  # writes sub-nodes

    def write_to_mongodb(self, mongodb, doc_id, parent=None, custom_collection_names=None, session=None):
        """
        Write this protocol results directory to a MongoDB database.

        Parameters
        ----------
        mongodb : pymongo.database.Database, optional
            The MongoDB instance to write data to.  Can be left as `None` if this
            results directory was initialized from a MongoDB, e.g. with
            :function:`pygsti.io.read_results_from_mongodb`, in which case the same
            database used for the initial read-in is used.

        doc_id : str, optional
            The user-defined identifier of the protocol results directory to write.  Can be
            left as `None` if this results directory was initialized from a MongoDB,
            in which case the same document identifier that was used at read-in is used.

        parent : ProtocolResultsDir, optional
            The parent protocol results directory, when a parent is writing this
            data as a sub-protocol-data object.  Otherwise leave as None.

        custom_collection_names : dict, optional
            Overrides for the default MongoDB collection names used for storing different types of
            pyGSTi objects.  In this case, the `"edesigns"` and `"data"` keys of this dictionary
            are relevant.  Default values are given by :method:`pygsti.io.mongodb_collection_names`.

        session : pymongo.client_session.ClientSession, optional
            MongoDB session object to use when interacting with the MongoDB
            database. This can be used to implement transactions
            among other things.

        Returns
        -------
        None
        """
        loaded_from = self.data.edesign._loaded_from if isinstance(self.data.edesign._loaded_from, tuple) \
            else None, None, None

        if mongodb is None:
            mongodb = loaded_from[0]
            if mongodb is None: raise ValueError("`mongodb` must be given because there's no default!")

        if doc_id is None:
            doc_id = loaded_from[1]
            if doc_id is None: raise ValueError("`doc_id` must be given because there's no default!")

        if custom_collection_names is None and loaded_from[2] is not None and doc_id is None:
            custom_collection_names = loaded_from[2]  # override if inferring document id

        if parent is None:
            self.data.write_to_mongodb(mongodb, doc_id, None, custom_collection_names, session)

        ##Write our class information to mongodb, even though we don't currently use this when loading
        #_io.write_obj_to_mongodb_auxtree(self, mongodb[collection_names['resultdirs']], doc_id,
        #                         auxfile_types_member=None,
        #                         omit_attributes=[...],
        #                         session=session, overwrite_existing=overwrite_existing)

        #write the results
        for name, results in self.for_protocol.items():
            assert(results.name == name)
            results.write_to_mongodb(mongodb, doc_id, data_already_written=True,
                                     custom_collection_names=custom_collection_names, session=session)
        self._write_children_to_mongodb(mongodb, doc_id, update_children_in_edesign=False,
                                        custom_collection_names=custom_collection_names,
                                        session=session)  # writes sub-resultdirs

    @classmethod
    def remove_from_mongodb(cls, mongodb, doc_id, custom_collection_names=None, session=None):
        """
        Remove a protocol results directory from a MongoDB database.

        Returns
        -------
        bool
            `True` if the specified data was removed, `False` if it didn't exist.
        """
        collection_names = _io.mongodb_collection_names(custom_collection_names)
        cls._remove_children_from_mongodb(mongodb, None, doc_id, custom_collection_names, session)

        delcnt = 0
        for res_doc in mongodb[collection_names['results']].find(
                {'directory_id': doc_id}, ['name']):
            if _io.remove_results_from_mongodb(mongodb, doc_id, res_doc['name'], comm=None,
                                               custom_collection_names=custom_collection_names, session=session):
                delcnt += 1

        # Perhaps parent has already done this, but try to remove data anyway
        _io.remove_data_from_mongodb(mongodb, doc_id, custom_collection_names, session)

        #FUTURE: if we use resultdirs:
        #delcnt = _io.remove_auxtree_from_mongodb(mongodb[collection_names['resultdirs']], doc_id, 'auxfile_types',
        #                                         session=session)
        return bool(delcnt >= 1)

    def _result_namedicts_on_this_node(self):
        nds = [v.to_nameddict() for v in self.for_protocol.values()]
        if len(nds) > 0:
            assert(all([nd.keyname == nds[0].keyname for nd in nds])), \
                "All protocols on a given node must return a NamedDict with the *same* root name!"  # eg "ProtocolName"
            results_on_this_node = nds[0]
            for nd in nds[1:]:
                results_on_this_node.update(nd)
        else:
            results_on_this_node = None
        return results_on_this_node

    def _addto_bypath_nameddict(self, dest, path):
        results_on_this_node = self._result_namedicts_on_this_node()
        if results_on_this_node is not None:
            dest[path] = results_on_this_node
        for k, v in self.items():
            v._addto_bypath_nameddict(dest, path + (k,))

    def to_nameddict(self):
        """
        Convert the results in this object into nested :class:`NamedDict` objects.

        Returns
        -------
        NamedDict
        """
        nd = _NamedDict('Path', 'object')  # so it can hold tuples of tuples, etc.
        self._addto_bypath_nameddict(nd, path=())
        return nd

    def to_dataframe(self, pivot_valuename=None, pivot_value=None, drop_columns=False):
        """
        Convert these results into Pandas dataframe.

        Parameters
        ----------
        pivot_valuename : str, optional
            If not None, the resulting dataframe is pivoted using `pivot_valuename`
            as the column whose values name the pivoted table's column names.
            If None and `pivot_value` is not None,`"ValueName"` is used.

        pivot_value : str, optional
            If not None, the resulting dataframe is pivoted such that values of
            the `pivot_value` column are rearranged into new columns whose names
            are given by the values of the `pivot_valuename` column. If None and
            `pivot_valuename` is not None,`"Value"` is used.

        drop_columns : bool or list, optional
            A list of column names to drop (prior to performing any pivot).  If
            `True` appears in this list or is given directly, then all
            constant-valued columns are dropped as well.  No columns are dropped
            when `drop_columns == False`.

        Returns
        -------
        DataFrame
        """
        df = self.to_nameddict().to_dataframe()
        return _process_dataframe(df, pivot_valuename, pivot_value, drop_columns)

    def __str__(self):
        import pprint
        P = pprint.PrettyPrinter()
        return P.pformat(self.to_nameddict())


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
    Similar to a protocol, but runs on an *existing* :class:`ProtocolResults` object.

    Running a :class:`ProtocolPostProcessor` produces a new (or updated)
    :class:`ProtocolResults` object.

    Parameters
    ----------
    name : str
        The name of this post-processor.
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


class DataSimulator(object):
    """
    An analysis routine that is run on an experiment design to produce per-circuit data.

    A DataSimulator fundamentally simulates a model to create data, taking an :class:`ExperimentDesign`
    as input and producing a :class:`ProtocolData` object as output.

    The produced data may consist of data counts for some/all of the circuit outcomes, and
    thus result in a :class:`ProtocolData` containsing a normal :class:`DataSet`.  Alternatively,
    a data simulator may compute arbitrary quantities to be associated with the circuits, resulting
    in a :class:`ProtocolData` containsing a normal :class:`FreeformDataSet`.

    """

    def __init__(self):
        pass

    def run(self, edesign, memlimit=None, comm=None):
        """
        Run this data simulator on an experiment design.

        Parameters
        ----------
        edesign : ExperimentDesign
            The input experiment design.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this data
            simulator in parallel.

        Returns
        -------
        ProtocolData
        """
        raise NotImplementedError("Derived classes should implement this!")


class DataCountsSimulator(DataSimulator):
    """
    Simulates data counts for each circuit outcome, producing a simulated data set.

    This object can also be used to compute the outcome probabilities for each
    circuit outcome instead of sampled counts by setting `sample_error="none"`.

    Parameters
    ----------
    model : Model
        The model to simulate.

    num_samples : int or list of ints or None, optional
        The simulated number of samples for each circuit.  This only has
        effect when  ``sample_error == "binomial"`` or ``"multinomial"``.  If an
        integer, all circuits have this number of total samples. If a list,
        integer elements specify the number of samples for the corresponding
        circuit.  If ``None``, then `model_or_dataset` must be a
        :class:`~pygsti.objects.DataSet`, and total counts are taken from it
        (on a per-circuit basis).

    sample_error : string, optional
        What type of sample error is included in the counts.  Can be:

        - "none"  - no sample error: counts are floating point numbers such
          that the exact probabilty can be found by the ratio of count / total.
        - "clip" - no sample error, but clip probabilities to [0,1] so, e.g.,
          counts are always positive.
        - "round" - same as "clip", except counts are rounded to the nearest
          integer.
        - "binomial" - the number of counts is taken from a binomial
          distribution.  Distribution has parameters p = (clipped) probability
          of the circuit and n = number of samples.  This can only be used
          when there are exactly two SPAM labels in model_or_dataset.
        - "multinomial" - counts are taken from a multinomial distribution.
          Distribution has parameters p_k = (clipped) probability of the gate
          string using the k-th SPAM label and n = number of samples.

    seed : int, optional
        If not ``None``, a seed for numpy's random number generator, which
        is used to sample from the binomial or multinomial distribution.

    rand_state : numpy.random.RandomState
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.

    alias_dict : dict, optional
        A dictionary mapping single operation labels into tuples of one or more
        other operation labels which translate the given circuits before values
        are computed using `model_or_dataset`.  The resulting Dataset, however,
        contains the *un-translated* circuits as keys.

    collision_action : {"aggregate", "keepseparate"}
        Determines how duplicate circuits are handled by the resulting
        `DataSet`.  Please see the constructor documentation for `DataSet`.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for
        potentially registering new outcome labels.

    times : iterable, optional
        When not None, a list of time-stamps at which data should be sampled.
        `num_samples` samples will be simulated at each time value, meaning that
        each circuit in `circuit_list` will be evaluated with the given time
        value as its *start time*.
    """

    def __init__(self, model, num_samples=1000, sample_error='multinomial',
                 seed=None, rand_state=None, alias_dict=None,
                 collision_action="aggregate", record_zero_counts=True, times=None):
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.sample_error = sample_error
        self.seed = seed
        self.rand_state = rand_state
        self.alias_dict = alias_dict
        self.collision_action = collision_action
        self.record_zero_counts = record_zero_counts
        self.times = times

    def run(self, edesign, memlimit=None, comm=None):
        """
        Run this data simulator on an experiment design.

        Parameters
        ----------
        edesign : ExperimentDesign
            The input experiment design.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this data
            simulator in parallel.

        Returns
        -------
        ProtocolData
        """
        from pygsti.data.datasetconstruction import simulate_data as _simulate_data
        ds = _simulate_data(self.model, edesign.all_circuits_needing_data, self.num_samples,
                            self.sample_error, self.seed, self.rand_state,
                            self.alias_dict, self.collision_action,
                            self.record_zero_counts, comm, memlimit, self.times)
        return ProtocolData(edesign, ds)


#In the future, we could put this function into a base class for
# the classes that utilize it above, so it would become a proper method.
def _convert_nameddict_attributes(obj):
    """
    A helper function that converts the elements of the "_nameddict_attributes"
    attribute of several classes to the (key, value, type) array expected by
    :method:`NamedDict.create_nested`.
    """
    keys_vals_types = []
    for tup in obj._nameddict_attributes:
        if len(tup) == 1: attr, key, typ = tup[0], tup[0], None
        elif len(tup) == 2: attr, key, typ = tup[0], tup[1], None
        elif len(tup) == 3: attr, key, typ = tup
        keys_vals_types.append((key, getattr(obj, attr), typ))
    return keys_vals_types

""" Supporting objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import json as _json
import pickle as _pickle
import pathlib as _pathlib
import importlib as _importlib
import copy as _copy

from .. import io as _io
from .. import objects as _objs

#TODO: move these to other places in the repo: tools or io?


#Class-name utils...
def full_class_name(x):
    return x.__class__.__module__ + '.' + x.__class__.__name__


def class_for_name(module_and_class_name):
    parts = module_and_class_name.split('.')
    module_name = '.'.join(parts[0:-1])
    class_name = parts[-1]
    # load the module, will raise ImportError if module cannot be loaded
    m = _importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


# ****************** Serialization into a directory with a meta.json *********************
def _get_auxfile_ext(typ):
    #get expected extension
    if typ == 'text-circuit-list': ext = '.txt'
    elif typ == 'text-circuit-lists': ext = '.txt'
    elif typ == 'list-of-protocolobjs': ext = ''
    elif typ == 'dict-of-protocolobjs': ext = ''
    elif typ == 'protocolobj': ext = ''
    elif typ == 'json': ext = '.json'
    elif typ == 'pickle': ext = '.pkl'
    elif typ == 'none': ext = '.NA'
    elif typ == 'reset': ext = '.NA'
    else: raise ValueError("Invalid aux-file type: %s" % typ)
    return ext


def load_meta_based_dir(root_dir, auxfile_types_member='auxfile_types',
                        ignore_meta=('type',), separate_auxfiletypes=False):
    
    root_dir = _pathlib.Path(root_dir)
    ret = {}
    with open(root_dir / 'meta.json', 'r') as f:
        meta = _json.load(f)

    for key, val in meta.items():
        if key in ignore_meta: continue
        ret[key] = val

    for key, typ in meta[auxfile_types_member].items():
        ext = _get_auxfile_ext(typ)

        #Process cases with non-standard expected path(s)
        if typ == 'none':  # member is serialized separatey and shouldn't be touched
            continue
        elif typ == 'reset':  # 'reset' doesn't write and loads in as None
            val = None  # no file exists for this member
            
        elif typ == 'text-circuit-lists':
            i = 0; val = []
            while True:
                pth = root_dir / (key + str(i) + ext)
                if not pth.exists(): break
                val.append(_io.load_circuit_list(pth)); i += 1

        elif typ == 'protocolobj':
            protocol_dir = root_dir / (key + ext)
            val = obj_from_meta_json(protocol_dir).from_dir(protocol_dir)

        elif typ == 'list-of-protocolobjs':
            i = 0; val = []
            while True:
                pth = root_dir / (key + str(i) + ext)
                if not pth.exists(): break
                val.append(obj_from_meta_json(pth).from_dir(pth)); i += 1

        elif typ == 'dict-of-protocolobjs':
            keys = meta[key]; paths = [root_dir / (key + "_" + k + ext) for k in keys]
            val = {k: obj_from_meta_json(pth).from_dir(pth) for k, pth in zip(keys, paths)}
            
        else:  # cases with 'standard' expected path

            pth = root_dir / (key + ext)
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

        ret[key] = val

    if separate_auxfiletypes:
        del ret[auxfile_types_member]
        return ret, meta[auxfile_types_member]
    else:
        return ret


def write_meta_based_dir(root_dir, valuedict, auxfile_types=None, init_meta=None):
    root_dir = _pathlib.Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if auxfile_types is None:
        auxfile_types = valuedict['auxfile_types']

    meta = {}
    if init_meta: meta.update(init_meta)
    meta['auxfile_types'] = auxfile_types

    for key, val in valuedict.items():
        if key in auxfile_types: continue  # member is serialized to a separate file (see below)
        if isinstance(val, _objs.VerbosityPrinter): val = val.verbosity  # HACK!!
        meta[key] = val
    #Wait to write meta.json until the end, since
    # aux-types may utilize it too.

    for auxnm, typ in auxfile_types.items():
        try:
            val = valuedict[auxnm]
        except:
            import bpdb; bpdb.set_trace()
            pass
        ext = _get_auxfile_ext(typ)

        if typ == 'text-circuit-lists':
            for i, circuit_list in enumerate(val):
                pth = root_dir / (auxnm + str(i) + ext)
                _io.write_circuit_list(pth, circuit_list)

        elif typ == 'protocolobj':
            val.write(root_dir / (auxnm + ext))

        elif typ == 'list-of-protocolobjs':
            for i, obj in enumerate(val):
                pth = root_dir / (auxnm + str(i) + ext)
                obj.write(pth)

        elif typ == 'dict-of-protocolobjs':
            meta[auxnm] = list(val.keys())  # just save a list of the keys in the metadata
            for k, obj in val.items():
                obj_dirname = auxnm + "_" + k + ext  # keys must be strings
                obj.write(root_dir / obj_dirname)

        else:
            # standard path cases
            pth = root_dir / (auxnm + ext)

            if typ == 'text-circuit-list':
                _io.write_circuit_list(pth, val)
            elif typ == 'json':
                with open(pth, 'w') as f:
                    _json.dump(val, f)
            elif typ == 'pickle':
                with open(pth, 'wb') as f:
                    _pickle.dump(val, f)
            elif typ in ('none', 'reset'):
                pass
            else:
                raise ValueError("Invalid aux-file type: %s" % typ)

    with open(root_dir / 'meta.json', 'w') as f:
        try:
            _json.dump(meta, f)
        except:
            import bpdb; bpdb.set_trace()
            pass


def obj_from_meta_json(dirname): #, subdir=None, parent=None, name=None):
    with open(_pathlib.Path(dirname) / 'meta.json', 'r') as f:
        meta = _json.load(f)
    return class_for_name(meta['type'])  # class of object to create


def write_obj_to_meta_based_dir(obj, dirname, auxfile_types_member):
    meta = {'type': full_class_name(obj)}
    write_meta_based_dir(dirname, obj.__dict__,
                         obj.__dict__[auxfile_types_member], init_meta=meta)


def read_json_or_pkl_files_to_dict(dirname):
    dirname = _pathlib.Path(dirname)
    if not dirname.is_dir():
        return {}

    ret = {}
    for pth in dirname.iterdir():
        if pth.suffix == '.json':
            with open(pth, 'r') as f:
                val = _json.load(f)
        elif pth.suffix == '.pkl':
            with open(pth, 'r') as f:
                val = _pickle.load(f)
        else:
            continue  # ignore cache file times we don't understand
        ret[pth.name] = val
    return ret


def write_dict_to_json_or_pkl_files(d, dirname):
    dirname = _pathlib.Path(dirname)
    dirname.mkdir(exist_ok=True)
    for key, val in d.items():
        try:
            with open(dirname / (key + '.json'), 'w') as f:
                _json.dump(val, f)
        except:
            #try to remove partial json file??
            with open(dirname / (key + '.pkl'), 'wb') as f:
                _pickle.dump(val, f)

#    with open(data_dir / 'meta.json', 'r') as f:
#        meta = _json.load(f)
#    dtype = meta['dataset_type']
#
#    if dtype == 'normal':
#        dataset = _io.load_dataset(data_dir / 'dataset.txt')
#    elif dtype == 'multi-single-file':
#        dataset = _io.load_multidataset(data_dir / 'dataset.txt')
#    elif dtype == 'multi-multiple-files':
#        raise NotImplementedError()  # should load in several datasetX.txt files as elements of a MultiDataSet
#    elif dtype == 'same-as-parent':
#        if parent is None:
#            parent = ProtocolData.from_dir(dirname / '..') #, load_subdatas=False)
#        dataset = parent.dataset
#    else:
#        raise ValueError("Invalid dataset type: %s" % dtype)


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


#class DirSerializable(object):
#
#    @classmethod
#    def from_dir(cls, dirname):
#        p = _pathlib.Path(dirname)
#        if cls.serialization_subdir:
#            obj_dir = p / cls.serialization_subdir
#        else:
#            obj_dir = p
#        ret = cls.__new__(cls)
#        ret.__dict__.update(load_meta_based_dir(obj_dir))
#        return ret
#
#    def __init__(self):
#        self.auxfile_types = {}
#
#    def write(self, dirname):
#        p = _pathlib.Path(dirname)
#        p.mkdir(parents=True, exist_ok=True)
#        if self.serialization_subdir:
#            obj_dir = p / self.serialization_subdir
#        else:
#            obj_dir = p
#
#        meta = {'type': full_class_name(self)}
#        write_meta_based_dir(obj_dir, self.__dict__, init_meta=meta)
#
#
#class KeyAccess(object):
#    def __init__(self, keysrc):
#        self._keysrc = keysrc
#        if isinstance(self, DirSerializable):
#            self.auxfile_types['_keysrc'] = 'none'
#
#    def keys(self):
#        return self._keysrc.keys()
#
#    def __contains__(self, key):
#        return key in self._keysrc
#
#    def __len__(self):
#        return len(self._keysrc)
#
#
#class DictAccess(KeyAccess):
#    def __init__(self, keyvalsrc):
#        super().__init__(keyvalsrc)
#
#    def items(self):
#        return self._keysrc.items()
#
#    def __getitem__(self, key):
#        return self._keysrc[key]
#
#
#class LazyDictAccess(KeyAccess):
#    def __init__(self, keysrc, valsrc):
#        super().__init__(keysrc)
#        self._valsrc = valsrc
#        if isinstance(self, DirSerializable):
#            self.auxfile_types['_valsrc'] = 'none'
#
#    def items(self):
#        for k in self._keysrc.keys():
#            yield k, self[k]
#
#    def __getitem__(self, key):
#        if key not in self._valsrc:
#            if key not in self._keysrc.keys():
#                raise KeyError("Invalid key: %s" % key)
#            self._valsrc[key] = self._lazy_create_dictval(key)
#        return self._valsrc[key]

class TreeNode(object):
    
    @classmethod
    def from_dir(cls, dirname, parent=None, name=None):
        raise NotImplementedError("Derived classes should implement from_dir(...)!")
        # *** This is a template function, showing how derived classes should implement from_dir ***
        # ret = cls.__new__(cls)  # create the object to return
        # # <perform derived class initialization, so that the 'parent' passed
        # #  to child nodes' from_dir(...) is as initialized as possible >
        # subdir_between_dirname_and_meta_json = None  # typically this identifies a particular derived class
        # ret._init_children(dirname, subdir_between_dirname_and_meta_json)  # loads child nodes
        # return ret
    
    def __init__(self, possible_child_name_dirs, child_values=None):
        self._dirs = possible_child_name_dirs  # maps possible child keys -> subdir name
        self._vals = child_values if child_values else {}

    def _init_children(self, dirname, meta_subdir=None):
        dirname = _pathlib.Path(dirname)
        input_dir = dirname / 'input'  # because subdirs.json is always & only in 'input'
        with open(input_dir / 'subdirs.json', 'r') as f:
            meta = _json.load(f)

        child_dirs = {}
        for d, nm in meta.get('directories', {}).items():
            if isinstance(nm, list): nm = tuple(nm)  # because json makes tuples->lists
            child_dirs[nm] = d

        self._dirs = child_dirs
        self._vals = {}

        for nm, subdir in child_dirs.items():
            subobj_dir = dirname / subdir
            if meta_subdir:
                submeta_dir = subobj_dir / meta_subdir
                if submeta_dir.exists():  # It's ok if not all possible sub-nodes exist
                    self._vals[nm] = obj_from_meta_json(submeta_dir).from_dir(subobj_dir, parent=self, name=nm)
            else:  # if meta_subdir is None, we default to the same class as self (ProtocolResultsDir case)
                self._vals[nm] = self.__class__.from_dir(subobj_dir, parent=self, name=nm)
            

    def keys(self):
        return self._dirs.keys()

    def __contains__(self, key):
        return key in self._dirs

    def __len__(self):
        return len(self._dirs)

    def items(self):
        for k in self._dirs:
            yield k, self[k]

    def __getitem__(self, key):
        if key not in self._dirs:
            raise KeyError("Invalid key: %s" % key)
        if key not in self._vals:
            self._vals[key] = self._create_childval(key)
        return self._vals[key]

    def _create_childval(self, key):
        raise NotImplementedError("Derived class needs to implement _create_childval to create valid key: %s" % key)
    
    def get_tree_paths(self):
        """Dictionary paths leading to input objects/nodes beneath this one"""
        paths = [()]  # path to self
        for child_name, child_node in self.items():
            paths.extend([(child_name,) + pth for pth in child_node.get_tree_paths()])
        return paths

    def view(self, keys_to_keep):
        if len(keys_to_keep) == 0: return self
        view = _copy.deepcopy(self)  # is deep copy really needed here??
        view._dirs = {k: self._dirs[k] for k in keys_to_keep}
        view._vals = {k: self[k] for k in keys_to_keep}
        return view

    def filter_paths(self, paths, paths_are_sorted=False):
        sorted_paths = paths if paths_are_sorted else sorted(paths)
        nPaths = len(sorted_paths)

        if nPaths == 1 and len(sorted_paths[0]) == 0:
            return self  # special case when this MultiInput itself is selected

        i = 0
        children_to_keep = {}
        while i < nPaths:
            assert(len(sorted_paths[i]) > 0), \
                "Cannot select a MultiInput *and* some/all of its elements using filter_paths!"
            ky = sorted_paths[i][0]

            paths_starting_with_ky = []
            while i < nPaths and sorted_paths[i][0] == ky:
                paths_starting_with_ky.append(sorted_paths[i][1:])
                i += 1
            children_to_keep[ky] = self[ky].filter_paths(paths_starting_with_ky, True)

        #assert(len(children_to_keep) > 0)
        view = _copy.deepcopy(self)  # copies type of multi-input
        view._dirs = {k: self._dirs[k] for k in children_to_keep}
        view._vals = children_to_keep
        return view

    def write(self, dirname, parent=None):
        raise NotImplementedError("Derived classes should implement write(...)!")
        # *** This is a template showing how derived classes should implement write(...) ***
        # # <write derived-class specific data to dirname>
        # #  write_subdir_json = True  # True only for the "master" type that defines the directory keys ('input')
        # self.write_children(dirname, write_subdir_json)

    def write_children(self, dirname, write_subdir_json=True):
        dirname = _pathlib.Path(dirname)

        if write_subdir_json:
            subdirs = {}
            subdirs['directories'] = {dirname: subname for subname, dirname in self._dirs.items()}
            # write self._dirs "backwards" b/c json doesn't allow tuple-like keys (sometimes keys are tuples)
            with open(dirname / 'input' / 'subdirs.json', 'w') as f:
                _json.dump(subdirs, f)

        for nm, val in self._vals.items():  # only write *existing* values
            subdir = self._dirs[nm]
            outdir = dirname / subdir
            outdir.mkdir(exist_ok=True)
            self._vals[nm].write(outdir, parent=self)

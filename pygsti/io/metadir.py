"""
Serialization routines to/from a meta.json based directory
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.sparse as _sps
import importlib as _importlib
import json as _json
import pathlib as _pathlib
import pickle as _pickle
import warnings as _warnings

from pygsti.io import loaders as _load
from pygsti.io import writers as _write
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter

QUICK_LOAD_MAX_SIZE = 10 * 1024  # 10 kilobytes


#Class-name utils...
def _full_class_name(x):
    """
    Returns the <module>.<classname> for `x`.

    Parameters
    ----------
    x : class
        The class whose full name you want.

    Returns
    -------
    str
    """
    return x.__class__.__module__ + '.' + x.__class__.__name__


def _class_for_name(module_and_class_name):
    """
    Return the class object given an name.

    Parameters
    ----------
    module_and_class_name : strictly
        The module and class name, e.g. "mymodule.MyClass"

    Returns
    -------
    class
    """
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
    #elif typ == 'text-circuit-lists': ext = '.txt'
    #elif typ == 'list-of-protocolobjs': ext = ''
    #elif typ == 'dict-of-protocolobjs': ext = ''
    #elif typ == 'dict-of-resultsobjs': ext = ''
    elif typ == 'dir-serialized-object': ext = ''  # a directory
    elif typ == 'partialdir-serialized-object': ext = ''  # a directory
    elif typ == 'serialized-object': ext = '.json'
    elif typ == 'circuit-str-json': ext = '.json'
    elif typ == 'numpy-array': ext = '.npy'
    elif typ == 'json': ext = '.json'
    elif typ == 'pickle': ext = '.pkl'
    elif typ == 'none': ext = '.NA'
    elif typ == 'reset': ext = '.NA'
    else: raise ValueError("Invalid aux-file type: %s" % typ)
    return ext


def load_meta_based_dir(root_dir, auxfile_types_member='auxfile_types',
                        ignore_meta=('type',), separate_auxfiletypes=False,
                        quick_load=False):
    """
    Load the contents of a `root_dir` into a dict.

    The de-serialization uses the 'meta.json' file within `root_dir` to describe
    how the directory was serialized.

    Parameters
    ----------
    root_dir : str
        The directory name.

    auxfile_types_member : str, optional
        The key within meta.json that is used to describe how other
        members have been serialized into files.  Unless you know what you're
        doing, leave this as the default.

    ignore_meta : tuple, optional
        Keys within meta.json that should be ignored, i.e. not loaded into
        elements of the returned dict.  By default, `"type"` is in this
        category because it describes a class name to be built and is used
        in a separate first-pass processing to construct a object.  Unless
        you know what you're doing, leave this as the default.

    separate_auxfiletypes : bool, optional
        If True, then return the `auxfile_types_member` element (a dict
        describing how quantities that aren't in 'meta.json' have been
        serialized) as a separate return value, instead of placing it
        within the returned dict.

    quick_load : bool, optional
        Setting this to True skips the loading of members that may take
        a long time to load, namely those in separate files whose files are
        large.  When the loading of an attribute is skipped, it is set to `None`.

    Returns
    -------
    loaded_qtys : dict
        A dictionary of the quantities in 'meta.json' plus any loaded
        from the auxiliary files.
    auxfile_types : dict
        Only returned as a separate value when `separate_auxfiletypes=True`.
        A dict describing how members of `loaded_qtys` that weren't loaded
        directly from 'meta.json' were serialized.
    """
    root_dir = _pathlib.Path(root_dir)
    ret = {}

    with open(str(root_dir / 'meta.json'), 'r') as f:
        meta = _json.load(f)

        #Convert lists => tuples, as we prefer immutable tuples
        #for key in meta:
        #    if type(meta[key]) == list:  # note: we don't want isinstance here - just simple *lists*
        #        meta[key] = tuple(meta[key])

    for key, val in meta.items():
        if key in ignore_meta: continue
        ret[key] = val

    for key, typ in meta[auxfile_types_member].items():
        if key in ignore_meta: continue  # don't load -> members items in ignore_meta

        bLoaded, val = _load_auxfile_member(root_dir, key, typ, meta.get(key, None), quick_load)
        if bLoaded:
            ret[key] = val
        elif val is True:  # val is value of whether to set value to None
            ret[key] = None  # Note: could just have bLoaded be True in this case and val=None

    if separate_auxfiletypes:
        del ret[auxfile_types_member]
        return ret, meta[auxfile_types_member]
    else:
        return ret


#REMOVE!!!
#                    pth = root_dir / (key + str(i) + ext)
#                    if not pth.exists(): break
#                    if should_skip_loading(pth):
#                        val.append(None)
#                    else:
#                        val.append(_load.load_circuit_list(pth))
#
#
#            
#        
#        for sub_typ in typ.split(':'):  # allows, e.g. "list:" or "dict:" prefixes
#            
#            
#            
#            
#        ext = _get_auxfile_ext(typ)
#
#        #Process cases with non-standard expected path(s)
#        if typ == 'none':  # member is serialized separatey and shouldn't be touched
#            continue
#        elif typ == 'reset':  # 'reset' doesn't write and loads in as None
#            val = None  # no file exists for this member
#
#        elif typ == 'text-circuit-lists':
#            i = 0; val = []
#            while True:
#                pth = root_dir / (key + str(i) + ext)
#                if not pth.exists(): break
#                if should_skip_loading(pth):
#                    val.append(None)
#                else:
#                    val.append(_load.load_circuit_list(pth))
#                i += 1
#
#        elif typ == 'protocolobj':
#            protocol_dir = root_dir / (key + ext)
#            val = _cls_from_meta_json(protocol_dir).from_dir(protocol_dir, quick_load=quick_load)
#
#        elif typ == 'list-of-protocolobjs':
#            i = 0; val = []
#            while True:
#                pth = root_dir / (key + str(i) + ext)
#                if not pth.exists(): break
#                val.append(_cls_from_meta_json(pth).from_dir(pth, quick_load=quick_load)); i += 1
#
#        elif typ == 'dict-of-protocolobjs':
#            keys = meta[key]; paths = [root_dir / (key + "_" + k + ext) for k in keys]
#            val = {k: _cls_from_meta_json(pth).from_dir(pth, quick_load=quick_load) for k, pth in zip(keys, paths)}
#
#        elif typ == 'dict-of-resultsobjs':
#            keys = meta[key]; paths = [root_dir / (key + "_" + k + ext) for k in keys]
#            val = {k: _cls_from_meta_json(pth)._from_dir_partial(pth, quick_load=quick_load)
#                   for k, pth in zip(keys, paths)}
#
#        else:  # cases with 'standard' expected path
#
#            pth = root_dir / (key + ext)
#            if pth.is_dir():
#                raise ValueError("Expected path: %s is a dir!" % pth)
#            elif not pth.exists() or should_skip_loading(pth):
#                val = None  # missing files => None values
#            elif typ == 'text-circuit-list':
#                val = _load.load_circuit_list(pth)
#            elif typ == 'json':
#                with open(str(pth), 'r') as f:
#                    val = _json.load(f)
#            elif typ == 'pickle':
#                with open(str(pth), 'rb') as f:
#                    val = _pickle.load(f)
#            else:
#                raise ValueError("Invalid aux-file type: %s" % typ)
#
#        ret[key] = val


def _load_auxfile_member(root_dir, filenm, typ, metadata, quick_load):
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    max_size = quick_load if isinstance(quick_load, int) else QUICK_LOAD_MAX_SIZE

    def should_skip_loading(path):
        return quick_load and (path.stat().st_size >= max_size)

    if cur_typ == 'list':
        if metadata is None:  # signals that value is None, otherwise would at least be an empty list
            val = None
        else:
            val = []
            for i, meta in enumerate(metadata):
                filenm_so_far = filenm + str(i)
                bLoaded, el = _load_auxfile_member(root_dir, filenm_so_far, next_typ, meta, quick_load)
                if bLoaded:
                    val.append(el)
                else:
                    break

    elif cur_typ == 'dict':
        if metadata is None:  # signals that value is None, otherwise would at least be an empty list
            val = None
        else:
            keys = list(metadata.keys())  # sort?
            val = {}
            for k in keys:
                filenm_so_far = filenm + "_" + k
                meta = metadata.get(k, None)
                bLoaded, v = _load_auxfile_member(root_dir, filenm_so_far, next_typ, meta, quick_load)
                if bLoaded:
                    val[k] = v
                else:
                    raise ValueError("Failed to load dictionary key " + str(k))

    elif cur_typ == 'fancykeydict':
        if metadata is None:  # signals that value is None, otherwise would at least be an empty list
            val = None
        else:
            keymeta_pairs = list(metadata)  # should be a list of (key, metadata_for_value) pairs
            val = {}
            for i, (k, meta) in enumerate(keymeta_pairs):
                filenm_so_far = filenm + "_kvpair" + str(i)
                bLoaded, el = _load_auxfile_member(root_dir, filenm_so_far, next_typ, meta, quick_load)
                if bLoaded:
                    val[k] = v
                else:
                    raise ValueError("Failed to load %d-th dictionary key: %s" % (i, str(k)))

    else:
        #Simple types that just load the given file
        ext = _get_auxfile_ext(cur_typ)
        pth = root_dir / (filenm + ext)
        if not pth.exists():
            return False, True  # failure to load, but not explicitly skipped, so set_to_None=True
            # Note: this behavior is needed when, for instance, a member that could be a
            # serializable object is None instead - in which case we just write nothing and load in None here.
        if cur_typ == 'none':  # member is serialized separatey and shouldn't be touched
            return False, False  # explicitly don't load or set value (so set_to_None=False)

        if should_skip_loading(pth):
            val = None  # load 'None' instead of actual data (skip loading this file)
        elif cur_typ == 'reset':  # 'reset' doesn't write and loads in as None
            val = None  # no file exists for this member
        elif cur_typ == 'text-circuit-list':
            val = _load.load_circuit_list(pth)
        elif cur_typ == 'dir-serialized-object':
            val = _cls_from_meta_json(pth).from_dir(pth, quick_load=quick_load)
        elif cur_typ == 'partialdir-serialized-object':
            val = _cls_from_meta_json(pth)._from_dir_partial(pth, quick_load=quick_load)
        elif cur_typ == 'serialized-object':
            val = _NicelySerializable.read(pth)
        elif cur_typ == 'circuit-str-json':
            val = _load.load_circuits_as_strs(pth)
        elif typ == 'numpy-array':
            val = _np.load(pth)
        elif typ == 'json':
            with open(str(pth), 'r') as f:
                val = _json.load(f)
        elif typ == 'pickle':
            with open(str(pth), 'rb') as f:
                val = _pickle.load(f)
        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return True, val  # loading successful - 2nd element is value loaded


def write_meta_based_dir(root_dir, valuedict, auxfile_types=None, init_meta=None):
    """
    Write a dictionary of quantities to a directory.

    Write the dictionary by placing everything in a 'meta.json' file except for
    special key/value pairs ("special" usually because they lend themselves to
    an non-JSON format or they simply cannot be rendered as JSON) which are placed
    in "auxiliary" files formatted according to `auxfile_types` (which itself is
    saved in meta.json).

    Parameters
    ----------
    root_dir : str
        The directory to write to (will be created if needed).

    valuedict : dict
        The dictionary of values to serialize to disk.

    auxfile_types : dict, optional
        A dictionary whose keys are a subset of the keys of `valuedict`,
        and whose values are known "aux-file" types.  `auxfile_types[key]`
        says that `valuedict[key]` should be serialized into a separate
        file (whose name is usually `key` + an appropriate extension) of
        the given format rather than be included directly in 'meta.json`.
        If None, this dictionary is assumed to be `valuedict['auxfile_types']`.

    init_meta : dict, optional
        A dictionary of "initial" meta-data to be included in the 'meta.json'
        (but that isn't in `valuedict`).  For example, the class name of an
        object is often stored as in the "type" field of meta.json when the_model
        objects .__dict__ is used as `valuedict`.

    Returns
    -------
    None
    """
    root_dir = _pathlib.Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    if auxfile_types is None:
        auxfile_types = valuedict['auxfile_types']

    meta = {}
    if init_meta: meta.update(init_meta)
    meta['auxfile_types'] = auxfile_types

    for key, val in valuedict.items():
        if key in auxfile_types: continue  # member is serialized to a separate file
        if isinstance(val, _VerbosityPrinter): val = val.verbosity  # HACK!!
        meta[key] = val
    #Wait to write meta.json until the end, since
    # aux-types may utilize it too.

    for auxnm, typ in auxfile_types.items():
        val = valuedict[auxnm]
        auxmeta = _write_auxfile_member(root_dir, auxnm, typ, val)
        if auxmeta is not None:
            meta[auxnm] = auxmeta  # metadata about auxfile(s) for this auxnm

    with open(str(root_dir / 'meta.json'), 'w') as f:
        _check_jsonable(meta)
        _json.dump(meta, f)

    #REMOVE
    #for auxnm, typ in auxfile_types.items():
    #    val = valuedict[auxnm]
    #    ext = _get_auxfile_ext(typ)
    #
    #    if typ == 'text-circuit-lists':
    #        for i, circuit_list in enumerate(val):
    #            pth = root_dir / (auxnm + str(i) + ext)
    #            _write.write_circuit_list(pth, circuit_list)
    #
    #    elif typ == 'protocolobj':
    #        val.write(root_dir / (auxnm + ext))
    #
    #    elif typ == 'list-of-protocolobjs':
    #        for i, obj in enumerate(val):
    #            pth = root_dir / (auxnm + str(i) + ext)
    #            obj.write(pth)
    #
    #    elif typ == 'dict-of-protocolobjs':
    #        meta[auxnm] = list(val.keys())  # just save a list of the keys in the metadata
    #        for k, obj in val.items():
    #            obj_dirname = auxnm + "_" + k + ext  # keys must be strings
    #            obj.write(root_dir / obj_dirname)
    #
    #    elif typ == 'dict-of-resultsobjs':
    #        meta[auxnm] = list(val.keys())  # just save a list of the keys in the metadata
    #        for k, obj in val.items():
    #            obj_dirname = auxnm + "_" + k + ext  # keys must be strings
    #            obj._write_partial(root_dir / obj_dirname)
    #
    #    else:
    #        # standard path cases
    #        pth = root_dir / (auxnm + ext)
    #
    #        if val is None:   # None values don't get written
    #            pass
    #        elif typ == 'text-circuit-list':
    #            _write.write_circuit_list(pth, val)
    #        elif typ == 'json':
    #            with open(str(pth), 'w') as f:
    #                _json.dump(val, f)
    #        elif typ == 'pickle':
    #            with open(str(pth), 'wb') as f:
    #                _pickle.dump(val, f)
    #        elif typ in ('none', 'reset'):
    #            pass
    #        else:
    #            raise ValueError("Invalid aux-file type: %s" % typ)


def _write_auxfile_member(root_dir, filenm, typ, val):
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    if cur_typ == 'list':
        if val is not None:
            metadata = []
            for i, el in enumerate(val):
                filenm_so_far = filenm + str(i)
                meta = _write_auxfile_member(root_dir, filenm_so_far, next_typ, el)
                metadata.append(meta)
        else:
            metadata = None

    elif cur_typ == 'dict':
        if val is not None:
            metadata = {}
            for k, v in val.items():
                filenm_so_far = filenm + "_" + k
                meta = _write_auxfile_member(root_dir, filenm_so_far, next_typ, v)
                metadata[k] = meta
        else:
            metadata = None

    elif cur_typ == 'fancykeydict':
        if val is not None:
            metadata = []
            for i, (k, v) in enumerate(val.items()):
                filenm_so_far = filenm + "_kvpair" + str(i)
                meta = _write_auxfile_member(root_dir, filenm_so_far, next_typ, v)
                metadata.append(k, meta)
        else:
            metadata = None

    else:
        #Simple types that just write the given file
        metadata = None
        ext = _get_auxfile_ext(cur_typ)
        pth = root_dir / (filenm + ext)

        if val is None:   # None values don't get written
            pass
        elif cur_typ in ('none', 'reset'):  # explicitly don't get written
            pass
        elif cur_typ == 'text-circuit-list':
            _write.write_circuit_list(pth, val)
        elif cur_typ == 'dir-serialized-object':
            val.write(pth)
        elif cur_typ == 'partialdir-serialized-object':
            val._write_partial(pth)
        elif cur_typ == 'serialized-object':
            assert(isinstance(val, _NicelySerializable)), \
                "Non-nicely-serializable '%s' object given for a 'serialized-object' auxfile type!" % (str(type(val)))
            val.write(pth)
        elif cur_typ == 'circuit-str-json':
            _write.write_circuits_as_strs(pth, val)
        elif cur_typ == 'numpy-array':
            _np.save(pth, val)
        elif typ == 'json':
            with open(str(pth), 'w') as f:
                _check_jsonable(val)
                _json.dump(val, f, indent=4)
        elif typ == 'pickle':
            with open(str(pth), 'wb') as f:
                _pickle.dump(val, f)
        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return metadata


def _cls_from_meta_json(dirname):
    """
    Get the object-type corresponding to the 'type' field in `dirname`/meta.json.

    Parameters
    ----------
    dirname : str
        the directory name.

    Returns
    -------
    class
    """
    with open(str(_pathlib.Path(dirname) / 'meta.json'), 'r') as f:
        meta = _json.load(f)
    return _class_for_name(meta['type'])  # class of object to create


def _obj_to_meta_json(obj, dirname):
    """
    Create a meta.json file within `dirname` that contains (only) the type of `obj` in its 'type' field.

    This is used to save an object that contains essentially no other data
    to a directory, in lieu of :function:`write_obj_to_meta_based_dir`.

    Parameters
    ----------
    obj : object
        he object whose type you want to save.

    dirname : str
        the directory name.

    Returns
    -------
    None
    """
    meta = {'type': _full_class_name(obj)}
    with open(str(_pathlib.Path(dirname) / 'meta.json'), 'w') as f:
        _check_jsonable(meta)
        _json.dump(meta, f)


def write_obj_to_meta_based_dir(obj, dirname, auxfile_types_member, omit_attributes=()):
    """
    Write the contents of `obj` to `dirname` using a 'meta.json' file and an auxfile-types dictionary.

    This is similar to :function:`write_meta_based_dir`, except it takes an object (`obj`)
    whose `.__dict__`, minus omitted attributes, is used as the dictionary to write and whose
    auxfile-types comes from another object attribute..

    Parameters
    ----------
    obj : object
        the object to serialize

    dirname : str
        the directory name

    auxfile_types_member : str
        the name of the attribute within `obj` that holds the dictionary
        mapping of attributes to auxiliary-file types.  Usually this is
        `"auxfile_types"`.

    omit_attributes : list or tuple
        List of (string-valued) names of attributes to omit when serializing
        this object.  Usually you should just leave this empty.

    Returns
    -------
    None
    """
    meta = {'type': _full_class_name(obj)}

    if len(omit_attributes) > 0:
        vals = obj.__dict__.copy()
        auxtypes = obj.__dict__[auxfile_types_member].copy()
        for o in omit_attributes:
            if o in vals: del vals[o]
            if o in auxtypes: del auxtypes[o]
    else:
        vals = obj.__dict__
        auxtypes = obj.__dict__[auxfile_types_member]

    write_meta_based_dir(dirname, vals, auxtypes, init_meta=meta)


def _read_json_or_pkl_files_to_dict(dirname):
    """
    Load any .json or .pkl files in `dirname` into a dict.

    Parameters
    ----------
    dirname : str
        the directory name.

    Returns
    -------
    dict
    """

    def _from_jsonable(x):
        if x is None or isinstance(x, (float, int, str)):
            return x
        elif isinstance(x, dict):
            if 'module' in x and 'class' in x:
                return _NicelySerializable.from_nice_serialization(x)
            else:  # assume a normal dictionary
                return {k: _from_jsonable(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [_from_jsonable(v) for v in x]
        else:
            raise ValueError("Cannot decode object of type '%s' within JSON'd values!" % str(type(x)))

    dirname = _pathlib.Path(dirname)
    if not dirname.is_dir():
        return {}

    ret = {}
    for pth in dirname.iterdir():
        if pth.suffix == '.json':
            with open(str(pth), 'r') as f:
                json_val = _json.load(f)
            val = _from_jsonable(json_val)
        elif pth.suffix == '.pkl':
            with open(str(pth), 'rb') as f:
                val = _pickle.load(f)
        else:
            continue  # ignore cache file times we don't understand
        ret[pth.stem] = val
    return ret


def write_dict_to_json_or_pkl_files(d, dirname):
    """
    Write each element of `d` into a separate file in `dirname`.

    If the element is json-able, it is JSON-serialized and the ".json"
    extension is used.  If not, pickle is used to serialize the element,
    and the ".pkl" extension is used.  This is the reverse of
    :function:`_read_json_or_pkl_files_to_dict`.

    Parameters
    ----------
    d : dict
        the dictionary of elements to serialize.

    dirname : str
        the directory name.

    Returns
    -------
    None
    """
    dirname = _pathlib.Path(dirname)
    dirname.mkdir(exist_ok=True)

    def _to_jsonable(val):
        if isinstance(val, _NicelySerializable):
            return val.to_nice_serialization()
        elif type(val) == list:  # don't use isinstance here
            return [_to_jsonable(v) for v in val]
        elif type(val) == dict:  # don't use isinstance here
            return {k: _to_jsonable(v) for k, v in val.items()}
        else:
            return val

    for key, val in d.items():
        try:
            jsonable = _to_jsonable(val)
            _check_jsonable(jsonable)
            with open(dirname / (key + '.json'), 'w') as f:
                _json.dump(jsonable, f)
        except Exception as e:
            fn = str(dirname / (key + '.json'))
            _warnings.warn("Could not write %s (falling back on pickle format):\n" % fn + str(e))
            #try to remove partial json file??
            with open(str(dirname / (key + '.pkl')), 'wb') as f:
                _pickle.dump(val, f)


def _check_jsonable(x):
    """ Checks that `x` can be properly converted to JSON, detecting
        errors that the json modules doesn't pick up. E.g. ensures that `x`
        doesn't contain dicts with non-string-valued keys """
    if x is None or isinstance(x, (float, int, str)):
        pass  # no problem
    elif isinstance(x, (tuple, list)):
        for i, v in enumerate(x):
            try:
                _check_jsonable(v)  # no problem as long as we don't mind tuples -> lists
            except ValueError as e:
                raise ValueError(("%d-th element : " % i) + str(e))
    elif isinstance(x, dict):
        if any([(not isinstance(k, str)) for k in x.keys()]):
            nonstr_keys = [k for k in x.keys() if not isinstance(k, str)]
            raise ValueError("Cannot convert a dictionary with non-string keys to JSON! (it won't decode properly):\n"
                             + '\n'.join(map(str,nonstr_keys)))
        for k, v in x.items():
            try:
                _check_jsonable(v)
            except ValueError as e:
                raise ValueError(("%s key : " % k) + str(e))
    else:
        raise ValueError("Cannot serialize object of type '%s' to JSON!" % str(type(x)))

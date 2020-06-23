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

import json as _json
import pickle as _pickle
import pathlib as _pathlib
import importlib as _importlib

from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from . import loaders as _load
from . import writers as _write

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
    elif typ == 'text-circuit-lists': ext = '.txt'
    elif typ == 'list-of-protocolobjs': ext = ''
    elif typ == 'dict-of-protocolobjs': ext = ''
    elif typ == 'dict-of-resultsobjs': ext = ''
    elif typ == 'protocolobj': ext = ''
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
    max_size = quick_load if isinstance(quick_load, int) else QUICK_LOAD_MAX_SIZE

    def should_skip_loading(path):
        return quick_load and (path.stat().st_size >= max_size)

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
                if should_skip_loading(pth):
                    val.append(None)
                else:
                    val.append(_load.load_circuit_list(pth))
                i += 1

        elif typ == 'protocolobj':
            protocol_dir = root_dir / (key + ext)
            val = _cls_from_meta_json(protocol_dir).from_dir(protocol_dir, quick_load=quick_load)

        elif typ == 'list-of-protocolobjs':
            i = 0; val = []
            while True:
                pth = root_dir / (key + str(i) + ext)
                if not pth.exists(): break
                val.append(_cls_from_meta_json(pth).from_dir(pth, quick_load=quick_load)); i += 1

        elif typ == 'dict-of-protocolobjs':
            keys = meta[key]; paths = [root_dir / (key + "_" + k + ext) for k in keys]
            val = {k: _cls_from_meta_json(pth).from_dir(pth, quick_load=quick_load) for k, pth in zip(keys, paths)}

        elif typ == 'dict-of-resultsobjs':
            keys = meta[key]; paths = [root_dir / (key + "_" + k + ext) for k in keys]
            val = {k: _cls_from_meta_json(pth)._from_dir_partial(pth, quick_load=quick_load)
                   for k, pth in zip(keys, paths)}

        else:  # cases with 'standard' expected path

            pth = root_dir / (key + ext)
            if pth.is_dir():
                raise ValueError("Expected path: %s is a dir!" % pth)
            elif not pth.exists() or should_skip_loading(pth):
                val = None  # missing files => None values
            elif typ == 'text-circuit-list':
                val = _load.load_circuit_list(pth)
            elif typ == 'json':
                with open(str(pth), 'r') as f:
                    val = _json.load(f)
            elif typ == 'pickle':
                with open(str(pth), 'rb') as f:
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
        ext = _get_auxfile_ext(typ)

        if typ == 'text-circuit-lists':
            for i, circuit_list in enumerate(val):
                pth = root_dir / (auxnm + str(i) + ext)
                _write.write_circuit_list(pth, circuit_list)

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

        elif typ == 'dict-of-resultsobjs':
            meta[auxnm] = list(val.keys())  # just save a list of the keys in the metadata
            for k, obj in val.items():
                obj_dirname = auxnm + "_" + k + ext  # keys must be strings
                obj._write_partial(root_dir / obj_dirname)

        else:
            # standard path cases
            pth = root_dir / (auxnm + ext)

            if val is None:   # None values don't get written
                pass
            elif typ == 'text-circuit-list':
                _write.write_circuit_list(pth, val)
            elif typ == 'json':
                with open(str(pth), 'w') as f:
                    _json.dump(val, f)
            elif typ == 'pickle':
                with open(str(pth), 'wb') as f:
                    _pickle.dump(val, f)
            elif typ in ('none', 'reset'):
                pass
            else:
                raise ValueError("Invalid aux-file type: %s" % typ)

    with open(str(root_dir / 'meta.json'), 'w') as f:
        _json.dump(meta, f)


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
    dirname = _pathlib.Path(dirname)
    if not dirname.is_dir():
        return {}

    ret = {}
    for pth in dirname.iterdir():
        if pth.suffix == '.json':
            with open(str(pth), 'r') as f:
                val = _json.load(f)
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
    for key, val in d.items():
        #TODO: fix this - as we can write some things to json that don't get read back correctly,
        # e.g. dicts with integer keys
        #try:
        #    with open(dirname / (key + '.json'), 'w') as f:
        #        _json.dump(val, f)
        #except:
        #try to remove partial json file??
        with open(str(dirname / (key + '.pkl')), 'wb') as f:
            _pickle.dump(val, f)

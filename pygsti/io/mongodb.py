"""
Serialization routines to/from a MongoDB database
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import pickle as _pickle
import warnings as _warnings
import numpy as _np

from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.data.dataset import DataSet as _DataSet

from pygsti.io import stdinput as _stdinput
from pygsti.io import readers as _load
#from pygsti.io import writers as _write
from pygsti.io.metadir import _check_jsonable, _full_class_name, _to_jsonable, _from_jsonable, _class_for_name
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.baseobjs.mongoserializable import MongoSerializable as _MongoSerializable
from pygsti.baseobjs.mongoserializable import RecursiveRemovalSpecification as _RecursiveRemovalSpecification
from pygsti.baseobjs.mongoserializable import WriteOpsByCollection as _WriteOpsByCollection
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter


def read_auxtree_from_mongodb(mongodb, collection_name, doc_id, auxfile_types_member='auxfile_types',
                              ignore_meta=('_id', 'type',), separate_auxfiletypes=False,
                              quick_load=False):
    """
    Read a document containing links to auxiliary documents from a MongoDB database.

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to load data from.

    collection_name : str
        the MongoDB collection within `mongodb` to read from.

    doc_id : bson.objectid.ObjectId
        The identifier, of the root document to load from the database.

    auxfile_types_member : str
        the name of the attribute within the document that holds the dictionary
        mapping of attributes to auxiliary-file (document) types.  Usually this is
        `"auxfile_types"`.

    ignore_meta : tuple, optional
        Keys within the root document that should be ignored, i.e. not loaded into
        elements of the returned dict.  By default, `"_id"` and `"type"` are in this
        category because they give the database id and a class name to be built,
        respectively, and are not needed in the constructed dictionary.  Unless
        you know what you're doing, leave this as the default.

    separate_auxfiletypes : bool, optional
        If True, then return the `auxfile_types_member` element (a dict
        describing how quantities that aren't in the main document have been
        serialized) as a separate return value, instead of placing it
        within the returned dict.

    quick_load : bool, optional
        Setting this to True skips the loading of members that may take
        a long time to load, namely those in separate documents that are
        large.  When the loading of an attribute is skipped, it is set to `None`.

    Returns
    -------
    loaded_qtys : dict
        A dictionary of the quantities in the main document plus any loaded
        from the auxiliary documents.
    auxfile_types : dict
        Only returned as a separate value when `separate_auxfiletypes=True`.
        A dict describing how members of `loaded_qtys` that weren't loaded
        directly from the main document were serialized.
    """
    doc = mongodb[collection_name].find_one({'_id': doc_id})
    return read_auxtree_from_mongodb_doc(mongodb, collection_name, doc, auxfile_types_member,
                                         ignore_meta, separate_auxfiletypes, quick_load)


def read_auxtree_from_mongodb_doc(mongodb, doc, auxfile_types_member='auxfile_types',
                                  ignore_meta=('_id', 'type',), separate_auxfiletypes=False,
                                  quick_load=False):
    """
    Load the contents of a MongoDB document into a dict.

    The de-serialization possibly uses metadata within to root document to describe
    how associated data is stored in other collections.

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to load data from.

    doc : dict
        The already-retrieved main document being read in.

    auxfile_types_member : str, optional
        The key within the root document that is used to describe how other
        members have been serialized into documents.  Unless you know what you're
        doing, leave this as the default.

    ignore_meta : tuple, optional
        Keys within the root document that should be ignored, i.e. not loaded into
        elements of the returned dict.  By default, `"_id"` and `"type"` are in this
        category because they give the database id and a class name to be built,
        respectively, and are not needed in the constructed dictionary.  Unless
        you know what you're doing, leave this as the default.

    separate_auxfiletypes : bool, optional
        If True, then return the `auxfile_types_member` element (a dict
        describing how quantities that aren't in root document have been
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
        directly from the root document were serialized.
    """
    ret = {}

    for key, val in doc.items():
        if key in ignore_meta: continue
        ret[key] = val

    for key, typ in doc[auxfile_types_member].items():
        if key in ignore_meta: continue  # don't load -> members items in ignore_meta

        bLoaded, val = _load_auxdoc_member(mongodb, key, typ,
                                           doc.get(key, None), quick_load)
        if bLoaded:
            ret[key] = val
        elif val is True:  # val is value of whether to set value to None
            ret[key] = None  # Note: could just have bLoaded be True in this case and val=None

    if separate_auxfiletypes:
        del ret[auxfile_types_member]
        return ret, doc[auxfile_types_member]
    else:
        return ret


def _load_auxdoc_member(mongodb, member_name, typ, metadata, quick_load):
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    # In FUTURE maybe we can implement "quick loading" from a MongoDB, but currently `quick_load` does nothing
    #max_size = quick_load if isinstance(quick_load, int) else QUICK_LOAD_MAX_SIZE
    #def should_skip_loading(path):
    #    return quick_load and (path.stat().st_size >= max_size)

    if cur_typ == 'list':
        if metadata is None:  # signals that value is None, otherwise would at least be an empty list
            val = None
        else:
            val = []
            for i, meta in enumerate(metadata):
                membernm_so_far = member_name + str(i)
                bLoaded, el = _load_auxdoc_member(mongodb, membernm_so_far,
                                                  next_typ, meta, quick_load)
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
                membernm_so_far = member_name + "_" + k
                meta = metadata.get(k, None)
                bLoaded, v = _load_auxdoc_member(mongodb, membernm_so_far,
                                                 next_typ, meta, quick_load)
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
                membernm_so_far = member_name + "_kvpair" + str(i)
                bLoaded, el = _load_auxdoc_member(mongodb, membernm_so_far,
                                                  next_typ, meta, quick_load)
                if bLoaded:
                    if isinstance(k, list): k = tuple(k)  # convert list-type keys -> tuples
                    val[k] = el
                else:
                    raise ValueError("Failed to load %d-th dictionary key: %s" % (i, str(k)))

    elif cur_typ in ('text-circuit-lists', 'protocolobj', 'list-of-protocolobjs', 'dict-of-resultsobjs'):
        raise ValueError("Cannot (and should never need to!) load from deprecated aux-file formats.")

    else:
        val = None  # the default value to load, which is left unchanged if there's no record

        if cur_typ == 'none':  # member is serialized separatey and shouldn't be touched
            return False, False  # explicitly don't load or set value (so set_to_None=False)

        #if should_skip_loading(pth):
        #    val = None  # load 'None' instead of actual data (skip loading this file)

        if cur_typ == 'reset':  # 'reset' doesn't write and loads in as None
            val = None  # no file exists for this member
        elif metadata is None:
            # value was None and we do nothing here
            val = None
        elif cur_typ == 'text-circuit-list':
            coll = mongodb[metadata['collection_name']]
            circuit_doc_ids = metadata['ids']

            circuit_strs = []
            for circuit_doc_id in circuit_doc_ids:
                cdoc = coll.find_one(circuit_doc_id)
                circuit_strs.append(cdoc['circuit_str'])
            val = _load.convert_strings_to_circuits(circuit_strs)

        elif cur_typ == 'dir-serialized-object':
            obj_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            val = _MongoSerializable.from_mongodb_doc(mongodb, metadata['collection_name'],
                                                      obj_doc, quick_load=quick_load)

        elif cur_typ == 'partialdir-serialized-object':
            obj_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            val = _MongoSerializable.from_mongodb_doc(mongodb, metadata['collection_name'],
                                                      obj_doc, quick_load=quick_load, load_data=False)

        elif cur_typ == 'serialized-object':
            obj_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            val = _MongoSerializable.from_mongodb_doc(mongodb, metadata['collection_name'], obj_doc)

        elif cur_typ == 'circuit-str-json':
            obj_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            val = _load.convert_strings_to_circuits(obj_doc['circuit_str_json'])

        elif typ == 'numpy-array':
            array_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            if array_doc is not None:
                assert(array_doc['auxdoc_type'] == cur_typ)
                val = _pickle.loads(array_doc['numpy_array_data'])

        elif typ == 'json':
            json_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            if json_doc is not None:
                assert(json_doc['auxdoc_type'] == cur_typ)
                val = json_doc['json_data']

        elif typ == 'pickle':
            pkl_doc = mongodb[metadata['collection_name']].find_one(metadata['id'])
            if pkl_doc is not None:
                assert(pkl_doc['auxdoc_type'] == cur_typ)
                val = _pickle.loads(pkl_doc['pickle_data'])

        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return True, val  # loading successful - 2nd element is value loaded


def write_obj_to_mongodb_auxtree(obj, mongodb, collection_name, doc_id, auxfile_types_member, omit_attributes=(),
                                 include_attributes=None, additional_meta=None, session=None, overwrite_existing=False):
    """
    Write the attributes of an object to a MongoDB database, potentially as multiple documents.

    Parameters
    ----------
    obj : object
        The object that is to be written.

    mongodb : pymongo.database.Database
        The MongoDB instance to write data to.

    collection_name : str
        the MongoDB collection within `mongodb` to write to.

    doc_id : bson.objectid.ObjectId
        The identifier, of the root document to store in the database.
        If `None` a new id will be created.

    auxfile_types_member : str, optional
        The attribute of `obj` that is used to describe how other
        members should be serialized into separate "auxiliary" documents.
        Unless you know what you're doing, leave this as the default.

    omit_attributes : list or tuple
        List of (string-valued) names of attributes to omit when serializing
        this object.  Usually you should just leave this empty.

    include_attributes : list or tuple or None
        A list of (string-valued) names of attributs to specifically include
        when serializing this object.  If `None`, then *all* attributes are
        included except those specifically omitted via `omit_attributes`.
        If `include_attributes` is not `None` then `omit_attributes` is
        ignored.

    additional_meta : dict, optional
        A dictionary of additional meta-data to be included in the main document
        (but that isn't an attribute of `obj`).

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
    bson.objectid.ObjectId
        The identifer of the root document that was written.
    """
    from bson.objectid import ObjectId

    if doc_id is None:
        doc_id = ObjectId()
    to_insert = {'_id': doc_id}
    write_ops = _WriteOpsByCollection(session)
    add_obj_auxtree_write_ops_and_update_doc(obj, to_insert, write_ops, mongodb, collection_name,
                                             auxfile_types_member, omit_attributes,
                                             include_attributes, additional_meta, overwrite_existing)
    write_ops.add_one_op(collection_name, {'_id': doc_id}, to_insert, overwrite_existing, mongodb)  # add main doc

    try:
        write_ops.execute(mongodb)
    except Exception as e:
        if session is None:
            #Unless this may be a transaction, Try to undo any DB writes we can by deleting the document
            # we just failed to write
            try:
                remove_auxtree_from_mongodb(mongodb, collection_name, doc_id, 'auxfile_types', session)
            except:
                pass  # ok if this fails
        raise e

    return doc_id


def add_obj_auxtree_write_ops_and_update_doc(obj, doc, write_ops, mongodb, collection_name,
                                             auxfile_types_member, omit_attributes=(),
                                             include_attributes=None, additional_meta=None, overwrite_existing=False):
    """
    Similar to `write_obj_to_mongodb_auxtree`, but just collect write operations and update a main-doc dictionary.

    This function effectively performs all the heavy-lifting to write an object to
    a MongoDB database without actually executing any write operations.  Instead, a
    dictionary representing the main document (which we typically assume will be written
    later) is updated and additional write operations (for auxiliary documents) are added
    to a :class:`WriteOpsByCollection` object.  This function is intended for use within
    a :class:`MongoSerializable`-derived object's `_add_auxiliary_write_ops_and_update_doc`
    method.

    Parameters
    ----------
    obj : object
        The object that is to be written.

    doc : dict
        The root-document data, which is updated as needed and is expected to
        be initialized at least with an `_id` key-value pair.

    write_ops : WriteOpsByCollection
        An object that keeps track of `pymongo` write operations on a per-collection
        basis.  This object accumulates write operations to be performed at some point
        in the future.

    mongodb : pymongo.database.Database
        The MongoDB instance that is planned to be written to.  Used to test for existing
        records and *not* to write to, as writing is assumed to be done later, potentially as
        a bulk write operaiton.

    collection_name : str
        the MongoDB collection within `mongodb` that is planned to write to.

    auxfile_types_member : str, optional
        The attribute of `obj` that is used to describe how other
        members should be serialized into separate "auxiliary" documents.
        Unless you know what you're doing, leave this as the default.

    omit_attributes : list or tuple
        List of (string-valued) names of attributes to omit when serializing
        this object.  Usually you should just leave this empty.

    include_attributes : list or tuple or None
        A list of (string-valued) names of attributs to specifically include
        when serializing this object.  If `None`, then *all* attributes are
        included except those specifically omitted via `omit_attributes`.
        If `include_attributes` is not `None` then `omit_attributes` is
        ignored.

    additional_meta : dict, optional
        A dictionary of additional meta-data to be included in the main document
        (but that isn't an attribute of `obj`).

    overwrite_existing : bool, optional
        Whether existing documents should be overwritten.  The default of `False` causes
        a ValueError to be raised if a document with the given `doc_id` already exists.
        Setting this to `True` mimics the behaviour of a typical filesystem, where writing
        to a path can be done regardless of whether it already exists.

    Returns
    -------
    bson.objectid.ObjectId
        The identifer of the root document that was written.
    """
    # Note: include_attributes = None means include everything not omitted
    # Note2: include_attributes takes precedence over omit_attributes
    meta = {'type': _full_class_name(obj)}
    if additional_meta is not None: meta.update(additional_meta)

    # Unless explicitly included, don't store _dbcoordinates in DB
    coords_explicitly_included = include_attributes is not None and '_dbcoordinates' in include_attributes
    if '_dbcoordinates' not in omit_attributes and not coords_explicitly_included:
        omit_attributes = tuple(omit_attributes) + ('_dbcoordinates',)

    if include_attributes is not None:  # include_attributes takes precedence over omit_attributes
        vals = {}
        auxtypes = {}
        potential_auxtypes = obj.__dict__[auxfile_types_member].copy() if (auxfile_types_member is not None) else {}
        for attr_name in include_attributes:
            if attr_name in obj.__dict__:
                vals[attr_name] = obj.__dict__[attr_name]
                if attr_name in potential_auxtypes:
                    auxtypes[attr_name] = potential_auxtypes[attr_name]

    elif len(omit_attributes) > 0:
        vals = obj.__dict__.copy()
        auxtypes = obj.__dict__[auxfile_types_member].copy() if (auxfile_types_member is not None) else {}
        for o in omit_attributes:
            if o in vals: del vals[o]
            if o in auxtypes: del auxtypes[o]
    else:
        vals = obj.__dict__.copy()
        auxtypes = obj.__dict__[auxfile_types_member] if (auxfile_types_member is not None) else {}

    return add_auxtree_write_ops_and_update_doc(doc, write_ops, mongodb, collection_name, vals,
                                                auxtypes, init_meta=meta, overwrite_existing=overwrite_existing)


def write_auxtree_to_mongodb(mongodb, collection_name, doc_id, valuedict, auxfile_types=None, init_meta=None,
                             session=None, overwrite_existing=False):
    """
    Write a dictionary of quantities to a MongoDB database, potentially as multiple documents.

    Write the dictionary by placing everything in `valuedict` into a root document except for
    special key/value pairs ("special" usually because they lend themselves to
    an non-JSON format or they can be particularly large and may exceed MongoDB's maximum
    document size) which are placed in "auxiliary" documents formatted according to
    `auxfile_types` (which itself is saved in the root document).

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to write data to.

    collection_name : str
        the MongoDB collection within `mongodb` to write to.

    doc_id : bson.objectid.ObjectId
        The identifier, of the root document to store in the database.
        If `None` a new id will be created.

    valuedict : dict
        The dictionary of values to serialize to disk.

    auxfile_types : dict, optional
        A dictionary whose keys are a subset of the keys of `valuedict`,
        and whose values are known "aux-file" types.  `auxfile_types[key]`
        says that `valuedict[key]` should be serialized into a separate
        document with the given format rather than be included directly in
        the root document.  If None, this dictionary is assumed to be
        `valuedict['auxfile_types']`.

    init_meta : dict, optional
        A dictionary of "initial" meta-data to be included in the root document
        (but that isn't in `valuedict`).  For example, the class name of an
        object is often stored as in the "type" field of meta.json when the_model
        objects .__dict__ is used as `valuedict`.

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
    bson.objectid.ObjectId
        The identifer of the root document that was written.
    """
    from bson.objectid import ObjectId

    if doc_id is None:
        doc_id = ObjectId()
    to_insert = {'_id': doc_id}
    write_ops = _WriteOpsByCollection(session)
    add_auxtree_write_ops_and_update_doc(to_insert, write_ops, mongodb, collection_name, valuedict,
                                         auxfile_types, init_meta, overwrite_existing)
    write_ops.add_one_op(collection_name, {'_id': doc_id}, to_insert, overwrite_existing, mongodb)  # add main doc

    try:
        write_ops.execute(mongodb)
    except Exception as e:
        if session is None:
            #Unless this may be a transaction, Try to undo any DB writes we can by deleting the document
            # we just failed to write
            try:
                remove_auxtree_from_mongodb(mongodb, collection_name, doc_id, 'auxfile_types', session,
                                            recursive='none')  # just be safe and not delete anything we need
            except:
                pass  # ok if this fails
        raise e

    return doc_id


def add_auxtree_write_ops_and_update_doc(doc, write_ops, mongodb, collection_name, valuedict,
                                         auxfile_types=None, init_meta=None, overwrite_existing=False):
    """
    Similar to `write_auxtree_to_mongodb`, but just collect write operations and update a main-doc dictionary.

    This function effectively performs all the heavy-lifting to write a dictionary to
    multiple documents within a MongoDB database without actually executing any write
    operations.  Instead, a dictionary representing the main document (which we typically
    assume will be written  later) is updated and additional write operations (for auxiliary
    documents) are added to a :class:`WriteOpsByCollection` object.  This function is intended
    for use within a :class:`MongoSerializable`-derived object's `_add_auxiliary_write_ops_and_update_doc`
    method.

    Parameters
    ----------
    doc : dict
        The root-document data, which is updated as needed and is expected to
        be initialized at least with an `_id` key-value pair.

    write_ops : WriteOpsByCollection
        An object that keeps track of `pymongo` write operations on a per-collection
        basis.  This object accumulates write operations to be performed at some point
        in the future.

    mongodb : pymongo.database.Database
        The MongoDB instance that is planned to be written to.  Used to test for existing
        records and *not* to write to, as writing is assumed to be done later, potentially as
        a bulk write operaiton.

    collection_name : str
        the MongoDB collection within `mongodb` that is planned to write to.

    valuedict : dict
        The dictionary of values to serialize to disk.

    auxfile_types : dict, optional
        A dictionary whose keys are a subset of the keys of `valuedict`,
        and whose values are known "aux-file" types.  `auxfile_types[key]`
        says that `valuedict[key]` should be serialized into a separate
        document with the given format rather than be included directly in
        the root document.  If None, this dictionary is assumed to be
        `valuedict['auxfile_types']`.

    init_meta : dict, optional
        A dictionary of "initial" meta-data to be included in the root document
        (but that isn't in `valuedict`).  For example, the class name of an
        object is often stored as in the "type" field of meta.json when the_model
        objects .__dict__ is used as `valuedict`.

    overwrite_existing : bool, optional
        Whether existing documents should be overwritten.  The default of `False` causes
        a ValueError to be raised if a document with the given `doc_id` already exists.
        Setting this to `True` mimics the behaviour of a typical filesystem, where writing
        to a path can be done regardless of whether it already exists.

    Returns
    -------
    bson.objectid.ObjectId
        The identifer of the root document that was written.
    """
    from pymongo import InsertOne, ReplaceOne
    to_insert = {}

    if auxfile_types is None:  # Note: this case may never be used
        auxfile_types = valuedict['auxfile_types']

    if init_meta: to_insert.update(init_meta)
    to_insert['auxfile_types'] = auxfile_types

    #Initial check -- REMOVED because it's ok to "overwrite" the *same* data with overwrite_existing=False
    #if not overwrite_existing and root_mongo_collection.count_documents({'_id': doc_id}, session=session) > 0:
    #    raise ValueError("Document with id=%s exists and `overwrite_existing=False`" % str(doc_id))

    for key, val in valuedict.items():
        if key in auxfile_types: continue  # member is serialized to a separate (sub-)collection
        if isinstance(val, _VerbosityPrinter): val = val.verbosity  # HACK!!
        to_insert[key] = val

    for auxnm, typ in auxfile_types.items():
        if typ in ('none', 'reset'):
            continue
        val = valuedict[auxnm]

        try:
            auxmeta = _write_auxdoc_member(mongodb, write_ops, collection_name, doc['_id'], auxnm,
                                           typ, val, overwrite_existing)
        except Exception as e:
            raise ValueError("FAILED to prepare to write aux doc member %s w/format %s (see direct cause above)"
                             % (auxnm, typ)) from e

        if auxmeta is not None:
            to_insert[auxnm] = auxmeta  # metadata about auxiliary document(s) for this aux name

    doc.update(to_insert)


def _write_auxdoc_member(mongodb, write_ops, parent_collection_name, parent_id, member_name, typ, val,
                         overwrite_existing=False):
    from bson.binary import Binary as _Binary
    from bson.objectid import ObjectId
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    if cur_typ == 'list':
        if val is not None:
            metadata = []
            for i, el in enumerate(val):
                membernm_so_far = member_name + str(i)
                meta = _write_auxdoc_member(mongodb, write_ops, parent_collection_name, parent_id, membernm_so_far,
                                            next_typ, el, overwrite_existing)
                metadata.append(meta)
        else:
            metadata = None

    elif cur_typ == 'dict':
        if val is not None:
            metadata = {}
            for k, v in val.items():
                membernm_so_far = member_name + "_" + k
                meta = _write_auxdoc_member(mongodb, write_ops, parent_collection_name, parent_id, membernm_so_far,
                                            next_typ, v, overwrite_existing)
                metadata[k] = meta
        else:
            metadata = None

    elif cur_typ == 'fancykeydict':
        if val is not None:
            metadata = []
            for i, (k, v) in enumerate(val.items()):
                membernm_so_far = member_name + "_kvpair" + str(i)
                meta = _write_auxdoc_member(mongodb, write_ops, parent_collection_name, parent_id, membernm_so_far,
                                            next_typ, v, overwrite_existing)
                metadata.append((k, meta))
        else:
            metadata = None

    else:
        #Simple types that just write the given file
        metadata = None

        if val is None:   # None values don't get written
            pass
        elif cur_typ in ('none', 'reset'):  # explicitly don't get written
            pass  # and really we shouldn't ever get here since we short circuit in auxmember loop

        elif cur_typ == 'text-circuit-list':
            circuit_doc_ids = []
            for i, circuit in enumerate(val):
                circuit_doc_ids.append(
                    write_ops.add_one_op('pygsti_circuits', {'circuit_str': circuit.str},
                                         {'circuit_str': circuit.str},  # add more circuit info in future?
                                         overwrite_existing, mongodb, check_local_ops=True))
            metadata = {'collection_name': 'pygsti_circuits', 'ids': circuit_doc_ids}

        elif cur_typ == 'dir-serialized-object':
            val_id = val.add_mongodb_write_ops(write_ops, mongodb, overwrite_existing)
            metadata = {'collection_name': val.collection_name, 'id': val_id}

        elif cur_typ == 'partialdir-serialized-object':
            val_id = val.add_mongodb_write_ops(write_ops, mongodb, overwrite_existing,
                                               already_written_data_id="N/A partial")
            metadata = {'collection_name': val.collection_name, 'id': val_id}

        elif cur_typ == 'serialized-object':
            assert(isinstance(val, _MongoSerializable)), \
                "Non-mongo-serializable '%s' object given for a 'serialized-object' auxfile type!" % (str(type(val)))
            val_id = val.add_mongodb_write_ops(write_ops, mongodb, overwrite_existing)
            metadata = {'collection_name': val.collection_name, 'id': val_id}

        elif cur_typ == 'circuit-str-json':
            from .writers import convert_circuits_to_strings
            id_dict = {'parent_id': parent_id, 'member_name': member_name}
            data = id_dict.copy()
            data['circuit_str_json'] = convert_circuits_to_strings(val)
            obj_doc_id = write_ops.add_one_op('pygsti_json_data', id_dict, data,
                                              overwrite_existing, mongodb, check_local_ops=True)
            metadata = {'collection_name': 'pygsti_json_data', 'id': obj_doc_id}

        elif cur_typ == 'numpy-array':
            member_id = {'parent_collection': parent_collection_name, 'parent': parent_id, 'member_name': member_name}
            val_doc = member_id.copy()
            val_doc.update({'auxdoc_type': cur_typ,
                            'numpy_array_data': _Binary(_pickle.dumps(val, protocol=2), subtype=128)})
            val_id = write_ops.add_one_op('pygsti_arrays', member_id, val_doc, overwrite_existing, mongodb)
            metadata = {'collection_name': 'pygsti_arrays', 'id': val_id}

        elif typ == 'json':
            _check_jsonable(val)
            member_id = {'parent_collection': parent_collection_name, 'parent': parent_id, 'member_name': member_name}
            val_doc = member_id.copy()
            val_doc.update({'auxdoc_type': cur_typ,
                            'json_data': val})
            val_id = write_ops.add_one_op('pygsti_json_data', member_id, val_doc, overwrite_existing, mongodb)
            metadata = {'collection_name': 'pygsti_json_data', 'id': val_id}

        elif typ == 'pickle':
            member_id = {'parent_collection': parent_collection_name, 'parent': parent_id, 'member_name': member_name}
            val_doc = member_id.copy()
            val_doc.update({'auxdoc_type': cur_typ,
                            'pickle_data': _Binary(_pickle.dumps(val, protocol=2), subtype=128)})
            val_id = write_ops.add_one_op('pygsti_pickle_data', member_id, val_doc, overwrite_existing, mongodb)
            metadata = {'collection_name': 'pygsti_pickle_data', 'id': val_id}
        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return metadata


def remove_auxtree_from_mongodb(mongodb, collection_name, doc_id, auxfile_types_member='auxfile_types', session=None,
                                recursive=None):
    """
    Remove some or all of the MongoDB documents written by `write_auxtree_to_mongodb`

    Removes a root document and possibly auxiliary documents.

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to remove documents from.

    collection_name : str
        the MongoDB collection within `mongodb` to remove document from.

    doc_id : bson.objectid.ObjectId
        The identifier of the root document stored in the database.

    auxfile_types_member : str, optional
        The key of the stored document used to describe how other
        members are serialized into separate "auxiliary" documents.
        Unless you know what you're doing, leave this as the default.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

    recursive : RecursiveRemovalSpecification, optional
        An object that filters the type of documents that are removed.
        Used when working with inter-related experiment designs, data,
        and results objects to only remove the types of documents you
        know aren't being shared with other documents.

    Returns
    -------
    pymongo.results.DeleteResult
        The result of deleting (or attempting to delete) the root record
    """
    #Note: grab entire document here since we may need values of some members to deleting linked records
    doc = mongodb[collection_name].find_one({'_id': doc_id}, session=session)  # [auxfile_types_member]
    recursive = _RecursiveRemovalSpecification.cast(recursive)
    if doc is None:
        return

    for key, typ in doc[auxfile_types_member].items():
        _remove_auxdoc_member(mongodb, key, typ, doc.get(key, None), session, recursive)

    return mongodb[collection_name].delete_one({'_id': doc_id}, session=session)  # returns deleted count (0 or 1)


def _remove_auxdoc_member(mongodb, member_name, typ, metadata, session, recursive):
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    if cur_typ == 'list':
        if metadata is not None:  # otherwise signals that value is None, and no auxdoc to remove
            for i, meta in enumerate(metadata):
                membernm_so_far = member_name + str(i)
                _remove_auxdoc_member(mongodb, membernm_so_far, next_typ, meta, session, recursive)

    elif cur_typ == 'dict':
        if metadata is not None:  # otherwise signals that value is None, and no auxdoc to remove
            keys = list(metadata.keys())  # sort?
            for k in keys:
                membernm_so_far = member_name + "_" + k
                meta = metadata.get(k, None)
                _remove_auxdoc_member(mongodb, membernm_so_far,
                                      next_typ, meta, session, recursive)

    elif cur_typ == 'fancykeydict':
        if metadata is not None:  # otherwise signals that value is None, and no auxdoc to remove
            keymeta_pairs = list(metadata)  # should be a list of (key, metadata_for_value) pairs
            for i, (k, meta) in enumerate(keymeta_pairs):
                membernm_so_far = member_name + "_kvpair" + str(i)
                _remove_auxdoc_member(mongodb, membernm_so_far,
                                      next_typ, meta, session, recursive)

    else:
        if cur_typ in ('none', 'reset'):  # no auxdoc exists for this member
            return  # done here
        elif metadata is None:
            return  # value was None and so no auxdoc was created -- nothing to remove
        elif cur_typ == 'text-circuit-list':
            if recursive.circuits:
                coll = mongodb[metadata['collection_name']]
                circuit_doc_ids = metadata['ids']
                for circuit_doc_id in circuit_doc_ids:
                    coll.delete_one({'_id': circuit_doc_id}, session=session)  # returns deleted count (0 or 1)
        elif cur_typ == 'circuit-str-json':
            mongodb[metadata['collection_name']].delete_one({'_id': metadata['id']}, session=session)
        elif cur_typ in ('dir-serialized-object', 'partialdir-serialized-object', 'serialized-object'):
            _MongoSerializable.remove_from_mongodb(mongodb, metadata['id'], metadata['collection_name'],
                                                   session, recursive=recursive)
        elif typ in ('numpy-array', 'json', 'pickle'):
            mongodb[metadata['collection_name']].delete_one({'_id': metadata['id']}, session=session)
        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return


def read_dict_from_mongodb(mongodb, collection_name, identifying_metadata):
    """
    Read a dictionary serialized via :func:`write_dict_to_mongodb` into a dictionary.

    The elements of the constructed dictionary are stored as a separate documents in a
    the specified MongoDB collection.

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to read data from.

    collection_name : str
        the MongoDB collection within `mongodb` to read from.

    identifying_metadata : dict
        JSON-able metadata that identifies the dictionary being
        retrieved.

    Returns
    -------
    dict
    """
    from bson.binary import Binary as _Binary

    ret = {}
    for doc in mongodb[collection_name].find(identifying_metadata):
        ret[doc['key']] = _pickle.loads(doc['value']) if isinstance(doc['value'], _Binary) \
            else _from_jsonable(doc['value'])
    return ret


def write_dict_to_mongodb(d, mongodb, collection_name, identifying_metadata, overwrite_existing=False, session=None):
    """
    Write each element of `d` as a separate document in a MongoDB collection

    A document corresponding to each (key, value) pair of `d` is created that contains:
    1. the metadata identifying the collection (`identifying_metadata`)
    2. the pair's key, stored under the key `"key"`
    3. the pair's value, stored under the key `"value"`

    If the element is json-able, it's value is written as a JSON-like dictionary.
    If not, pickle is used to serialize the element and store it in a `bson.binary.Binary`
    object within the database.

    Parameters
    ----------
    d : dict
        the dictionary of elements to serialize.

    mongodb : pymongo.database.Database
        The MongoDB instance to write data to.

    collection_name : str
        the MongoDB collection within `mongodb` to write to.

    identifying_metadata : dict
        JSON-able metadata that identifies the dictionary being
        serialized.  This metadata should be saved for later retrieving
        the elements of `d` from `mongodb_collection`.

    overwrite_existing : bool, optional
        Whether existing documents should be overwritten.  The default of `False` causes
        a ValueError to be raised if a document with the given `doc_id` already exists.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

    Returns
    -------
    None
    """
    write_ops = _WriteOpsByCollection(session)
    add_dict_to_mongodb_write_ops(d, write_ops, mongodb, collection_name, identifying_metadata, overwrite_existing)
    write_ops.execute(mongodb)


def add_dict_to_mongodb_write_ops(d, write_ops, mongodb, collection_name, identifying_metadata, overwrite_existing):
    """
    Similar to `write_dict_to_mongodb`, but just collect write operations and update a main-doc dictionary.

    Parameters
    ----------
    d : dict
        the dictionary of elements to serialize.

    write_ops : WriteOpsByCollection
        An object that keeps track of `pymongo` write operations on a per-collection
        basis.  This object accumulates write operations to be performed at some point
        in the future.

    mongodb : pymongo.database.Database
        The MongoDB instance that is planned to be written to.  Used to test for existing
        records and *not* to write to, as writing is assumed to be done later, potentially as
        a bulk write operaiton.

    collection_name : str
        the MongoDB collection within `mongodb` that is planned to write to.

    identifying_metadata : dict
        JSON-able metadata that identifies the dictionary being
        serialized.  This metadata should be saved for later retrieving
        the elements of `d` from `mongodb_collection`.

    overwrite_existing : bool, optional
        Whether existing documents should be overwritten.  The default of `False` causes
        a ValueError to be raised if a document with the given `doc_id` already exists.

    Returns
    -------
    None
    """
    from bson.binary import Binary as _Binary

    for key, val in d.items():
        to_insert = identifying_metadata.copy()
        to_insert['key'] = key
        full_id = to_insert.copy()  # id metadata and key = full id which should be unique in collection
        try:
            val_to_insert = _to_jsonable(val)
            _check_jsonable(val_to_insert)
        except Exception as e:
            _warnings.warn("Could not write %s key as jsonable (falling back on pickle format):\n" % key + str(e))
            val_to_insert = _Binary(_pickle.dumps(val, protocol=2), subtype=128)

        to_insert['value'] = val_to_insert
        write_ops.add_one_op(collection_name, full_id, to_insert, overwrite_existing, mongodb)


def remove_dict_from_mongodb(mongodb, collection_name, identifying_metadata, session=None):
    """
    Remove elements of (separate documents) of a dictionary stored in a MongoDB collection

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to remove data from.

    collection_name : str
        the MongoDB collection within `mongodb` to remove documents from.

    identifying_metadata : dict
        JSON-able metadata that identifies the dictionary being
        serialized.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

    Returns
    -------
    None
    """
    return mongodb[collection_name].delete_many(identifying_metadata, session=session)


def create_mongodb_indices_for_pygsti_collections(mongodb):
    """
    Create, if not existing already, indices useful for speeding up pyGSTi MongoDB operations.

    Indices are created as necessary within `pygsti_*` collections.  While
    not necessary for database operations, these indices may dramatically speed
    up the reading and writing of pygsti objects to/from a Mongo database.  You
    only need to call this *once* per database, typically when the database is
    first setup.

    Parameters
    ----------
    mongodb : pymongo.database.Database
        The MongoDB instance to create indices in.

    Returns
    -------
    None
    """
    import pymongo as _pymongo

    def create_unique_index(collection_name, index_name, keys):
        ii = mongodb[collection_name].index_information()
        if index_name in ii:
            print("Index %s in %s collection already exists." % (index_name, collection_name))
        else:
            mongodb[collection_name].create_index(keys, name=index_name, unique=True)
            print("Created index %s in %s collection." % (index_name, collection_name))

    create_unique_index('pygsti_circuits', 'circuit_str', [('circuit_str', _pymongo.ASCENDING)])
    create_unique_index('pygsti_datarows', 'parent_and_circuit', [('parent', _pymongo.ASCENDING),
                                                                  ('circuit', _pymongo.ASCENDING)])
    create_unique_index('pygsti_protocol_data_caches', 'parent_member_key',
                        [('protocoldata_parent', _pymongo.ASCENDING),
                         ('member', _pymongo.ASCENDING),
                         ('key', _pymongo.ASCENDING)])

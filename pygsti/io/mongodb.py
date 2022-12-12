"""
Serialization routines to/from a MongoDB database
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter


#Names used for subcollections
subcollection_names = {
    'circuits': 'circuits',
    'objects': 'objects',
    'arrays': 'numpy_arrays',
    'datarows': 'datarows',
}

#Top-level collection name for storing stand-alone objects
# (those with their own write_to_mongodb methods that but are *not* treenodes, e.g., Estimates)
STANDALONE_COLLECTION_NAME = 'pygsti_standalone_objects'


def cnm(subcollection_key):
    return subcollection_names[subcollection_key]


def read_auxtree_from_mongodb(root_mongo_collection, doc_id, auxfile_types_member='auxfile_types',
                              ignore_meta=('_id', 'type',), separate_auxfiletypes=False,
                              quick_load=False):
    """
    Load the contents of a MongoDB document into a dict.

    The de-serialization possibly uses metadata within to root document to describe
    how associated data is stored in other collections.

    Parameters
    ----------
    root_mongo_collection : pymongo.collection.Collection
        The root MongoDB collection to load data from.  (Sub-collections are read
        from as needed.)

    doc_id : object
        The identifier, usually a `bson.objectid.ObjectId` or string, of the
        root document to load from the database.

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
    doc = root_mongo_collection.find_one({'_id': doc_id})
    ret = {}

    for key, val in doc.items():
        if key in ignore_meta: continue
        ret[key] = val

    for key, typ in doc[auxfile_types_member].items():
        if key in ignore_meta: continue  # don't load -> members items in ignore_meta

        bLoaded, val = _load_subcollection_member(root_mongo_collection, doc_id, key, typ,
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


def _load_subcollection_member(root_mongo_collection, parent_id, member_name, typ, metadata, quick_load):
    from pymongo import ASCENDING, DESCENDING
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
                bLoaded, el = _load_subcollection_member(root_mongo_collection, parent_id, membernm_so_far,
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
                bLoaded, v = _load_subcollection_member(root_mongo_collection, parent_id, membernm_so_far,
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
                bLoaded, el = _load_subcollection_member(root_mongo_collection, parent_id, membernm_so_far,
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
        elif cur_typ == 'text-circuit-list':
            coll = root_mongo_collection[cnm('circuits')]
            circuit_strs = []
            for i, cdoc in enumerate(coll.find({'parent': parent_id,
                                                'member_name': member_name}).sort('index', ASCENDING)):
                assert(cdoc['auxfile_type'] == cur_typ)
                assert(cdoc['index'] == i)
                circuit_strs.append(cdoc['circuit_str'])
            val = _load.convert_strings_to_circuits(circuit_strs)

        elif cur_typ == 'dir-serialized-object':
            coll = root_mongo_collection[cnm('objects')]
            link_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})
            if link_doc is not None:
                assert(link_doc['auxfile_type'] == cur_typ)
                standalone_id = link_doc['standalone_object_id']
    
                coll = root_mongo_collection.database[STANDALONE_COLLECTION_NAME]
                obj_doc = coll.find_one({'_id': standalone_id}, ['type'])
                val = _class_for_name(obj_doc['type']).from_mongodb(coll, standalone_id, quick_load=quick_load)

        elif cur_typ == 'partialdir-serialized-object':
            coll = root_mongo_collection[cnm('objects')]
            link_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})

            if link_doc is not None:
                assert(link_doc['auxfile_type'] == cur_typ)
                standalone_id = link_doc['standalone_object_id']

                coll = root_mongo_collection.database[STANDALONE_COLLECTION_NAME]
                obj_doc = coll.find_one({'_id': standalone_id}, ['type'])
                val = _class_for_name(obj_doc['type'])._from_mongodb_partial(coll, standalone_id, quick_load=quick_load)

        elif cur_typ == 'serialized-object':
            coll = root_mongo_collection[cnm('objects')]
            obj_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})
            if obj_doc is not None:
                assert(obj_doc['auxfile_type'] == cur_typ)
                if obj_doc['serialization_type'] == 'mongodb':
                    val = _MongoSerializable.from_mongodb_serialization(obj_doc['object'],
                                                                        root_mongo_collection.database)
                else:
                    val = _NicelySerializable.from_nice_serialization(obj_doc['object'])

        elif cur_typ == 'circuit-str-json':
            coll = root_mongo_collection[cnm('circuits')]
            circuit_strs = []
            for i, cdoc in enumerate(coll.find({'parent': parent_id,
                                                'member_name': member_name}).sort('index', ASCENDING)):
                assert(cdoc['auxfile_type'] == cur_typ)
                assert(cdoc['index'] == i)
                circuit_strs.append(cdoc['circuit_str'])
            val = _load.convert_strings_to_circuits(circuit_strs)

        elif typ == 'numpy-array':
            coll = root_mongo_collection[cnm('arrays')]
            array_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})
            if array_doc is not None:
                assert(array_doc['auxfile_type'] == cur_typ)
                val = _pickle.loads(array_doc['numpy_array'])

        elif typ == 'json':
            coll = root_mongo_collection[cnm('objects')]
            obj_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})
            if obj_doc is not None:
                assert(obj_doc['auxfile_type'] == typ)
                val = obj_doc['object']

        elif typ == 'pickle':
            coll = root_mongo_collection[cnm('objects')]
            obj_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})
            if obj_doc is not None:
                assert(obj_doc['auxfile_type'] == typ)
                val = _pickle.loads(obj_doc['object'])

        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return True, val  # loading successful - 2nd element is value loaded


class WriteOpsBySubcollection(dict):
    def __init__(self, allowed_subcollection_names=None):
        if allowed_subcollection_names is not None:
            self.allowed_subcollection_names = allowed_subcollection_names
        else:
            self.allowed_subcollection_names = (cnm('circuits'), cnm('objects'), cnm('arrays'))

        # separate "special" list for stand-alone object (i.e. objects
        #  with a .write_to_mongo method) writes 'pygsti_standalone_objects'
        self.standalone_writes = []

        super().__init__({k: [] for k in self.allowed_subcollection_names})

    def add_ops_by_subcollection(self, other_ops):
        assert(isinstance(other_ops, WriteOpsBySubcollection))
        for k, v in other_ops.items():
            self[k].extend(v)
        self.standalone_writes.extend(other_ops.standalone_writes)

    def add_one_op(self, subcollection_name, full_id, rest_of_info, overwrite_existing, root_mongo_collection):
        from pymongo import InsertOne, ReplaceOne
        assert(subcollection_name in self.allowed_subcollection_names)
        info = full_id.copy()
        info.update(rest_of_info)
        if overwrite_existing is True:
            self[subcollection_name].append(ReplaceOne(full_id, info, upsert=True))
        else:
            existing_doc = root_mongo_collection[subcollection_name].find_one(full_id)
            if existing_doc is None:  # then insert the document as given
                self[subcollection_name].append(InsertOne(info))
            else:
                info_to_check = prepare_doc_for_existing_doc_check(info, existing_doc)
                if existing_doc != info_to_check:
                    # overwrite_existing=False and a doc already exists AND docs are *different* => error
                    diff_lines = recursive_compare_str(existing_doc, info_to_check, 'existing doc', 'doc to insert')
                    raise ValueError("*Different* document with %s exists and `overwrite_existing=False`:\n%s"
                                     % (str(full_id), '\n'.join(diff_lines)))
                # else do nothing, since doc exists and matches what we want to write => no error


def write_obj_to_mongodb_auxtree(obj, root_mongo_collection, doc_id, auxfile_types_member, omit_attributes=(),
                                 include_attributes=None, additional_meta=None, session=None, overwrite_existing=False):
    # Note: include_attributes = None means include everything not omitted
    # Note2: include_attributes takes precedence over omit_attributes
    meta = {'type': _full_class_name(obj)}
    if additional_meta is not None: meta.update(additional_meta)

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
        vals = obj.__dict__
        auxtypes = obj.__dict__[auxfile_types_member] if (auxfile_types_member is not None) else {}

    return write_auxtree_to_mongodb(root_mongo_collection, doc_id, vals, auxtypes, init_meta=meta,
                                    session=session, overwrite_existing=overwrite_existing)


def write_auxtree_to_mongodb(root_mongo_collection, doc_id, valuedict, auxfile_types=None, init_meta=None,
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
    root_mongo_collection : pymongo.collection.Collection
        The root MongoDB collection to write data to.  Sub-collections are created
        and written to as needed.

    doc_id : object
        The identifier, usually a `bson.objectid.ObjectId` or string, of the
        root document to store in the database.  If `None` a new id will be
        created.

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
    None
    """
    from bson.objectid import ObjectId
    from pymongo import InsertOne, ReplaceOne
    to_insert = {}

    if auxfile_types is None:  # Note: this case may never be used
        auxfile_types = valuedict['auxfile_types']

    if doc_id is None:
        doc_id = ObjectId()  # Note: may run into trouble if _id must be str

    if init_meta: to_insert.update(init_meta)
    to_insert['auxfile_types'] = auxfile_types
    to_insert['_id'] = doc_id

    #Initial check -- REMOVED because it's ok to "overwrite" the *same* data with overwrite_existing=False
    #if not overwrite_existing and root_mongo_collection.count_documents({'_id': doc_id}, session=session) > 0:
    #    raise ValueError("Document with id=%s exists and `overwrite_existing=False`" % str(doc_id))

    for key, val in valuedict.items():
        if key in auxfile_types: continue  # member is serialized to a separate (sub-)collection
        if isinstance(val, _VerbosityPrinter): val = val.verbosity  # HACK!!
        to_insert[key] = val

    write_ops = WriteOpsBySubcollection()
    for auxnm, typ in auxfile_types.items():
        val = valuedict[auxnm]

        try:
            auxmeta, ops = _write_subcollection_member(root_mongo_collection, doc_id, auxnm, typ, val,
                                                       session, overwrite_existing)
        except Exception as e:
            raise ValueError("FAILED to prepare to write aux doc member %s w/format %s (see direct cause above)"
                             % (auxnm, typ)) from e

        write_ops.add_ops_by_subcollection(ops)
        if auxmeta is not None:
            to_insert[auxnm] = auxmeta  # metadata about auxfile(s) for this auxnm

    try:
        for standalone_id, obj, bPartial in write_ops.standalone_writes:
            if not bPartial:
                obj.write_to_mongodb(root_mongo_collection.database[STANDALONE_COLLECTION_NAME], standalone_id,
                                     session=session, overwrite_existing=overwrite_existing)
            else:
                obj._write_partial_to_mongodb(root_mongo_collection.database[STANDALONE_COLLECTION_NAME], standalone_id,
                                              session=session, overwrite_existing=overwrite_existing)

        for subcollection_name, ops in write_ops.items():
            if len(ops) > 0:  # bulk_write fails if ops is an empty list
                root_mongo_collection[subcollection_name].bulk_write(ops, session=session)

        if overwrite_existing:
            root_mongo_collection.replace_one({'_id': doc_id}, to_insert, upsert=True, session=session)
        else:
            existing_doc = root_mongo_collection.find_one({'_id': doc_id})
            if existing_doc is None:  # then insert the document as given
                root_mongo_collection.insert_one(to_insert, session=session)
            else:
                to_check = prepare_doc_for_existing_doc_check(to_insert, existing_doc)
                if existing_doc != to_check:
                    diff_lines = recursive_compare_str(existing_doc, to_check, 'existing doc', 'doc to insert')
                    raise ValueError("*Different* document with _id=%s exists and `overwrite_existing=False`:\n%s"
                                     % (str(doc_id), '\n'.join(diff_lines)))
                # else do nothing, since doc exists and matches what we want to write => no error

    except Exception as e:
        if session is None:
            #Unless this may be a transaction, Try to undo any DB writes we can by deleting the document
            # we just failed to write
            try:
                remove_auxtree_from_mongodb(root_mongo_collection, doc_id, 'auxfile_types', session)
            except:
                pass  # ok if this fails
        raise e


def _write_subcollection_member(root_mongo_collection, parent_id, member_name, typ, val,
                                session=None, overwrite_existing=False):
    from bson.binary import Binary as _Binary
    from bson.objectid import ObjectId
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    write_ops = WriteOpsBySubcollection()
    if cur_typ == 'list':
        if val is not None:
            metadata = []
            for i, el in enumerate(val):
                membernm_so_far = member_name + str(i)
                meta, ops = _write_subcollection_member(root_mongo_collection, parent_id, membernm_so_far,
                                                        next_typ, el, session, overwrite_existing)
                write_ops.add_ops_by_subcollection(ops)
                metadata.append(meta)
        else:
            metadata = None

    elif cur_typ == 'dict':
        if val is not None:
            metadata = {}
            for k, v in val.items():
                membernm_so_far = member_name + "_" + k
                meta, ops = _write_subcollection_member(root_mongo_collection, parent_id, membernm_so_far,
                                                        next_typ, v, session, overwrite_existing)
                write_ops.add_ops_by_subcollection(ops)
                metadata[k] = meta
        else:
            metadata = None

    elif cur_typ == 'fancykeydict':
        if val is not None:
            metadata = []
            for i, (k, v) in enumerate(val.items()):
                membernm_so_far = member_name + "_kvpair" + str(i)
                meta, ops = _write_subcollection_member(root_mongo_collection, parent_id, membernm_so_far,
                                                        next_typ, v, session, overwrite_existing)
                write_ops.add_ops_by_subcollection(ops)
                metadata.append((k, meta))
        else:
            metadata = None

    else:
        #Simple types that just write the given file
        metadata = None
        member_id = {'parent': parent_id, 'member_name': member_name}

        if val is None:   # None values don't get written
            pass
        elif cur_typ in ('none', 'reset'):  # explicitly don't get written
            pass

        elif cur_typ == 'text-circuit-list':
            for i, circuit in enumerate(val):
                write_ops.add_one_op(cnm('circuits'), member_id,
                                     {'auxfile_type': cur_typ, 'index': i, 'circuit_str': circuit.str},
                                     overwrite_existing, root_mongo_collection)

        elif cur_typ == 'dir-serialized-object':
            existing_link_doc = root_mongo_collection[cnm('objects')].find_one(member_id)
            standalone_id = ObjectId() if (existing_link_doc is None) else existing_link_doc['standalone_object_id']
            write_ops.standalone_writes.append((standalone_id, val, False))  # list of (id, object_to_write, bPartial)
            write_ops.add_one_op(cnm('objects'), member_id,
                                 {'auxfile_type': cur_typ, 'standalone_object_id': standalone_id},
                                 overwrite_existing, root_mongo_collection)

        elif cur_typ == 'partialdir-serialized-object':
            existing_link_doc = root_mongo_collection[cnm('objects')].find_one(member_id)
            standalone_id = ObjectId() if (existing_link_doc is None) else existing_link_doc['standalone_object_id']
            write_ops.standalone_writes.append((standalone_id, val, True))  # list of (id, object_to_write, bPartial)
            write_ops.add_one_op(cnm('objects'), member_id,
                                 {'auxfile_type': cur_typ, 'standalone_object_id': standalone_id},
                                 overwrite_existing, root_mongo_collection)

        elif cur_typ == 'serialized-object':
            assert(isinstance(val, _NicelySerializable)), \
                "Non-nicely-serializable '%s' object given for a 'serialized-object' auxfile type!" % (str(type(val)))
            if isinstance(val, _MongoSerializable):
                jsonable = val.to_mongodb_serialization(root_mongo_collection.database)
                sertype = "mongodb"
            else:
                jsonable = val.to_nice_serialization()
                sertype = "nice"
            write_ops.add_one_op(cnm('objects'), member_id, {'auxfile_type': cur_typ, 'object': jsonable,
                                                             'serialization_type': sertype},
                                 overwrite_existing, root_mongo_collection)

        elif cur_typ == 'circuit-str-json':
            for i, circuit in enumerate(val):
                write_ops.add_one_op(cnm('circuits'), member_id,
                                     {'auxfile_type': cur_typ, 'index': i, 'circuit_str': circuit.str},
                                     overwrite_existing, root_mongo_collection)

        elif cur_typ == 'numpy-array':
            write_ops.add_one_op(cnm('arrays'), member_id,
                                 {'auxfile_type': cur_typ,
                                  'numpy_array': _Binary(_pickle.dumps(val, protocol=2), subtype=128)},
                                 overwrite_existing, root_mongo_collection)

        elif typ == 'json':
            _check_jsonable(val)
            write_ops.add_one_op(cnm('objects'), member_id,
                                 {'auxfile_type': cur_typ, 'object': val},
                                 overwrite_existing, root_mongo_collection)

        elif typ == 'pickle':
            write_ops.add_one_op(cnm('objects'), member_id,
                                 {'auxfile_type': cur_typ,
                                  'object': _Binary(_pickle.dumps(val, protocol=2), subtype=128)},
                                 overwrite_existing, root_mongo_collection)

        else:
            raise ValueError("Invalid aux-file type: %s" % typ)

    return metadata, write_ops


def remove_auxtree_from_mongodb(root_mongo_collection, doc_id, auxfile_types_member='auxfile_types', session=None):
    """
    Remove a stored dictionary from a MongoDB database.

    Removes the root document and any auxiliary documents.

    Parameters
    ----------
    root_mongo_collection : pymongo.collection.Collection
        The MongoDB collection of the root document to remove.

    doc_id : object
        The identifier, usually a `bson.objectid.ObjectId` or string, of the
        root document.

    auxfile_types_member : str, optional
        The key within the root document that is used to describe how other
        members have been serialized into documents.  Unless you know what you're
        doing, leave this as the default.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

    Returns
    -------
    pymongo.results.DeleteResult
        The result of deleting (or attempting to delete) the root record
    """
    #Note: grab entire document here since we may need values of some members to deleting linked records
    doc = root_mongo_collection.find_one({'_id': doc_id}, session=session)  # [auxfile_types_member]
    if doc is None:
        return

    #Remove any auxfile members that could have linked standalone records that we can't just clear in bulk
    all_standalone_ids = []
    for key, typ in doc[auxfile_types_member].items():
        if 'dir-serialized-object' in typ:  # allow for "partialdir" and for, e.g., "dict:dir-serialized-object"
            all_standalone_ids.extend(_get_standalone_ids_of_member(root_mongo_collection, doc_id, key, typ,
                                                                    doc.get(key, None)))

    #Remove standalone ids
    coll = root_mongo_collection.database[STANDALONE_COLLECTION_NAME]
    for standalone_id in all_standalone_ids:
        obj_doc = coll.find_one({'_id': standalone_id}, ['type'])
        _class_for_name(obj_doc['type']).remove_from_mongodb(coll, standalone_id, session=session)

    # Begin removing DB documents here -- before this point we're just figuring out what to remove

    #Remove other auxfile documents and the original
    for aux_subcollection_name in (cnm('objects'), cnm('circuits'), cnm('arrays')):
        root_mongo_collection[aux_subcollection_name].delete_many({'parent': doc_id}, session=session)

    return root_mongo_collection.delete_one({'_id': doc_id}, session=session)  # returns deleted count (0 or 1)


def _get_standalone_ids_of_member(root_mongo_collection, parent_id, member_name, typ, val):
    subtypes = typ.split(':')
    cur_typ = subtypes[0]
    next_typ = ':'.join(subtypes[1:])

    standalone_ids = []
    if cur_typ == 'list':
        if val is not None:
            for i, el in enumerate(val):
                membernm_so_far = member_name + str(i)
                ids = _get_standalone_ids_of_member(root_mongo_collection, parent_id, membernm_so_far,
                                                    next_typ, el)
                standalone_ids.extend(ids)

    elif cur_typ == 'dict':
        if val is not None:
            for k, v in val.items():
                membernm_so_far = member_name + "_" + k
                ids = _get_standalone_ids_of_member(root_mongo_collection, parent_id, membernm_so_far,
                                                    next_typ, v)
                standalone_ids.extend(ids)

    elif cur_typ == 'fancykeydict':
        if val is not None:
            for i, (k, v) in enumerate(val.items()):
                membernm_so_far = member_name + "_kvpair" + str(i)
                ids = _get_standalone_ids_of_member(root_mongo_collection, parent_id, membernm_so_far,
                                                    next_typ, v)
                standalone_ids.extend(ids)
    else:
        if typ == 'dir-serialized-object' or typ == 'partialdir-serialized-object':
            coll = root_mongo_collection[cnm('objects')]
            link_doc = coll.find_one({'parent': parent_id, 'member_name': member_name})
            if link_doc is not None:
                assert(link_doc['auxfile_type'] == typ)
                standalone_id = link_doc['standalone_object_id']
                standalone_ids.append(standalone_id)

    return standalone_ids


def read_dict_from_mongodb(mongodb_collection, identifying_metadata):
    """
    Read a dictionary serialized via :function:`write_dict_to_mongodb` into a dictionary.

    The elements of the constructed dictionary are stored as a separate documents in a
    the specified MongoDB collection.

    Parameters
    ----------
    mongodb_collection : pymongo.collection.Collection
        the MongoDB collection to read from.

    identifying_metadata : dict
        JSON-able metadata that identifies the dictionary being
        retrieved.

    Returns
    -------
    dict
    """
    from bson.binary import Binary as _Binary

    ret = {}
    for doc in mongodb_collection.find(identifying_metadata):
        ret[doc['key']] = _pickle.loads(doc['value']) if isinstance(doc['value'], _Binary) \
            else _from_jsonable(doc['value'])
    return ret


def write_dict_to_mongodb(d, mongodb_collection, identifying_metadata, session=None):
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

    mongodb_collection : pymongo.collection.Collection
        the MongoDB collection to write to.

    identifying_metadata : dict
        JSON-able metadata that identifies the dictionary being
        serialized.  This metadata should be saved for later retrieving
        the elements of `d` from `mongodb_collection`.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

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
        mongodb_collection.replace_one(full_id, to_insert, upsert=True, session=session)


def remove_dict_from_mongodb(mongodb_collection, identifying_metadata, session=None):
    """
    Remove elements of (separate documents) of a dictionary stored in a MongoDB collection

    Parameters
    ----------
    mongodb_collection : pymongo.collection.Collection
        the MongoDB collection to remove documents from.

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
    return mongodb_collection.delete_many(identifying_metadata, session=session)


def write_dataset_to_mongodb(dataset, mongodb_collection, identifying_metadata,
                             circuits=None, outcome_label_order=None, with_times="auto",
                             datarow_subcollection_name=None, session=None):
    """
    Write a data set to a MongoDB database.

    A single document is created in the given `mongodb_collection` representing
    the entire dataset, and one document per data-row (circuit) is created in the
    sub-collection of `mongodb_collection` named by `datarow_subollection_name`.

    Parameters
    ----------
    dataset : DataSet
        the dictionary of elements to serialize.

    mongodb_collection : pymongo.collection.Collection
        the MongoDB collection to write to.

    identifying_metadata : dict
        JSON-able metadata that identifies the data set being
        serialized.  This metadata should be saved for later retrieving
        the elements of `d` from `mongodb_collection`.

    circuits : list of Circuits, optional
        The list of circuits to include in the written dataset.
        If None, all circuits are output.

    outcome_label_order : list, optional
        A list of the outcome labels in dataset which specifies
        the column order in the output file.

    with_times : bool or "auto", optional
        Whether to include (save) time-stamp information in output.  This
        can only be True when `fixed_column_mode=False`.  `"auto"` will set
        this to True if `fixed_column_mode=False` and `dataset` has data at
        non-trivial (non-zero) times.

    datarow_subcollection_name : str, optional
        The name of the MongoDB subcollection that holds the written
        data set's row data as one record per row.  If `None`, defaults
        to `pygsti.io.mongodb.subcollection_names['datarows']`.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

    Returns
    -------
    None
    """
    if circuits is not None:
        if len(circuits) > 0 and not isinstance(circuits[0], _Circuit):
            raise ValueError("Argument circuits must be a list of Circuit objects!")
    else:
        circuits = list(dataset.keys())

    if datarow_subcollection_name is None:
        datarow_subcollection_name = cnm('datarows')

    if outcome_label_order is not None:  # convert to tuples if needed
        outcome_label_order = [(ol,) if isinstance(ol, str) else ol
                               for ol in outcome_label_order]

    outcomeLabels = dataset.outcome_labels
    if outcome_label_order is not None:
        assert(len(outcome_label_order) == len(outcomeLabels))
        assert(all([ol in outcomeLabels for ol in outcome_label_order]))
        assert(all([ol in outcome_label_order for ol in outcomeLabels]))
        outcomeLabels = outcome_label_order
        oli_map_data = {dataset.olIndex[ol]: i for i, ol in enumerate(outcomeLabels)}  # dataset -> stored indices

        def oli_map(outcome_label_indices):
            return [oli_map_data[i] for i in outcome_label_indices]
    else:
        def oli_map(outcome_label_indices):
            return [i.item() for i in outcome_label_indices]  # converts numpy types -> native python types

    dataset_doc = identifying_metadata.copy()
    dataset_doc['outcomes'] = outcomeLabels
    dataset_doc['comment'] = dataset.comment if hasattr(dataset, 'comment') else None
    dataset_doc['datarow_subcollection_name'] = datarow_subcollection_name

    if with_times == "auto":
        trivial_times = dataset.has_trivial_timedependence
    else:
        trivial_times = not with_times

    if '_id' in identifying_metadata:
        dataset_id = identifying_metadata['_id']
    else:
        from bson.objectid import ObjectId as _ObjectId
        existing_dataset_doc = mongodb_collection.find_one(identifying_metadata, session=session)
        dataset_id = existing_dataset_doc['_id'] if (existing_dataset_doc is not None) else _ObjectId()
        identifying_metadata = identifying_metadata.copy()
        identifying_metadata['_id'] = dataset_id  # be sure to query & replace the same ID below

    mongodb_collection.replace_one(identifying_metadata, dataset_doc, upsert=True, session=session)
    datarow_collection = mongodb_collection[datarow_subcollection_name]

    for i, circuit in enumerate(circuits):  # circuit should be a Circuit object here
        dataRow = dataset[circuit]
        datarow_doc = {'index': i,
                       'circuit': circuit.str,
                       'parent': dataset_id,
                       'outcome_indices': oli_map(dataRow.oli),
                       'repetitions': [r.item() for r in dataRow.reps]  # converts numpy -> Python native types
                       }

        if trivial_times:  # ensure that "repetitions" are just "counts" in trivial-time case
            assert(len(dataRow.oli) == len(set(dataRow.oli))), "Duplicate value in trivial-time data set row!"
        else:
            datarow_doc['times'] = list(dataRow.time)

        if dataRow.aux:
            datarow_doc['aux'] = dataRow.aux  # needs to be JSON-able!

        datarow_collection.replace_one({'circuit': circuit.str, 'parent': dataset_id}, datarow_doc,
                                       upsert=True, session=session)


def remove_dataset_from_mongodb(mongodb_collection, identifying_metadata,
                                datarow_subcollection_name=None, session=None):
    """
    Remove (delete) a data set from a MongoDB database.

    Parameters
    ----------
    mongodb_collection : pymongo.collection.Collection
        the MongoDB collection to remove a data set from.

    identifying_metadata : dict
        JSON-able metadata that identifies the data set being
        serialized.

    datarow_subcollection_name : str, optional
        The name of the MongoDB subcollection that holds the
        data set's row data.  If `None`, defaults to
        `pygsti.io.mongodb.subcollection_names['datarows']`.

    session : pymongo.client_session.ClientSession, optional
        MongoDB session object to use when interacting with the MongoDB
        database. This can be used to implement transactions
        among other things.

    Returns
    -------
    None
    """
    if datarow_subcollection_name is None:
        datarow_subcollection_name = cnm('datarows')
    dataset_doc = mongodb_collection.find_one_and_delete(identifying_metadata, session=session)
    mongodb_collection[datarow_subcollection_name].delete_many({'parent': dataset_doc['_id']}, session=session)


def read_dataset_from_mongodb(mongodb_collection, identifying_metadata, collision_action="aggregate",
                              record_zero_counts=False, with_times="auto", circuit_parse_cache=None, verbosity=1):
    """
    Load a DataSet from a MongoDB database.

    Parameters
    ----------
    mongodb_collection : pymongo.collection.Collection
        the MongoDB collection to read from.

    identifying_metadata : dict
        JSON-able metadata that identifies the data set being
        loaded.

    collision_action : {"aggregate", "keepseparate"}
        Specifies how duplicate circuits should be handled.  "aggregate"
        adds duplicate-circuit counts, whereas "keepseparate" tags duplicate
        circuits by setting their `.occurrence` IDs to sequential positive integers.

    record_zero_counts : bool, optional
        Whether zero-counts are actually recorded (stored) in the returned
        DataSet.  If False, then zero counts are ignored, except for potentially
        registering new outcome labels.

    with_times : bool or "auto", optional
        Whether to the time-stamped data format should be read in.  If
        "auto", then the time-stamped format is allowed but not required on a
        per-circuit basis (so the dataset can contain both formats).  Typically
        you only need to set this to False when reading in a template file.

    circuit_parse_cache : dict, optional
        A dictionary mapping qubit string representations into created
        :class:`Circuit` objects, which can improve performance by reducing
        or eliminating the need to parse circuit strings.

    verbosity : int, optional
        If zero, no output is shown.  If greater than zero,
        loading progress is shown.

    Returns
    -------
    DataSet
    """
    from pymongo import ASCENDING, DESCENDING

    dataset_doc = mongodb_collection.find_one(identifying_metadata)
    if dataset_doc is None:
        raise ValueError("Could not find a dataset with the given identifying metadata!")

    datarow_subcollection_name = dataset_doc['datarow_subcollection_name']
    outcomeLabels = dataset_doc['outcomes']

    dataset = _DataSet(outcome_labels=outcomeLabels, collision_action=collision_action,
                       comment=dataset_doc['comment'])
    parser = _stdinput.StdInputParser()

    datarow_collection = mongodb_collection[datarow_subcollection_name]
    for i, datarow_doc in enumerate(datarow_collection.find({'parent': dataset_doc['_id']}).sort('index', ASCENDING)):
        if i != datarow_doc['index']:
            _warnings.warn("Data set's row data is incomplete! There seem to be missing rows.")

        circuit = parser.parse_circuit(datarow_doc['circuit'], lookup={},  # allow a lookup to be passed?
                                       create_subcircuits=not _Circuit.default_expand_subcircuits)

        oliArray = _np.array(datarow_doc['outcome_indices'], dataset.oliType)
        countArray = _np.array(datarow_doc['repetitions'], dataset.repType)
        if 'times' not in datarow_doc:  # with_times can be False or 'auto'
            if with_times is True:
                raise ValueError("Circuit %d does not contain time information and 'with_times=True'" % i)
            timeArray = _np.zeros(countArray.shape[0], dataset.timeType)
        else:
            if with_times is False:
                raise ValueError("Circuit %d contains time information and 'with_times=False'" % i)
            timeArray = _np.array(datarow_doc['time'], dataset.timeType)

        dataset._add_raw_arrays(circuit, oliArray, timeArray, countArray,
                                overwrite_existing=True,
                                record_zero_counts=record_zero_counts,
                                aux=datarow_doc.get('aux', {}))

    dataset.done_adding_data()
    return dataset


def mongodb_collection_names(custom_collection_names=None):
    """
    Construct a dictionary listing the MongoDB collection names used by pyGSTi read/write methods.

    The keys of the returned dictionary correspond to types or broader categories of
    pyGSTi objects that can serialize themselves to a MongoDB instance.  The values give
    the corresponding MongoDB collection name used by that type.

    Use of such a dictionary is needed (as opposed to just giving each I/O call
    a collection name, for instance) because pyGSTi I/O calls often read or write
    a hierarchy of objects with different types (e.g., experiment designs, data sets,
    and results).

    Parameters
    ----------
    custom_collection_names : dict, optional
        Overrides the default collection names.  The keys of this dictionary must be a
        subset of the valid type-names (the keys of the return value when this argument is `None`).

    Returns
    -------
    dict
    """
    collection_names = {'edesigns': 'pygsti_experiment_designs',
                        'data': 'pygsti_protocol_data',
                        'results': 'pygsti_protocol_results'}
    if custom_collection_names is not None:
        for k, v in custom_collection_names.items():
            assert(k in collection_names), "%s is an invalid collection-type (key)!" % str(k)
            collection_names[k] = v
    return collection_names


def prepare_doc_for_existing_doc_check(doc, existing_doc, set_id=True, convert_tuples_to_lists=True):
    """
    Prepares a to-be inserted document for comparison with an existing document.

    Optionally (see parameters):
    1) sets _id of `doc` to that of `existing_doc`.  This is useful in cases where the _id
       field is redundant with other uniquely identifying fields in the document, and so inserted
       documents don't need to match this field.
    2) converts all of `doc`'s tuples to lists, as the existing_doc is typically read from a MongoDB
       which only stores lists and doesn't distinguish between lists and tuples.

    Parameters
    ----------
    doc : dict
        the document to prepare

    existing_doc : dict
        the existing document

    set_id : bool, optional
        when `True`, add an `'_id'` field to `doc` matching `existing_doc` when one
        is not already present.

    convert_tuples_to_lists : bool, optional
        when `True` convert all of the tuples within `doc` to lists.

    Returns
    -------
    dict
        the prepared document.
    """
    def tup_to_list(obj):
        if isinstance(obj, (tuple, list)):
            return [tup_to_list(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: tup_to_list(v) for k, v in obj.items()}
        else:
            return obj

    doc = doc.copy()
    if '_id' not in doc:  # Allow doc to lack an _id field and still be considered same as existing_doc
        doc['_id'] = existing_doc['_id']
    return tup_to_list(doc)


def recursive_compare_str(a, b, a_name='first obj', b_name='second obj', prefix="", diff_accum=None):
    """
    Compares two objects and generates a list of strings describing how they differ.

    Recursively traverses dictionaries, tuples, and lists.

    Parameters
    ----------
    a, b : object
        The objects to compare

    a_name, b_name : str, optional
        Names for the `a` and `b` objects for referencing them in the output strings.

    prefix : str, optional
        An optional prefix to the descriptions in the returned strings.

    diff_accum : list, optional
        An existing list that is accumulating difference-descriptions.
        `None` means start a new list.

    Returns
    -------
    list
        A list of strings, each describing a difference between the objects.
    """
    typ = type(a)
    if diff_accum is None:
        diff_accum = []

    if typ != type(b):
        diff_accum.append(prefix + f": are different types ({typ} vs {type(b)})")

    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for k in a_keys.intersection(b_keys):
            recursive_compare_str(a[k], b[k], a_name, b_name, f"{prefix}.{k}", diff_accum)
        for k in (b_keys - a_keys):  # keys missing from a
            diff_accum.append(f"{prefix}.{k}: missing from {a_name}")
        for k in (a_keys - b_keys):  # keys missing from b
            diff_accum.append(f"{prefix}.{k}: missing from {b_name}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diff_accum.append(f"{prefix}: have different lengths ({len(a)} vs {len(b)})")
        else:
            for i, (va, vb) in enumerate(zip(a, b)):
                recursive_compare_str(va, vb, a_name, b_name, f"{prefix}[{i}]", diff_accum)
    else:
        if a != b:
            diff_accum.append(f"{prefix}: {a} != {b}")
    return diff_accum

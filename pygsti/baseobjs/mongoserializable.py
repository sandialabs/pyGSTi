"""
Defines the MongoSerializable class
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import importlib as _importlib
import numpy as _np

_indexinformation_cache = {}  # used to speedup mondodb index_information() queries


class MongoSerializable(object):
    """
    Base class for objects that can be serialized to a MongoDB database.

    At the very least, saving an object to a database creates a document in the
    collection named by the `collection_name` class variable (which can be overriden by
    derived classes).  Additionally, saving the object may create other documents outside
    of this collection (e.g., if the object contains MongoSerializable attributes that specify
    their own collection name).

    This interface also allows an object to save large chunks of data using, e.g.,
    MongoDB's GridFS system, when serializing itself instead of trying to write an
    enourmous JSON dictionary as a single document (as an object that is NicelySerializable
    might do).
    """
    collection_name = 'pygsti_objects'

    @classmethod
    def _create_obj_from_doc_and_mongodb(cls, doc, mongodb, **kwargs):
        """ Create a new object from the already-loaded main document `doc` and by access to the database `mongodb` """
        raise NotImplementedError("Class '%s' doesn't implement _create_obj_from_doc_and_mongodb!"
                                  % str(doc['module'] + '.' + doc['class']))

    def _add_auxiliary_write_ops_and_update_doc(self, doc, write_ops, mongodb, collection_name, overwrite_existing,
                                                **kwargs):
        """ Add to `write_ops` and update `doc` so that all of `self` 's data is serialized """
        raise NotImplementedError("Subclasses must implement this!")

    @classmethod
    def _remove_from_mongodb(cls, mongodb, collection_name, doc_id, session, recursive):
        """ Remove the records corresponding to the object with `doc_id`, minding the filter given by `recursive` """
        mongodb[collection_name].delete_one({'_id': doc_id}, session=session)  # just delete main document

    def __init__(self, doc_id=None):
        self._dbcoordinates = (self.collection_name, doc_id) if (doc_id is not None) else None
        #self.tags = {}  # FUTURE feature: save and load tags automatically?

    @classmethod
    def from_mongodb(cls, mongodb, doc_id, **kwargs):
        """
        Create and initialize an object from a MongoDB instance.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to load from.

        doc_id : bson.objecctid.ObjectId or dict
            The object ID or filter used to find a single object ID wihtin the database.  This
            document is loaded from the collection given by the `collection_name` attribute of
            this class.

        `**kwargs` : dict
            Additional keyword arguments poentially used by subclass implementations.  Any arguments
            allowed by a subclass's `_create_obj_from_doc_and_mongodb` method is allowed here.

        Returns
        -------
        object
        """
        # Implementation note:
        # This method is similar to _from_nice_serialization, but will defer to the method of a derived class
        # when once is specified in the state dictionary.  This method should thus be used when de-serializing
        # using a potential base class, i.e.  BaseClass._from_nice_serialization_base(state).
        # (This method should rarely need to be overridden in derived (sub) classes.)
        doc = _find_one_doc(mongodb, cls.collection_name, doc_id)
        if doc is None:
            raise ValueError(f"No document found in collection '{cls.collection_name}' identified by {doc_id}")
        return cls.from_mongodb_doc(mongodb, cls.collection_name, doc, **kwargs)

    @classmethod
    def from_mongodb_doc(cls, mongodb, collection_name, doc, **kwargs):
        """
        Create and initialize an object from a MongoDB instance and pre-loaded primary document.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to load from.

        collection_name : str
            The collection name within `mongodb` that `doc` was loaded from.  This is needed
            for the sole purpose of setting the created (returned) object's database "coordinates".

        doc : dict
            The already-retrieved main document for the object being loaded.  This takes the place
            of giving an identifier for this object.

        `**kwargs` : dict
            Additional keyword arguments poentially used by subclass implementations.  Any arguments
            allowed by a subclass's `_create_obj_from_doc_and_mongodb` method is allowed here.

        Returns
        -------
        object
        """
        if doc['module'] == cls.__module__ and doc['class'] == cls.__name__:
            # then the state is actually for this class and we should call its _from_nice_serialization method:
            ret = cls._create_obj_from_doc_and_mongodb(doc, mongodb, **kwargs)
        else:
            c = cls._doc_class(doc)
            if not issubclass(c, cls):
                raise ValueError("Class '%s' is trying to load a serialized '%s' object (not a subclass)!"
                                 % (cls.__module__ + '.' + cls.__name__, doc['module'] + '.' + doc['class']))
            ret = c._create_obj_from_doc_and_mongodb(doc, mongodb, **kwargs)
        ret._dbcoordinates = (collection_name, doc['_id'])
        return ret

    def write_to_mongodb(self, mongodb, session=None, overwrite_existing=False, **kwargs):
        """
        Write this object to a MongoDB database.

        The collection name used is `self.collection_name`, and the `_id` is either:
        1) the ID used by a previous write or initial read-in, if one exists, OR
        2) a new random `bson.objectid.ObjectId`

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to write data to.

        session : pymongo.client_session.ClientSession, optional
            MongoDB session object to use when interacting with the MongoDB
            database. This can be used to implement transactions among other things.

        overwrite_existing : bool, optional
            Whether existing documents should be overwritten.  The default of `False` causes
            a ValueError to be raised if a document with the given `_id` already exists
            and is different from what is being written.

        `**kwargs` : dict
            Additional keyword arguments poentially used by subclass implementations.  Any arguments
            allowed by a subclass's `_add_auxiliary_write_ops_and_update_doc` method is allowed here.

        Returns
        -------
        bson.objectid.ObjectId
            The identifier (`_id` value) of the main document that was written.
        """
        write_ops = WriteOpsByCollection(session)
        my_id = self.add_mongodb_write_ops(write_ops, mongodb, overwrite_existing, **kwargs)
        write_ops.execute(mongodb)
        return my_id

    def add_mongodb_write_ops(self, write_ops, mongodb, overwrite_existing=False, **kwargs):
        """
        Accumulate write and update operations for writing this object to a MongoDB database.

        Similar to :meth:`write_to_mongodb` but collects write operations instead of actually
        executing any write operations on the database.  This function may be preferred to
        :meth:`write_to_mongodb` when this object is being written as a part of a larger entity
        and executing write operations is saved until the end.

        As in :meth:`write_to_mongodb`, `self.collection_name` is the collection name and `_id` is either:
        1) the ID used by a previous write or initial read-in, if one exists, OR
        2) a new random `bson.objectid.ObjectId`

        Parameters
        ----------
        write_ops : WriteOpsByCollection
            An object that keeps track of `pymongo` write operations on a per-collection
            basis.  This object accumulates write operations to be performed at some point
            in the future.

        mongodb : pymongo.database.Database
            The MongoDB instance to write data to.

        overwrite_existing : bool, optional
            Whether existing documents should be overwritten.  The default of `False` causes
            a ValueError to be raised if a document with the given `_id` already exists
            and is different from what is being written.

        `**kwargs` : dict
            Additional keyword arguments poentially used by subclass implementations.  Any arguments
            allowed by a subclass's `_add_auxiliary_write_ops_and_update_doc` method is allowed here.

        Returns
        -------
        bson.objectid.ObjectId
            The identifier (`_id` value) of the main document that was written.
        """
        from bson.objectid import ObjectId as _ObjectId
        collection_name = self.collection_name if (self._dbcoordinates is None) else self._dbcoordinates[0]
        doc_id = _ObjectId() if (self._dbcoordinates is None) else self._dbcoordinates[1]
        doc = {'_id': doc_id,
               'module': self.__class__.__module__,
               'class': self.__class__.__name__,
               'version': 0}
        self._add_auxiliary_write_ops_and_update_doc(doc, write_ops, mongodb, collection_name,
                                                     overwrite_existing, **kwargs)
        write_ops.add_one_op(collection_name, {'_id': doc_id}, doc, overwrite_existing, mongodb)
        self._dbcoordinates = (collection_name, doc_id)
        return doc_id

    def remove_me_from_mongodb(self, mongodb, session=None, recursive="default"):
        assert self._dbcoordinates is not None, "Cannot remove object that wasn't loaded from or written to DB!"
        collection_name, doc_id = self._dbcoordinates[0], self._dbcoordinates[1]
        recursive = RecursiveRemovalSpecification.cast(recursive, self.__class__)
        self._remove_from_mongodb(mongodb, collection_name, doc_id, session, recursive)

    @classmethod
    def remove_from_mongodb(cls, mongodb, doc_id, collection_name=None, session=None, recursive="default"):
        """
        Remove the documents corresponding to an instance of this class from a MongoDB database.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to remove documents from.

        doc_id : bson.objectid.ObjectId
            The identifier of the root document stored in the database.

        collection_name : str, optional
            the MongoDB collection within `mongodb` where the main document resides.
            If `None`, then `<this_class>.collection_name` is used (which is usually
            what you want).

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
        None
        """
        if collection_name is None:
            collection_name = cls.collection_name
        doc = mongodb[collection_name].find_one({'_id': doc_id}, ('module', 'class'))
        if doc is None:
            return
        elif doc['module'] == cls.__module__ and doc['class'] == cls.__name__:
            recursive = RecursiveRemovalSpecification.cast(recursive, cls)
            cls._remove_from_mongodb(mongodb, collection_name, doc_id, session, recursive)
        else:
            c = cls._doc_class(doc)
            recursive = RecursiveRemovalSpecification.cast(recursive, c)
            if not issubclass(c, cls):
                raise ValueError("Class '%s' is trying to remove a serialized '%s' object (not a subclass)!"
                                 % (cls.__module__ + '.' + cls.__name__, doc['module'] + '.' + doc['class']))
            c._remove_from_mongodb(mongodb, collection_name, doc_id, session, recursive)

    @classmethod
    def _doc_class(cls, doc, check_is_subclass=True):
        """ Returns the class specified by the given db document"""
        m = _importlib.import_module(doc['module'])
        c = getattr(m, doc['class'])  # will raise AttributeError if class cannot be found
        if check_is_subclass and not issubclass(c, cls):
            raise ValueError("Expected a subclass or instance of '%s' but state dict has '%s'!"
                             % (cls.__module__ + '.' + cls.__name__, doc['module'] + '.' + doc['class']))
        return c


class WriteOpsByCollection(dict):
    """
    A dictionary of `pymongo` write operations stored on a per-collection basis.

    A dictionary with collection-name (string) keys and where each value is a list of
    `pymongo.InsertOne` and `pymongo.ReplaceOne` operations to be performed on the
    named collection at some later (unpsecified) time.

    The collection names (keys) can be restricted to a predefined set if desired.
    """
    def __init__(self, session=None, allowed_collection_names=None):
        self.allowed_collection_names = allowed_collection_names
        if self.allowed_collection_names is not None:
            super().__init__({k: [] for k in self.allowed_collection_names})
        else:
            super().__init__()
        self.special_ops = []
        self.session = session

    def add_ops_by_collection(self, other_ops):
        """
        Merge the information from another :class:`WriteOpsByCollection` into this one.

        Parameters
        ----------
        other_ops : WriteOpsByCollection
            The dictionary of write operations to merge in.

        Returns
        -------
        None
        """
        assert isinstance(other_ops, WriteOpsByCollection)
        for k, v in other_ops.items():
            if self.allowed_collection_names is None and k not in self:
                self[k] = []
            self[k].extend(v)

    def add_one_op(self, collection_name, uid, doc, overwrite_existing, mongodb, check_local_ops=True):
        """
        Add a single write operation to this dictionary, if one is allowed and needed.

        Adds a `pymongo.InsertOne` or `pymongo.ReplaceOne` object to `self[collection_name]` in
        order to write a document `doc` with unique identifer `uid`.  If `overwrite_existing` is
        `False` then an error is raised if a document with `uid` already exists -- either in the
        database or within the list of existing write operations -- *and* the existing document
        doesn't match the document being written (`doc`).

        Parameters
        ----------
        collection_name : str
            The collection name (key) to add an operation to.

        uid : bson.object.ObjectId or dict
            Unique identifier for the document to write.

        doc : dict
            The document to write.

        overwrite_existing : bool, optional
            Whether existing documents should be overwritten.  The default of `False` causes
            a ValueError to be raised if a document with the given `uid` already exists
            *and* is different from `doc`.

        mongodb : pymongo.database.Database
            The MongoDB instance documents will eventually be written to.  This database is
            *not* written to, and used solely to test for existing documents so that insert
            vs. replace operations are chosen correctly.

        check_local_ops : bool, optional
            Whether the queued-up write operations contained within this :class:`WriteOpsByCollection`
            object are considered when testing for the existence of documents.  Leaving this set
            to `True` is the safe option.  Set this to `False` to get a performance increase when
            you're sure there's no possibility a document with `uid` could have been "written" to
            this `WriteOpsByCollection` since its initialization.

        Returns
        -------
        bson.objectid.ObjectId
            The identifier (`_id` value) of the document to be written or that already exists
            and matches the one being written.
        """
        from pymongo import InsertOne, ReplaceOne
        from bson.objectid import ObjectId as _ObjectId
        assert self.allowed_collection_names is None or collection_name in self.allowed_collection_names
        if self.allowed_collection_names is None and collection_name not in self:
            self[collection_name] = []

        existing_doc = None
        tried_to_find_existing_doc = False

        #Get or create an id for the document to be inserted or replaced (or left as is)
        if isinstance(uid, dict):
            if '_id' in uid:
                doc_id = uid['_id']
            else:
                existing_doc = _find_one_doc(mongodb, collection_name, uid) #, ('_id',))  # entire doc so can use later
                if existing_doc is None and check_local_ops:
                    existing_doc = self._find_one_local_doc(collection_name, uid)
                doc_id = _ObjectId() if (existing_doc is None) else existing_doc['_id']
                tried_to_find_existing_doc = True
        else:
            assert isinstance(uid, _ObjectId)
            doc_id = uid

        assert ('_id' not in doc) or (doc['_id'] == doc_id)
        doc['_id'] = doc_id

        if overwrite_existing is True:
            self[collection_name].append(ReplaceOne({'_id': doc_id}, doc, upsert=True))
            #mongodb[collection_name].replace_one(doc_id, doc, upsert=True)  # alt line for DEBUG
        else:
            if not tried_to_find_existing_doc:  # then try now
                existing_doc = mongodb[collection_name].find_one(doc_id)
            if existing_doc is None and check_local_ops:
                existing_doc = self._find_one_local_doc(collection_name, doc_id)
                if existing_doc is not None:
                    existing_doc = prepare_doc_for_existing_doc_check(existing_doc, existing_doc)  # eg can have tuples

            if existing_doc is None:  # then insert the document as given
                self[collection_name].append(InsertOne(doc))
                #mongodb[collection_name].insert_one(doc)  # alt line for DEBUG
            else:
                #print("Found existing doc: ", doc.get('module','?'), doc.get('class','?'), doc.get('circuit_str','?'))
                existing_to_chk = prepare_doc_for_existing_doc_check(existing_doc, None, False, False, False)
                info_to_check = prepare_doc_for_existing_doc_check(doc, existing_doc)
                if existing_to_chk != info_to_check:
                    # overwrite_existing=False and a doc already exists AND docs are *different* => error
                    diff_lines = recursive_compare_str(existing_to_chk, info_to_check, 'existing doc', 'doc to insert')
                    raise ValueError("*Different* document with %s exists and `overwrite_existing=False`:\n%s"
                                     % (str(uid), '\n'.join(diff_lines)))
                # else do nothing, since doc exists and matches what we want to write => no error
        return doc_id

    def add_gridfs_put_op(self, collection_name, doc_id, binary_data, overwrite_existing, mongodb):
        """
        Add a GridFS put operation to this dictionary of write operations.

        This is a special type of operation for placing large chunks of binary data into a MongoDB.
        Arguments are similar to :meth:`add_one_op`.
        """
        import gridfs as _gridfs
        fs = _gridfs.GridFS(mongodb, collection=collection_name)
        if fs.exists(doc_id) and not overwrite_existing:
            pass  # don't check for equality here -- too slow since this is large data; just don't overwrite
        else:  # either the file doesn't exist or overwrite_existing=True, so write it!
            self.special_ops.append({'type': 'GridFS_put',
                                     'collection_name': collection_name,
                                     'data': binary_data,
                                     'overwrite_existing': overwrite_existing,
                                     'id': doc_id})
        return doc_id

    def execute(self, mongodb):
        """
        Execute all of the "queued" operations within this dictionary on a MongoDB instance.

        Note that `mongodb` should be the same as the `mongodb` given to any :meth:`add_one_op` and
        :meth:`add_gridfs_put_op` method calls.  The session given at the initialization of
        this object is used for these write operations.  On exit, this dictionary is empty, indicating
        there are no more queued operations.

        Parameters
        ----------
        mongodb : pymongo.database.Database
            The MongoDB instance to execute write operations on.

        Returns
        -------
        None
        """
        for op_info in self.special_ops:
            if op_info['type'] == 'GridFS_put':
                import gridfs as _gridfs
                fs = _gridfs.GridFS(mongodb, collection=op_info['collection_name'])
                try:
                    file_id = fs.put(op_info['data'], _id=op_info['id'])
                except _gridfs.errors.FileExists as e:
                    if op_info['overwrite_existing']:
                        fs.delete(op_info['id'])
                        file_id = fs.put(op_info['data'], _id=op_info['id'])  # try again
                    else:
                        raise e

                assert file_id == op_info['id']
            else:
                raise ValueError("Invalid special op:" + str(op_info))

        for collection_name, ops in self.items():
            if len(ops) > 0:  # bulk_write fails if ops is an empty list
                mongodb[collection_name].bulk_write(ops, session=self.session)
        self.clear()
        self.special_ops = []

    def _find_one_local_doc(self, collection_name, uid):
        if not isinstance(uid, dict): uid = {'_id': uid}
        for op in self.get(collection_name, []):
            doc = op._doc  # Warning: using *private* attribute of InsertOne & ReplaceOne objects
            for key, val in uid.items():
                if (key not in doc) or doc[key] != val:
                    break  # doc does not match uid
            else:  # doc *does* match uid
                return doc
        return None


def prepare_doc_for_existing_doc_check(doc, existing_doc, set_id=True, convert_tuples_to_lists=True,
                                       convert_numpy_dtypes=True, round_to_sigfigs=6):
    """
    Prepares a to-be inserted document for comparison with an existing document.

    Optionally (see parameters):
    1. sets _id of `doc` to that of `existing_doc` .  This is useful in cases where the _id
       field is redundant with other uniquely identifying fields in the document, and so inserted
       documents don't need to match this field.
    2. converts all of `doc` 's tuples to lists, as the existing_doc is typically read from a MongoDB
       which only stores lists and doesn't distinguish between lists and tuples.
    3. converts numpy datatypes to native python types
    4. rounds floating point values

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

    convert_numpy_dtypes : bool, optional
        when `True` convert numpy data types to native python types, e.g. np.float64 -> float.

    Returns
    -------
    dict
        the prepared document.
    """
    def process(obj):
        if isinstance(obj, list):
            return [process(v) for v in obj]
        elif isinstance(obj, tuple):
            if convert_tuples_to_lists:
                return [process(v) for v in obj]  # changes tuple -> list
            else:
                return tuple((process(v) for v in obj))
        elif isinstance(obj, dict):
            return {k: process(v) for k, v in obj.items()}
        elif isinstance(obj, _np.generic) and convert_numpy_dtypes:
            obj = obj.item()

        if isinstance(obj, float) and round_to_sigfigs:
            obj = float(('%.' + str(round_to_sigfigs) + 'g') % obj)
        return obj

    doc = doc.copy()
    if '_id' not in doc and set_id:  # Allow doc to lack an _id field and still be considered same as existing_doc
        doc['_id'] = existing_doc['_id']
    return process(doc)


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
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diff_accum.append(f"{prefix}: have different lengths ({len(a)} vs {len(b)})")
        else:
            for i, (va, vb) in enumerate(zip(a, b)):
                recursive_compare_str(va, vb, a_name, b_name, f"{prefix}[{i}]", diff_accum)
    else:
        if a != b:
            diff_accum.append(f"{prefix}: {a} != {b}")
    return diff_accum


def _find_one_doc(db, collection_name, doc_id, projection=None, error_if_no_doc=False):
    if doc_id is not None and not isinstance(doc_id, dict):
        doc_id = {'_id': doc_id}

    if '_id' not in doc_id:  # otherwise can bypass since if _id is specified a doc must be unique
        #Check to see if uid is a single lookup that corresponds to a *unique* index
        doc_id_known_to_be_unique = False
        doc_id_fields = set(doc_id.keys())

        if (id(db), collection_name) not in _indexinformation_cache:  # cache index_information call for speed
            _indexinformation_cache[id(db), collection_name] = db[collection_name].index_information()
        for index_name, index_info in _indexinformation_cache[id(db), collection_name].items():
            if index_info.get('unique', False) and doc_id_fields.issubset(set([k for k, d in index_info['key']])):
                doc_id_known_to_be_unique = True
                break
        if not doc_id_known_to_be_unique:  # then check that this doc_id is indeed unique with a db call
            # for DEBUG: print("UID without _id and not known unique: ", doc_id, " for ", collection_name)
            if db[collection_name].count_documents(doc_id, limit=2) > 1:
                raise ValueError((f"Multiple records where identified by the given `doc_id` ({doc_id})."
                                  " `doc_id` must specify exactly one record."))

    single_doc = db[collection_name].find_one(doc_id, projection)
    if single_doc is None and error_if_no_doc:
        raise ValueError(f"Could not find document specified by {doc_id}!")
    return single_doc


class RecursiveRemovalSpecification(object):
    """
    Specifies which types of objects to remove when performing a recursive removal of MongoDB documents.

    Parameters
    ----------
    edesigns : bool, optional
        Whether experiment designs are allowed to be removed.

    data : bool, optional
        Whether protocol data objects are allowed to be removed.

    results : bool, optional
        Whether result objects and directories are allowed to be removed.

    circuits : bool, optional
        Whether circuit objects stored in common collections (e.g. `"pygsti_circuits"`) are allowed to be removed.

    protocols : bool, optional
        Whether protocol objects are allowed to be removed.

    children : bool, optional
        Whether child objects of :class:`TreeNode` objects are allowed to be removed.
    """

    @classmethod
    def cast(cls, obj, root_cls_being_deleted=None):
        """
        Create a :class:`RecursiveRemovalSpecification` object from another object.

        If `obj` is already a :class:`RecursiveRemovalSpecification` object then it is
        just returned directly.  Otherwise, `obj` can be a string, boolean value, or `None`:

        - False, None, or `"none"`: no objects are removed recursively (the safe option)
        - True or `"all"`: all recursive removal operation are permitted (the un-safe option)
        - `"default"`: only recursive removal of the type being removed is allowed.  For example,
                       when removing a `ProtocolData` object, it and any child `ProtocolData` objects
                       are removed but experiment designs are not, nor are (potentially shared) circuits.
        - `"upstream"`: recursive removal of the type being removed and "upstream" types is allowed.  "Upstream"
                        refers to items closer to the front of the list: experiment designs, data objects, result
                        objects. For example, when removing a `ProtocolData` object, it and any child `ProtocolData`
                        objects, along with their experiment designs are removed, but circuit objects are not.
        - `"all_but_circuits"`: everything but circuit objects are allowed to be removed.  Circuit objects are
                                treated specially because they are very likely to be re-used (shared).

        Parameters
        ----------
        obj : object
            The object to convert.  See description above.

        root_cls_being_deleted : class, optional
            The Python class of the main object being deleted.  This additional information is needed
            when (and only when) `obj == "default"`.

        Returns
        -------
        RecursiveRemovalSpecification
        """
        from pygsti.protocols import ExperimentDesign as _ExperimentDesign, ProtocolData as _ProtocolData
        from pygsti.protocols import ProtocolResults as _ProtocolResults, ProtocolResultsDir as _ProtocolResultsDir
        if isinstance(obj, RecursiveRemovalSpecification):
            return obj
        elif obj is False or obj is None or obj == "none":
            return cls()  # defaults are all to *not* remove objects
        elif obj is True or obj == "all":
            return cls(True, True, True, True, True, True)
        elif obj == "default":
            if root_cls_being_deleted is None:
                raise ValueError("Must supply `root_cls_being_deleted` when using 'default' recursive removal spec!")
            if issubclass(root_cls_being_deleted, _ExperimentDesign):
                return cls(edesigns=True, children=True)
            elif issubclass(root_cls_being_deleted, _ProtocolData):
                return cls(data=True, children=True)
            elif issubclass(root_cls_being_deleted, (_ProtocolResults, _ProtocolResultsDir)):
                return cls(results=True, children=True)
            else:
                return cls()
        elif obj == "default+":
            if root_cls_being_deleted is None:
                raise ValueError("Must supply `root_cls_being_deleted` when using 'default' recursive removal spec!")
            if issubclass(root_cls_being_deleted, _ExperimentDesign):
                return cls(edesigns=True, children=True)
            elif issubclass(root_cls_being_deleted, _ProtocolData):
                return cls(data=True, edesigns=True, children=True)
            elif issubclass(root_cls_being_deleted, (_ProtocolResults, _ProtocolResultsDir)):
                return cls(results=True, data=True, edesigns=True, protocols=True, children=True)
            else:
                return cls()
        elif obj == "all_but_circuits":
            return cls(edesigns=True, data=True, results=True, circuits=False, protocols=True, children=True)
        else:
            raise ValueError("Could not cast %s to a RecursiveRemovalSpecification!" % str(obj))

    def __init__(self, edesigns=False, data=False, results=False, circuits=False, protocols=False, children=False):
        self.edesigns = edesigns
        self.data = data
        self.results = results
        self.circuits = circuits
        self.protocols = protocols
        self.children = children

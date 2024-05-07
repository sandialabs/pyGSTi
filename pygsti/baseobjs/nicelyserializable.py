"""
Defines the NicelySerializable class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import importlib as _importlib
import pathlib as _pathlib
import json as _json
import numpy as _np
import scipy.sparse as _sps
import warnings as _warnings
from pygsti.baseobjs.mongoserializable import MongoSerializable

class_location_changes = {}  # (module, class) mapping from OLD to NEW locations


class NicelySerializable(MongoSerializable):
    """
    The base class for all "nicely serializable" objects in pyGSTi.

    A "nicely serializable" object can be converted to and created from a
    native Python object (like a string or dict) that contains only other native
    Python objects.  In addition, there are constraints on the makeup of these
    objects so that they can be easily serialized to standard text-based formats,
    e.g. JSON.  For example, dictionary keys must be strings, and the list vs. tuple
    distinction cannot be assumed to be preserved during serialization.
    """

    @classmethod
    def read(cls, path, format=None):
        """
        Read an object of this type, or a subclass of this type, from a file.

        Parameters
        ----------
        path : str or Path or file-like
            The filename to open or an already open input stream.

        format : {'json', None}
            The format of the file.  If `None` this is determined automatically
            by the filename extension of a given path.

        Returns
        -------
        NicelySerializable
        """
        if format is None:
            if str(path).endswith('.json'):
                format = 'json'
            else:
                raise ValueError("Cannot determine format from extension of filename: %s" % str(path))

        with open(str(path), 'r') as f:
            return cls.load(f, format)

    @classmethod
    def load(cls, f, format='json'):
        """
        Load an object of this type, or a subclass of this type, from an input stream.

        Parameters
        ----------
        f : file-like
            An open input stream to read from.

        format : {'json'}
            The format of the input stream data.

        Returns
        -------
        NicelySerializable
        """
        if format == 'json':
            state = _json.load(f)
        else:
            raise ValueError("Invalid `format` value: %s" % str(format))
        return cls.from_nice_serialization(state)

    @classmethod
    def loads(cls, s, format='json'):
        """
        Load an object of this type, or a subclass of this type, from a string.

        Parameters
        ----------
        s : str
            The serialized object.

        format : {'json'}
            The format of the string data.

        Returns
        -------
        NicelySerializable
        """
        if format == 'json':
            state = _json.loads(s)
        else:
            raise ValueError("Invalid `format` value: %s" % str(format))
        return cls.from_nice_serialization(state)

    @classmethod
    def from_nice_serialization(cls, state):
        """
        Create and initialize an object from a "nice" serialization.

        A "nice" serialization here means one created by a prior call to `to_nice_serialization` using this
        class or a subclass of it.  Nice serializations adhere to additional rules (e.g. that dictionary
        keys must be strings) that make them amenable to common file formats (e.g. JSON).

        The `state` argument is typically a dictionary containing 'module' and 'state' keys specifying
        the type of object that should be created.  This type must be this class or a subclass of it.

        Parameters
        ----------
        state : object
            An object, usually a dictionary, representing the object to de-serialize.

        Returns
        -------
        object
        """
        # Implementation note:
        # This method is similar to _from_nice_serialization, but will defer to the method of a derived class
        # when once is specified in the state dictionary.  This method should thus be used when de-serializing
        # using a potential base class, i.e.  BaseClass._from_nice_serialization_base(state).
        # (This method should rarely need to be overridden in derived (sub) classes.)
        if isinstance(state, dict) and state['module'] == cls.__module__ and state['class'] == cls.__name__:
            # then the state is actually for this class and we should call its _from_nice_serialization method:
            ret = cls._from_nice_serialization(state)
        else:
            # otherwise, this call functions as a base class call that defers to the correct derived class
            ret = NicelySerializable._from_nice_serialization.__func__(cls, state)

        if 'dbcoordinates' in state:  # set ._dbcoordinates from a 'dbcoordinates' state value
            try:
                from bson.objectid import ObjectId as _ObjectId
                ret._dbcoordinates = (state['dbcoordinates'][0], _ObjectId(state['dbcoordinates'][1]))
            except ImportError:  # in case bson isn't installed
                _warnings.warn("Could not read-in database coordinates because `bson` package is not installed.")
                ret._dbcoordinates = None
        else:
            ret._dbcoordinates = None

        return ret

    def to_nice_serialization(self):
        """
        Serialize this object in a way that adheres to "niceness" rules of common text file formats.

        Returns
        -------
        object
            Usually a dictionary representing the serialized object, but may also be another native
            Python type, e.g. a string or list.
        """
        state = self._to_nice_serialization()
        if self._dbcoordinates is not None and isinstance(state, dict):
            # HACK: the code below adds the special dbcoordinates attribute to `state` when we're serializing to
            # something other than a MongoDB (which holds the dbcoordinates intrinsically in the document's
            # location within the database and '_id' field).  The problem is that to_nice_serialization can't easily
            # tell whether it's being called as part of a MongoDB serialization or not.  In the future we could plumb
            # a flag or something else that determines this, but for now we add this fragile HACK that checks the
            # caller names for 'add_mongodb_write_ops' (only in the call stack when we're serializing to MongoDB).
            import inspect
            outer_function_names = set([outer.function for outer in inspect.getouterframes(inspect.currentframe())])
            if 'add_mongodb_write_ops' not in outer_function_names:
                state['dbcoordinates'] = (self._dbcoordinates[0], str(self._dbcoordinates[1]))  # ObjectId -> str
            # Note: don't do this in _to_nice_serialization, which is also used to for mongodb
            #  methods, and we don't want dbcoordinates in DB documents
        return state

    def write(self, path, **format_kwargs):
        """
        Writes this object to a file.

        Parameters
        ----------
        path : str or Path
            The name of the file that is written.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        None
        """
        from pygsti.io.metadir import _check_jsonable
        if str(path).endswith('.json'):
            format = 'json'
        else:
            raise ValueError("Cannot determine format from extension of filename: %s" % str(path))

        with open(str(path), 'w') as f:
            self.dump(f, format)

    def dump(self, f, format='json', **format_kwargs):
        """
        Serializes and writes this object to a given output stream.

        Parameters
        ----------
        f : file-like
            A writable output stream.

        format : {'json', 'repr'}
            The format to write.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        None
        """
        assert(f is not None), "Must supply a valid `f` argument!"
        self._dump_or_dumps(f, format, **format_kwargs)

    def dumps(self, format='json', **format_kwargs):
        """
        Serializes this object and returns it as a string.

        Parameters
        ----------
        format : {'json', 'repr'}
            The format to write.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        str
        """
        return self._dump_or_dumps(None, format, **format_kwargs)

    def _dump_or_dumps(self, f, format='json', **format_kwargs):
        """
        Serializes and writes this object to a given output stream.

        Parameters
        ----------
        f : file-like
            A writable output stream.  If `None`, then object is written
            as a string and returned.

        format : {'json', 'repr'}
            The format to write.

        format_kwargs : dict, optional
            Additional arguments specific to the format being used.
            For example, the JSON format accepts `indent` as an argument
            because `json.dump` does.

        Returns
        -------
        str or None
            If `f` is None, then the serialized object as a string is returned.  Otherwise,
            `None` is returned.
        """
        from pygsti.io.metadir import _check_jsonable

        if format == 'json':
            if 'indent' not in format_kwargs:  # default indent=4 JSON argument
                format_kwargs = format_kwargs.copy()  # don't update caller's dict!
                format_kwargs['indent'] = 4

            json_dict = self.to_nice_serialization()  # use non-underscore version so allows dbcoordinates
            _check_jsonable(json_dict)
            if f is not None:
                _json.dump(json_dict, f, **format_kwargs)
            else:
                return _json.dumps(json_dict, **format_kwargs)
        else:
            raise ValueError("Invalid `format` argument: %s" % str(format))

    def _to_nice_serialization(self):
        state = {'module': self.__class__.__module__,
                 'class': self.__class__.__name__,
                 'version': 0}
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        c = cls._state_class(state)
        if not issubclass(c, cls):
            raise ValueError("Class '%s' is trying to load a serialized '%s' object (not a subclass)!"
                             % (cls.__module__ + '.' + cls.__name__, state['module'] + '.' + state['class']))
        implementing_cls = cls
        for candidate_cls in c.__mro__:
            if '_from_nice_serialization' in candidate_cls.__dict__:
                implementing_cls = candidate_cls; break

        if implementing_cls == cls:  # then there's no actual derived-class implementation to call!
            raise NotImplementedError("Class '%s' doesn't implement _from_nice_serialization!"
                                      % str(state['module'] + '.' + state['class']))
        else:
            return c._from_nice_serialization(state)

    @classmethod
    def _state_class(cls, state, check_is_subclass=True):
        """ Returns the class specified by the given state dictionary"""
        if (state['module'], state['class']) in class_location_changes:
            state['module'], state['class'] = class_location_changes[state['module'], state['class']]
        try:
            m = _importlib.import_module(state['module'])
            c = getattr(m, state['class'])  # will raise AttributeError if class cannot be found
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(("Class or module not found when instantiating a NicelySerializable"
                               f" {state['module']}.{state['class']} object!  If this class has"
                               " moved, consider adding (module, classname) mapping to"
                               " pygsti.baseobjs.nicelyserializable.class_location_changes dict")) from e

        if check_is_subclass and not issubclass(c, cls):
            raise ValueError("Expected a subclass or instance of '%s' but state dict has '%s'!"
                             % (cls.__module__ + '.' + cls.__name__, state['module'] + '.' + state['class']))
        return c

    @classmethod
    def _check_compatible_nice_state(cls, state):
        if (state['module'] != cls.__module__
           or state['class'] != cls.__name__):
            raise ValueError("Nice serialization type mismatch: %s != %s"
                             % (state['module'] + '.' + state['class'],
                                cls.__module__ + "." + cls.__name__))

    @classmethod
    def _encodemx(cls, mx):
        if mx is None:
            return None
        elif _sps.issparse(mx):
            csr_mx = _sps.csr_matrix(mx)  # convert to CSR and save in this format
            return {'sparse_matrix_type': 'csr',
                    'data': cls._encodemx(csr_mx.data), 'indices': cls._encodemx(csr_mx.indices),
                    'indptr': cls._encodemx(csr_mx.indptr), 'shape': csr_mx.shape}
        else:
            enc = str if _np.iscomplexobj(mx) else \
                ((lambda x: int(x)) if (mx.dtype == _np.int64) else (lambda x: x))
            encoded = _np.array([enc(x) for x in mx.flat])
            return encoded.reshape(mx.shape).tolist()

    @classmethod
    def _decodemx(cls, mx):
        if mx is None:
            decoded = None
        elif isinstance(mx, dict):  # then a sparse mx
            assert (mx['sparse_matrix_type'] == 'csr')
            data = cls._decodemx(mx['data'])
            indices = cls._decodemx(mx['indices'])
            indptr = cls._decodemx(mx['indptr'])
            decoded = _sps.csr_matrix((data, indices, indptr), shape=mx['shape'])
        else:
            basemx = _np.array(mx)
            if basemx.dtype.kind == 'U':  # character type array => complex numbers as strings
                decoded = _np.array([complex(x) for x in basemx.flat])
                decoded = decoded.reshape(basemx.shape)
            else:
                decoded = basemx
        return decoded

    @classmethod
    def _encodevalue(cls, val):
        return str(val) if _np.iscomplexobj(val) else \
            (int(val) if isinstance(val, _np.int64) else val)

    @classmethod
    def _decodevalue(cls, val):
        if isinstance(val, str):
            return complex(val)
        else:
            return val

    @classmethod
    def _create_obj_from_doc_and_mongodb(cls, doc, mongodb):
        #Ignore mongodb, just init from doc:
        return cls.from_nice_serialization(doc)

    def _add_auxiliary_write_ops_and_update_doc(self, doc, write_ops, mongodb, collection_name, overwrite_existing):
        doc.update(self._to_nice_serialization())  # use _to... so we don't have 'dbcoordinates'

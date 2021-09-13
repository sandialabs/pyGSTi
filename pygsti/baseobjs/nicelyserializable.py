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


class NicelySerializable(object):
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
            return cls._from_nice_serialization(state)
        else:
            # otherwise, this call functions as a base class call that defers to the correct derived class
            return NicelySerializable._from_nice_serialization.__func__(cls, state)

    def to_nice_serialization(self):
        """
        Serialize this object in a way that adheres to "niceness" rules of common text file formats.

        Returns
        -------
        object
            Usually a dictionary representing the serialized object, but may also be another native
            Python type, e.g. a string or list.
        """
        return self._to_nice_serialization()

    def write(self, path, **format_kwargs):
        """
        Writes this object to a file.

        Parameters
        ----------
        path : str or Path
            The name of the file that is written.

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
        from pygsti.io.metadir import _check_jsonable
        if str(path).endswith('.json'):
            format = 'json'
        else:
            raise ValueError("Cannot determine format from extension of filename: %s" % str(filename))

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

            json_dict = self._to_nice_serialization()
            _check_jsonable(json_dict)
            if f is not None:
                _json.dump(json_dict, f, **format_kwargs)
            else:
                return _json.dumps(json_dict, **format_kwargs)
        else:
            raise ValueError("Invalid `format` argument: %s" % str(format))

    def _to_nice_serialization(self):
        state = {'module': self.__class__.__module__,
                 'class': self.__class__.__name__}
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
        m = _importlib.import_module(state['module'])
        c = getattr(m, state['class'])  # will raise AttributeError if class cannot be found
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
        if _sps.issparse(mx):
            csr_mx = _sps.csr_matrix(mx)  # convert to CSR and save in this format
            return {'sparse_matrix_type': 'csr',
                    'data': cls._encodemx(csr_mx.data), 'indices': cls._encodemx(csr_mx.indices),
                    'indptr': cls._encodemx(csr_mx.indptr), 'shape': csr_mx.shape}
        else:
            enc = str if _np.iscomplexobj(mx) else (lambda x: x)
            encoded = _np.array([enc(x) for x in mx.flat])
            return encoded.reshape(mx.shape).tolist()

    @classmethod
    def _decodemx(cls, mx):
        if isinstance(mx, dict):  # then a sparse mx
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

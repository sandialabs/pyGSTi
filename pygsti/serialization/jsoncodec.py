"""
Defines JSON-format encoding and decoding functions
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import base64 as _base64
import collections as _collections
import importlib as _importlib
import types as _types
import uuid as _uuid
import numpy as _np
import scipy.sparse as _sps


def _class_hasattr(instance, attr):
    """
    Helper function for checking if `instance.__class__` has an attribute

    Parameters
    ----------
    instance : obj
        instance to check

    attr : str
        attribute name

    Returns
    -------
    bool
    """
    return hasattr(instance.__class__, attr)


def encode_obj(py_obj, binary):
    """
    Returns JSON-compatible version of `py_obj`.

    Constructs in-memory a JSON-format-compatible copy of the Python object
    `py_obj`, handling pyGSTi objects appropriately.  When `binary=False`,
    the output must contain only ASCII-compatible strings (no 'bytes'),
    otherwise the output is allowed to contain non-ASCII string values (OK for
    binary formats like MSGPACK and BSON).

    Parameters
    ----------
    py_obj : object
        The object to encode.

    binary : bool
        Whether the output is allowed to have binary-mode strings or not.

    Returns
    -------
    object
        A JSON-format compatible object.  Usually a dict, list, or string.
    """
    #print("ENCODING ", str(type(py_obj)))
    is_pygsti_obj = hasattr(py_obj, '__class__') and \
        hasattr(py_obj.__class__, '__module__') and \
        py_obj.__class__.__module__.startswith('pygsti')

    is_pygsti_class = isinstance(py_obj, type) and hasattr(py_obj, '__module__') \
        and py_obj.__module__.startswith('pygsti')

    is_plotly_fig = hasattr(py_obj, '__class__') and \
        hasattr(py_obj.__class__, '__module__') and \
        py_obj.__class__.__module__ == 'plotly.graph_objs._figure' and \
        py_obj.__class__.__name__ == "Figure"
    # just needed for v3 plotly where figures aren't dicts...

    # Pygsti class encoding
    if is_pygsti_class:  # or _class_hasattr(py_obj, '__pygsti_getstate__')
        return {'__pygsticlass__': (py_obj.__module__, py_obj.__name__)}

    # Pygsti object encoding
    elif is_pygsti_obj:  # or _class_hasattr(py_obj, '__pygsti_getstate__')

        #Get State (and/or init args)
        if _class_hasattr(py_obj, '__pygsti_reduce__'):
            red = py_obj.__pygsti_reduce__()  # returns class, construtor_args, state
            assert(callable(red[0]))
            init_args = red[1] if len(red) > 1 else []
            state = red[2] if len(red) > 2 else ()
            if state is None: state = ()
            if not isinstance(state, dict): state = {'__state_obj__': state}  # when state is, e.g, a tuple
            if red[0] is not py_obj.__class__:
                state['__init_fn__'] = (red[0].__module__, red[0].__name__)  # Note: 'object' type has module == None
            state.update({'__init_args__': init_args})
        elif _class_hasattr(py_obj, '__pygsti_getstate__'):
            state = py_obj.__pygsti_getstate__()  # must return a dict
        elif _class_hasattr(py_obj, '__getstate__'):
            state = py_obj.__getstate__()
        elif hasattr(py_obj, '__dict__'):
            state = py_obj.__dict__  # take __dict__ as state
        elif _class_hasattr(py_obj, '__reduce__'):
            red = py_obj.__reduce__()  # returns class, construtor_args, state
            if red[0] is not py_obj.__class__:
                state = None  # weird reducing can happen, for instance, for namedtuples - just punt
            else:
                init_args = red[1] if len(red) > 1 else []
                state = red[2] if len(red) > 2 else {}
                if state is None: state = {}
                state.update({'__init_args__': init_args})
        else:
            state = None

        if state is None:  # Note: __dict__ and __getstate__ may *return* None (python 2.7)
            if hasattr(py_obj, '_asdict'):  # named tuples
                state = {'__init_args__': list(py_obj._asdict().values())}
                # values will be ordered as per __init__ so no need for keys
            else:
                raise ValueError("Can't get state of %s object" % type(py_obj))

        d = {k: encode_obj(v, binary) for k, v in state.items()}

        #DEBUG (instead of above line)
        #import json as _json
        #d = {}
        #print("DB: Encoding state for pyGSTi %s object:" % type(py_obj))
        #for k,v in state.items():
        #    print(">>> Encoding key: ",k)
        #    d[k] = encode_obj(v,binary)
        #    print("<<< Done encoding key ",k)
        #    try: _json.dumps(d[k])
        #    except Exception as e:
        #        print("Cannot JSON %s key: " % k, d[k])
        #        raise e

        d.update({'__pygstiobj__': (py_obj.__class__.__module__,
                                    py_obj.__class__.__name__)})

        #Currently, don't add standard-base-class state
        #if we know how to __init__, since we'll assume this
        # should initialize the entire (base class included) instance
        encode_std_base = bool('__init_args__' not in d)

        if encode_std_base:
            std_encode = _encode_std_obj(py_obj, binary)
            if std_encode is not py_obj:  # if there's something to encode
                # this pygsti object is also a standard-object instance
                assert(isinstance(std_encode, dict))
                d['__std_base__'] = std_encode

        #try:
        #    _json.dumps(d)
        #except Exception as e:
        #    print("Cannot JSON ",type(py_obj))
        #    raise e

        return d

    #Special case: a plotly Figure object - these need special help being serialized
    elif is_plotly_fig and hasattr(py_obj, 'to_dict'):
        return {'__plotlyfig__': _encode_std_obj(py_obj.to_dict(), binary)}

    else:
        return _encode_std_obj(py_obj, binary)


def _encode_std_obj(py_obj, binary):
    """
    Helper to :func:`encode_obj` that encodes only "standard" (non-pyGSTi) types

    Parameters
    ----------
    py_obj : object
        standard Python object to encode

    binary : bool
        whether to use binary-mode strings

    Returns
    -------
    dict
    """
    # Other builtin or standard object encoding
    #print("Encoding std type: ",str(type(py_obj)))
    if isinstance(py_obj, tuple):
        return {'__tuple__': [encode_obj(v, binary) for v in py_obj]}
    elif isinstance(py_obj, list):
        return {'__list__': [encode_obj(v, binary) for v in py_obj]}
    elif isinstance(py_obj, set):
        return {'__set__': [encode_obj(v, binary) for v in py_obj]}
    elif isinstance(py_obj, slice):
        return {'__slice__': [encode_obj(py_obj.start, binary),
                              encode_obj(py_obj.stop, binary),
                              encode_obj(py_obj.step, binary)]}
    elif isinstance(py_obj, range):
        return {'__range__': (py_obj.start, py_obj.stop, py_obj.step)}
    elif isinstance(py_obj, _collections.OrderedDict):
        return {'__odict__': [(encode_obj(k, binary), encode_obj(v, binary))
                              for k, v in py_obj.items()]}
    elif isinstance(py_obj, _collections.Counter):
        return {'__counter__': [(encode_obj(k, binary), encode_obj(v, binary))
                                for k, v in dict(py_obj).items()]}
    elif isinstance(py_obj, dict):
        return {'__ndict__': [(encode_obj(k, binary), encode_obj(v, binary))
                              for k, v in py_obj.items()]}
    elif isinstance(py_obj, _uuid.UUID):
        return {'__uuid__': str(py_obj.hex)}
    elif isinstance(py_obj, complex):
        rep = py_obj.__repr__()  # a string
        data = _tobin(rep) if binary else rep  # binary if need be
        return {'__complex__': data}
    elif not binary and isinstance(py_obj, bytes):
        return {'__bytes__': _tostr(_base64.b64encode(py_obj))}
    elif binary and isinstance(py_obj, str):
        return {'__string__': _tobin(py_obj)}

    #Numpy encoding
    elif isinstance(py_obj, _np.ndarray):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if py_obj.dtype.kind == 'V':
            kind = 'V'
            descr = _tobin(py_obj.dtype.descr) if binary else _tostr(py_obj.dtype.descr)
        else:
            kind = py_obj.dtype.kind  # can be '' or 'O' (for object types)
            descr = _tobin(py_obj.dtype.str) if binary else _tostr(py_obj.dtype.str)

        if kind == 'O':
            #Special case of object arrays:  store flattened array data
            data = [encode_obj(el, binary) for el in py_obj.flat]
            assert(len(data) == _np.prod(py_obj.shape))
        else:
            data = py_obj.tobytes() if binary else _tostr(_base64.b64encode(py_obj.tobytes()))

        return {'__ndarray__': data,
                'dtype': descr,
                'kind': kind,
                'shape': py_obj.shape}

    #Scipy sparse matrix encoding
    elif isinstance(py_obj, _sps.csr_matrix):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if py_obj.dtype.kind == 'V':
            kind = 'V'
            descr = _tobin(py_obj.dtype.descr) if binary else _tostr(py_obj.dtype.descr)
        else:
            kind = py_obj.dtype.kind  # can be '' or 'O' (for object types)
            descr = _tobin(py_obj.dtype.str) if binary else _tostr(py_obj.dtype.str)
        if kind == 'O':
            raise TypeError("Cannot serialize sparse matrices of *objects*!")

        return {'__scipy_csrmatrix__': encode_obj(py_obj.data, binary),
                'indices': encode_obj(py_obj.indices, binary),
                'indptr': encode_obj(py_obj.indptr, binary),
                'dtype': descr,
                'kind': kind,
                'shape': py_obj.shape}

    elif isinstance(py_obj, (_np.bool_, _np.number)):
        data = py_obj.tobytes() if binary else _tostr(_base64.b64encode(py_obj.tobytes()))
        return {'__npgeneric__': data,
                'dtype': _tostr(py_obj.dtype.str)}

    elif isinstance(py_obj, _types.FunctionType):  # functions
        # OLD: elif callable(py_obj): #incorrectly includes pygsti classes w/__call__ (e.g. AutoGator)
        return {'__function__': (py_obj.__module__, py_obj.__name__)}

    return py_obj  # assume the bare py_obj is json-able


def decode_obj(json_obj, binary):
    """
    Inverse of :func:`encode_obj`.

    Decodes the JSON-compatible `json_obj` object into the original Python
    object that was encoded.

    Parameters
    ----------
    json_obj : object
        The JSON-compabtible object to decode.  Note that this is NOT a JSON
        string, but rather the object that would be decoded from such a string
        (by `json.loads`, for instance).

    binary : bool
        Whether `json_obj` is a binary format or not.  If so, then the decoding
        expects all strings to be binary strings i.e. `b'name'` instead of just
        `'name'`.  The value of this argument should match that used in the
        original call to :func:`encode_obj`.

    Returns
    -------
    object
        A Python object.
    """
    B = _tobin if binary else _ident

    if isinstance(json_obj, dict):
        if B('__pygsticlass__') in json_obj:
            modname, clsname = json_obj[B('__pygsticlass__')]
            module = _importlib.import_module(_tostr(modname))
            class_ = getattr(module, _tostr(clsname))
            return class_

        elif B('__pygstiobj__') in json_obj:
            #DEBUG
            #print("DB: creating %s" % str(json_obj['__pygstiobj__']))
            #print("DB: json_obj is type %s with keyvals:" % type(json_obj))
            #for k,v in json_obj.items():
            #    print("%s (%s): %s (%s)" % (k,type(k),v,type(v)))

            modname, clsname = json_obj[B('__pygstiobj__')]
            module = _importlib.import_module(_tostr(modname))
            class_ = getattr(module, _tostr(clsname))

            if B('__init_fn__') in json_obj:  # construct via this function instead of class_.__init__
                ifn_modname, ifn_fnname = decode_obj(json_obj[B('__init_fn__')], binary)
                if ifn_modname is None and ifn_fnname == "__new__":  # special behavior
                    initfn = class_.__new__
                else:
                    initfn = getattr(_importlib.import_module(_tostr(ifn_modname)), _tostr(ifn_fnname))
            else:
                initfn = class_  # just use the class a the callable initialization function

            if B('__init_args__') in json_obj:  # construct via __init__
                args = decode_obj(json_obj[B('__init_args__')], binary)
                instance = initfn(*args)

            else:  # init via __new__ and set state
                try:
                    instance = class_.__new__(class_)
                except Exception as e:
                    raise ValueError("Could not create class " + str(class_) + ": " + str(e))

            #Create state dict
            state_dict = {}
            for k, v in json_obj.items():
                if k in (B('__pygstiobj__'), B('__init_args__'), B('__std_base__')): continue
                state_dict[_tostr(k)] = decode_obj(v, binary)
            state_obj = state_dict.get('__state_obj__', state_dict)

            #Set state
            if _class_hasattr(instance, '__pygsti_setstate__'):
                instance.__pygsti_setstate__(state_obj)
            elif _class_hasattr(instance, '__setstate__'):
                instance.__setstate__(state_obj)
            elif hasattr(instance, '__dict__'):  # just update __dict__
                instance.__dict__.update(state_dict)
            elif len(state_dict) > 0:
                raise ValueError("Cannot set nontrivial state of %s object" % type(instance))

            #update instance with std-object info if needed (only if __init__ not called)
            if B('__std_base__') in json_obj:
                _decode_std_base(json_obj[B('__std_base__')], instance, binary)

            return instance

        elif B('__plotlyfig__') in json_obj:
            import plotly.graph_objs as go
            return go.Figure(decode_obj(json_obj[B('__plotlyfig__')], binary))

        else:
            return _decode_std_obj(json_obj, binary)
    else:
        return json_obj


def _decode_std_base(json_obj, start, binary):
    """
    Helper to :func:`decode_obj` for decoding pyGSTi objects that are derived from a standard type.

    Parameters
    ----------
    json_obj : dict
        json-loaded dict to decode from

    start : various
        Starting object that serves as a container for elements of the
        standard-Python base class (e.g. a list).

    binary : bool
        Whether or not to use binary-mode strings as dict keys.

    Returns
    -------
    object
    """
    B = _tobin if binary else _ident

    if B('__tuple__') in json_obj:
        #OK if __init_args since this means we knew how to construct it (e.g. namedtuples)
        assert(B('__init_args') in json_obj), "No support for sub-classing tuple"
    elif B('__list__') in json_obj:
        for v in json_obj[B('__list__')]:
            start.append(decode_obj(v, binary))
    elif B('__set__') in json_obj:
        for v in json_obj[B('__set__')]:
            start.add(decode_obj(v, binary))
    elif B('__ndict__') in json_obj:
        for k, v in json_obj[B('__ndict__')]:
            start[decode_obj(k, binary)] = decode_obj(v, binary)
    elif B('__odict__') in json_obj:
        for k, v in json_obj[B('__odict__')]:
            start[decode_obj(k, binary)] = decode_obj(v, binary)
    elif B('__uuid__') in json_obj:
        assert(False), "No support for sub-classing UUID"
    elif B('__ndarray__') in json_obj:
        assert(False), "No support for sub-classing ndarray"
    elif B('__npgeneric__') in json_obj:
        assert(False), "No support for sub-classing numpy generics"
    elif B('__complex__') in json_obj:
        assert(False), "No support for sub-classing complex"
    elif B('__counter__') in json_obj:
        assert(False), "No support for sub-classing Counter"
    elif B('__slice__') in json_obj:
        assert(False), "No support for sub-classing slice"


def _decode_std_obj(json_obj, binary):
    """
    Helper to :func:`decode_obj` that decodes standard (non-pyGSTi) types.

    Parameters
    ----------
    json_obj : dict
        json-loaded dictionary encoding an object

    binary : bool
        Whether or not to use binary-mode strings as dict keys.

    Returns
    -------
    object
    """
    B = _tobin if binary else _ident

    if B('__tuple__') in json_obj:
        return tuple([decode_obj(v, binary) for v in json_obj[B('__tuple__')]])
    elif B('__list__') in json_obj:
        return list([decode_obj(v, binary) for v in json_obj[B('__list__')]])
    elif B('__set__') in json_obj:
        return set([decode_obj(v, binary) for v in json_obj[B('__set__')]])
    elif B('__slice__') in json_obj:
        v = json_obj[B('__slice__')]
        return slice(decode_obj(v[0], binary), decode_obj(v[1], binary),
                     decode_obj(v[2], binary))
    elif B('__range__') in json_obj:
        start, stop, step = json_obj[B('__range__')]
        return range(start, stop, step)
    elif B('__ndict__') in json_obj:
        return dict([(decode_obj(k, binary), decode_obj(v, binary))
                     for k, v in json_obj[B('__ndict__')]])
    elif B('__odict__') in json_obj:
        return _collections.OrderedDict(
            [(decode_obj(k, binary), decode_obj(v, binary)) for k, v in json_obj[B('__odict__')]])
    elif B('__counter__') in json_obj:
        return _collections.Counter(
            {decode_obj(k, binary): decode_obj(v, binary) for k, v in json_obj[B('__counter__')]})
    elif B('__uuid__') in json_obj:
        return _uuid.UUID(hex=_tostr(json_obj[B('__uuid__')]))
    elif B('__bytes__') in json_obj:
        return json_obj[B('__bytes__')] if binary else \
            _base64.b64decode(json_obj[B('__bytes__')])
    elif B('__string__') in json_obj:
        return _tostr(json_obj[B('__string__')]) if binary else \
            json_obj[B('__string__')]

    # check for numpy
    elif B('__ndarray__') in json_obj:
        # Check if 'kind' is in json_obj to enable decoding of data
        # serialized with older versions:
        if json_obj[B('kind')] == B('V'):
            descr = [tuple(_tostr(t) if isinstance(t, bytes) else t for t in d)
                     for d in json_obj[B('dtype')]]
        else:
            descr = json_obj[B('dtype')]

        if json_obj[B('kind')] == B('O'):  # special decoding for object-type arrays
            data = [decode_obj(el, binary) for el in json_obj[B('__ndarray__')]]
            flat_ar = _np.empty(len(data), dtype=_np.dtype(descr))
            for i, el in enumerate(data):
                flat_ar[i] = el  # can't just make a np.array(data) because data may be, e.g., tuples
            return flat_ar.reshape(json_obj[B('shape')])
        else:
            data = json_obj[B('__ndarray__')] if binary else \
                _base64.b64decode(json_obj[B('__ndarray__')])
            return _np.fromstring(data, dtype=_np.dtype(descr)).reshape(json_obj[B('shape')])
    elif B('__scipy_csrmatrix__') in json_obj:
        if json_obj[B('kind')] == 'V':
            descr = [tuple(_tostr(t) if isinstance(t, bytes) else t for t in d)
                     for d in json_obj[B('dtype')]]
        else:
            descr = json_obj[B('dtype')]
        data = decode_obj(json_obj[B('__scipy_csrmatrix__')], binary)
        indices = decode_obj(json_obj[B('indices')], binary)
        indptr = decode_obj(json_obj[B('indptr')], binary)
        return _sps.csr_matrix((data, indices, indptr), dtype=_np.dtype(descr))
    elif B('__npgeneric__') in json_obj:
        data = json_obj[B('__npgeneric__')] if binary else \
            _base64.b64decode(json_obj[B('__npgeneric__')])
        return _np.fromstring(
            data, dtype=_np.dtype(json_obj[B('dtype')])
        )[0]
    elif B('__complex__') in json_obj:
        return complex(_tostr(json_obj[B('__complex__')]))
    elif B('__function__') in json_obj:
        modname, fnname = json_obj[B('__function__')]
        module = _importlib.import_module(_tostr(modname))
        return getattr(module, _tostr(fnname))


def _tostr(x):
    """
    Convert a value to the native string format.

    Parameters
    ----------
    x : str or bytes
        value to convert to a native string.

    Returns
    -------
    str
    """
    if isinstance(x, bytes):
        return x.decode()
    else:
        return str(x)


def _tobin(x):
    """
    Serialize strings to UTF8

    Parameters
    ----------
    x : str or bytes
        value to convert to a UTF8 binary string.

    Returns
    -------
    bytes
    """
    if isinstance(x, str):
        return bytes(x, 'utf-8')
    else:
        return x


def _ident(x):
    return x

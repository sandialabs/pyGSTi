"""
Defines json package interface capable of encoding pyGSTi objects
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import json as _json

from pygsti.serialization.jsoncodec import decode_obj
from pygsti.serialization.jsoncodec import encode_obj

from pygsti.tools.legacytools import deprecate as _deprecated_fn


class PygstiJSONEncoder(_json.JSONEncoder):
    """
    JSON Encoder capable of handling pyGSTi types
    """

    def encode(self, item):
        """
        Main encoding function

        Parameters
        ----------
        item : various
            item to encode

        Returns
        -------
        various
        """
        return super(PygstiJSONEncoder, self).encode(encode_obj(item, False))


_deprecation_msg= 'Use of the python json module for serialization of pygsti objects is deprecated.'\
                  +' Most pysgti objects now natively support json serialization and deserialization and '\
                  + 'users should migrate to that functionality when possible.'

@_deprecated_fn(_deprecation_msg)
def dumps(obj, **kwargs):
    """
    An overload of json.dumps that works with pyGSTi types

    Parameters
    ----------
    obj : object
        object to serialize.

    Returns
    -------
    str
    """
    kwargs['cls'] = PygstiJSONEncoder
    return _json.dumps(obj, **kwargs)

@_deprecated_fn(_deprecation_msg)
def dump(obj, f, **kwargs):
    """
    An overload of json.dump that works with pyGSTi types

    Parameters
    ----------
    obj : object
        object to serialize

    f : file
        output file

    Returns
    -------
    None
    """
    kwargs['cls'] = PygstiJSONEncoder
    enc = encode_obj(obj, False)  # this shouldn't be needed... bug in json I think.
    return _json.dump(enc, f, **kwargs)

@_deprecated_fn(_deprecation_msg)
def loads(s, **kwargs):
    """
    An overload of json.loads that works with pyGSTi types

    Parameters
    ----------
    s : str
        serialized object(s)

    Returns
    -------
    object
    """
    decoded_json = _json.loads(s, **kwargs)  # load normal JSON
    return decode_obj(decoded_json, False)  # makes pygsti objects

@_deprecated_fn(_deprecation_msg)
def load(f, **kwargs):
    """
    An overload of json.load that works with pyGSTi types

    Parameters
    ----------
    f : file
        open file to read from

    Returns
    -------
    object
    """
    decoded_json = _json.load(f, **kwargs)  # load normal JSON
    return decode_obj(decoded_json, False)  # makes pygsti objects

"""
Defines msgpack package interface capable of encoding pyGSTi objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import msgpack as _msgpack
msgpack_uses_binary_strs = _msgpack.version < (1, 0, 0)  # msgpack only used binary strings in pre 1.0 versions

from pygsti.serialization.jsoncodec import encode_obj
from pygsti.serialization.jsoncodec import decode_obj

from pygsti.tools.legacytools import deprecate as _deprecated_fn

_deprecation_msg= 'Use of the python msgpack module for serialization of pygsti objects is deprecated.'\
                  +' Most pysgti objects now natively support json serialization and deserialization and '\
                  + 'users should migrate to that functionality when possible.'

@_deprecated_fn(_deprecation_msg)
def dumps(obj, **kwargs):
    """
    An overload of msgpack.dumps that works with pyGSTi types

    Parameters
    ----------
    obj : object
        object to serialize.

    Returns
    -------
    str
    """
    enc = encode_obj(obj, msgpack_uses_binary_strs)
    return _msgpack.packb(enc, **kwargs)

@_deprecated_fn(_deprecation_msg)
def dump(obj, f, **kwargs):
    """
    An overload of msgpack.dump that works with pyGSTi types

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
    enc = encode_obj(obj, msgpack_uses_binary_strs)
    _msgpack.pack(enc, f, **kwargs)

@_deprecated_fn(_deprecation_msg)
def loads(s, **kwargs):
    """
    An overload of msgpack.loads that works with pyGSTi types

    Parameters
    ----------
    s : str
        serialized object(s)

    Returns
    -------
    object
    """
    decoded_msgpack = _msgpack.unpackb(s, **kwargs)  # load normal MSGPACK
    return decode_obj(decoded_msgpack, msgpack_uses_binary_strs)  # makes pygsti objects

@_deprecated_fn(_deprecation_msg)
def load(f, **kwargs):
    """
    An overload of msgpack.load that works with pyGSTi types

    Parameters
    ----------
    f : file
        open file to read from

    Returns
    -------
    object
    """
    decoded_msgpack = _msgpack.unpack(f, **kwargs)  # load normal MSGPACK
    return decode_obj(decoded_msgpack, msgpack_uses_binary_strs)  # makes pygsti objects

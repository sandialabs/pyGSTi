from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines msgpack package interface capable of encoding pyGSTi objects"""

import msgpack as _msgpack

from .jsoncodec import encode_obj
from .jsoncodec import decode_obj

def encode_msgpack_obj(obj):
    return encode_obj(obj, binary=True)

def decode_msgpack_obj(obj):
    return decode_obj(obj, binary=True)

def dumps(obj, **kwargs):
    enc = encode_msgpack_obj(obj)
    return _msgpack.packb(enc, **kwargs) #, default=encode_msgpack_obj)

def dump(obj, f, **kwargs):
    enc = encode_msgpack_obj(obj)
    _msgpack.pack(enc, f, **kwargs) #, default=encode_msgpack_obj)

def loads(s, **kwargs):
    decoded_msgpack = _msgpack.unpackb(s, **kwargs) #load normal MSGPACK
    return decode_msgpack_obj(decoded_msgpack) #makes pygsti objects

def load(f, **kwargs):
    decoded_msgpack = _msgpack.unpack(f, **kwargs) #load normal MSGPACK
    return decode_msgpack_obj(decoded_msgpack) #makes pygsti objects


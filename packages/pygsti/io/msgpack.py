""" Defines msgpack package interface capable of encoding pyGSTi objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import msgpack as _msgpack

from .jsoncodec import encode_obj
from .jsoncodec import decode_obj

def dumps(obj, **kwargs):
    """ An overload of msgpack.dumps that works with pyGSTi types """
    enc = encode_obj(obj,True)
    return _msgpack.packb(enc, **kwargs)

def dump(obj, f, **kwargs):
    """ An overload of msgpack.dump that works with pyGSTi types """
    enc = encode_obj(obj,True)
    _msgpack.pack(enc, f, **kwargs)

def loads(s, **kwargs):
    """ An overload of msgpack.loads that works with pyGSTi types """
    decoded_msgpack = _msgpack.unpackb(s, **kwargs) #load normal MSGPACK
    return decode_obj(decoded_msgpack,True) #makes pygsti objects

def load(f, **kwargs):
    """ An overload of msgpack.load that works with pyGSTi types """
    decoded_msgpack = _msgpack.unpack(f, **kwargs) #load normal MSGPACK
    return decode_obj(decoded_msgpack,True) #makes pygsti objects


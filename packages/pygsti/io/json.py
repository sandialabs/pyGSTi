""" Defines json package interface capable of encoding pyGSTi objects"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import json as _json

from .jsoncodec import encode_obj
from .jsoncodec import decode_obj

class PygstiJSONEncoder(_json.JSONEncoder):
    """ JSON Encoder capable of handling pyGSTi types """
    def encode(self, item):
        """ Main encoding function """
        return super(PygstiJSONEncoder, self).encode( encode_obj(item,False) )

def dumps(obj, **kwargs):
    """ An overload of json.dumps that works with pyGSTi types """
    kwargs['cls']=PygstiJSONEncoder
    return _json.dumps(obj, **kwargs)

def dump(obj, f, **kwargs):
    """ An overload of json.dump that works with pyGSTi types """
    kwargs['cls']=PygstiJSONEncoder
    enc = encode_obj(obj,False) #this shouldn't be needed... bug in json I think.
    return _json.dump(enc, f, **kwargs)

def loads(s, **kwargs):
    """ An overload of json.loads that works with pyGSTi types """
    decoded_json = _json.loads(s,**kwargs) #load normal JSON
    return decode_obj(decoded_json,False) #makes pygsti objects

def load(f, **kwargs):
    """ An overload of json.load that works with pyGSTi types """
    decoded_json = _json.load(f,**kwargs) #load normal JSON
    return decode_obj(decoded_json,False) #makes pygsti objects


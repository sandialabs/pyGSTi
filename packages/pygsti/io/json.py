from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines json package interface capable of encoding pyGSTi objects"""

import json as _json

from .jsoncodec import encode_obj
from .jsoncodec import decode_obj

class PygstiJSONEncoder(_json.JSONEncoder):
    def encode(self, item):
        return super(PygstiJSONEncoder, self).encode( encode_obj(item) )

def dumps(obj, **kwargs):
    kwargs['cls']=PygstiJSONEncoder, 
    return _json.dumps(obj, **kwargs)

def dump(obj, f, **kwargs):
    kwargs['cls']=PygstiJSONEncoder
    enc = encode_obj(obj) #this shouldn't be needed... bug in json I think.
    return _json.dump(enc, f, **kwargs)

def loads(s, **kwargs):
    decoded_json = _json.loads(s,**kwargs) #load normal JSON
    return decode_obj(decoded_json) #makes pygsti objects

def load(f, **kwargs):
    decoded_json = _json.load(f,**kwargs) #load normal JSON
    return decode_obj(decoded_json) #makes pygsti objects


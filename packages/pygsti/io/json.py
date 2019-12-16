""" Defines json package interface capable of encoding pyGSTi objects"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import json as _json

from .jsoncodec import encode_obj
from .jsoncodec import decode_obj


class PygstiJSONEncoder(_json.JSONEncoder):
    """ JSON Encoder capable of handling pyGSTi types """

    def encode(self, item):
        """ Main encoding function """
        return super(PygstiJSONEncoder, self).encode(encode_obj(item, False))


def dumps(obj, **kwargs):
    """ An overload of json.dumps that works with pyGSTi types """
    kwargs['cls'] = PygstiJSONEncoder
    return _json.dumps(obj, **kwargs)


def dump(obj, f, **kwargs):
    """ An overload of json.dump that works with pyGSTi types """
    kwargs['cls'] = PygstiJSONEncoder
    enc = encode_obj(obj, False)  # this shouldn't be needed... bug in json I think.
    return _json.dump(enc, f, **kwargs)


def loads(s, **kwargs):
    """ An overload of json.loads that works with pyGSTi types """
    decoded_json = _json.loads(s, **kwargs)  # load normal JSON
    return decode_obj(decoded_json, False)  # makes pygsti objects


def load(f, **kwargs):
    """ An overload of json.load that works with pyGSTi types """
    decoded_json = _json.load(f, **kwargs)  # load normal JSON
    return decode_obj(decoded_json, False)  # makes pygsti objects

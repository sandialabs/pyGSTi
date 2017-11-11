import sys as _sys
import importlib as _importlib
import json as _json
import base64 as _base64
import numpy as _np
import uuid as _uuid
import collections as _collections
import pygsti.objects

if _sys.version_info >= (3, 0):
    range_type = range
else:
    range_type = xrange


def class_hasattr(instance, attr):
    return hasattr(instance.__class__,attr)

class PygstiJSONEncoder(_json.JSONEncoder):
    def encode(self, item):
        return super(PygstiJSONEncoder, self).encode( encode_obj(item) )

def encode_obj(item):
    """ Returns json-able version of item """
    print("DB: encoding type ", type(item)) #DEBUG
    is_pygsti_obj = hasattr(item,'__class__') and \
                    hasattr(item.__class__,'__module__') and \
                    item.__class__.__module__.startswith('pygsti')

    is_pygsti_class = isinstance(item,type) and hasattr(item,'__module__') \
                      and item.__module__.startswith('pygsti')

    # Pygsti class encoding
    if is_pygsti_class: # or class_hasattr(item, '__pygsti_getstate__')
        return {'__pygsticlass__': (item.__module__,item.__name__)}

    # Pygsti object encoding
    elif is_pygsti_obj: # or class_hasattr(item, '__pygsti_getstate__')

        #Get State (and/or init args)
        if class_hasattr(item, '__pygsti_reduce__'):
            red = item.__pygsti_reduce__() #returns class, construtor_args, state
            assert(red[0] is item.__class__), "No support for weird reducing!"
            init_args = red[1] if len(red) > 1 else []
            state = red[2] if len(red) > 2 else {}
            if state is None: state = {}
            state.update({ '__init_args__': init_args })
        elif class_hasattr(item, '__pygsti_getstate__'):
            state = item.__pygsti_getstate__() #must return a dict
        elif class_hasattr(item,'__getstate__'):
            state = item.__getstate__()
        elif hasattr(item,'__dict__'):
            state = item.__dict__  #take __dict__ as state
        else:
            state = None

        if state is None: #Note: __dict__ and __getstate__ may *return* None (python 2.7)
            if hasattr(item,'_asdict'): #named tuples
                state = { '__init_args__': list(item._asdict().values()) }
                  # values will be ordered as per __init__ so no need for keys
            else:
                raise ValueError("Can't get state of %s object" % type(item))
            

        #d = { k: encode_obj(v) for k,v in state.items() }

        #DEBUG (instead of above line)
        d = {}
        #print("DB: Encoding state for %s object:" % type(item))
        for k,v in state.items():
            #print("Encoding key: ",k)
            d[k] = encode_obj(v)
            try: _json.dumps(d[k])
            except Exception as e:
                print("Cannot JSON %s key: " % k, d[k])
                raise e


        d.update({ '__pygstiobj__': (item.__class__.__module__,
                                     item.__class__.__name__)})

        #Currently, don't add standard-base-class state
        #if we know how to __init__, since we'll assume this
        # should initialize the entire (base class included) instance
        encode_std_base = bool('__init_args__' not in d)
        
        if encode_std_base:
            std_encode = encode_std_obj(item)
            if std_encode is not item: #if there's something to encode
                # this pygsti object is also a standard-object instance
                assert(isinstance(std_encode,dict))
                d['__std_base__'] = std_encode

        try:
            _json.dumps(d)
        except Exception as e:
            print("Cannot JSON ",type(item))
            raise e
            
        return d
    else:
        return encode_std_obj(item)

def encode_std_obj(item):
    # Other builtin or standard object encoding
    if isinstance(item, tuple):
        return {'__tuple__': [encode_obj(v) for v in item]}
    elif isinstance(item, list):
        return {'__list__': [encode_obj(v) for v in item]}
    elif isinstance(item, set):
        return {'__set__': [encode_obj(v) for v in item]}
    elif isinstance(item, range_type):
        if _sys.version_info >= (3, 0):
            return {'__range__': (item.start, item.stop, item.step) }
        else:
            return {'__list__': list(item) } #python2 -> serialze ranges as lists
    elif isinstance(item, _collections.OrderedDict):
        return {'__odict__': [(encode_obj(k),encode_obj(v))
                              for k,v in item.items()]}
    elif isinstance(item, _collections.Counter):
        return {'__counter__': [(encode_obj(k),encode_obj(v))
                              for k,v in dict(item).items()]}
    elif isinstance(item, dict):
        return {'__ndict__': [(encode_obj(k),encode_obj(v))
                              for k,v in item.items()]}
    elif isinstance(item, _uuid.UUID):
        return {'__uuid__': str(item.hex) }
    elif isinstance(item, complex):
        return  {'__complex__': item.__repr__()}
    elif isinstance(item, bytes):
        return {'__bytes__': tostr(_base64.b64encode(item)) }
        
    #Numpy encoding
    elif isinstance(item, _np.ndarray):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if item.dtype.kind == 'V':
            kind = 'V'
            descr = tostr(item.dtype.descr)
        else:
            kind = ''
            descr = tostr(item.dtype.str)
        return {'__ndarray__': tostr(_base64.b64encode(item.tobytes())),
                'dtype': descr,
                'kind': kind,
                'shape': item.shape}
        
    elif isinstance(item, (_np.bool_, _np.number)):
        return {'__npgeneric__': tostr(_base64.b64encode(item.tobytes())),
                'dtype': tostr(item.dtype.str)}

    elif callable(item): #functions
        return {'__function__': (item.__module__, item.__name__) }

    return item # assume the bare item is json-able



def decode_obj(obj):
    if isinstance(obj, dict):
        if '__pygsticlass__' in obj:
            modname, clsname = obj['__pygsticlass__']
            module = _importlib.import_module(modname)
            class_ = getattr(module, clsname)
            return class_
            
        elif '__pygstiobj__' in obj:
            #DEBUG
            #print("DB: creating %s" % str(obj['__pygstiobj__']))
            #print("DB: obj is type %s with keyvals:" % type(obj))
            #for k,v in obj.items():
            #    print("%s (%s): %s (%s)" % (k,type(k),v,type(v)))
            
            modname, clsname = obj['__pygstiobj__']
            module = _importlib.import_module(modname)
            class_ = getattr(module, clsname)

            if '__init_args__' in obj: # construct via __init__
                args = decode_obj(obj['__init_args__'])
                instance = class_(*args)
                
            else: #init via __new__ and set state
                instance = class_.__new__(class_)

            #Create state dict
            state_dict = {}
            for k,v in obj.items():
                if k in ('__pygstiobj__','__init_args__','__std_base__'): continue
                state_dict[k] = decode_obj(v)

            #Set state
            if class_hasattr(instance, '__pygsti_setstate__'):
                instance.__pygsti_setstate__(state_dict)
            elif class_hasattr(instance, '__setstate__'):
                instance.__setstate__(state_dict)
            elif hasattr(instance,'__dict__'): #just update __dict__
                instance.__dict__.update(state_dict)
            elif len(state_dict) > 0:
                raise ValueError("Cannot set nontrivial state of %s object" % type(instance))

            #update instance with std-object info if needed (only if __init__ not called)
            if '__std_base__' in obj:
                decode_std_base(obj['__std_base__'], instance)

            return instance
        else:
            return decode_std_obj(obj)
    else:
        return obj

def decode_std_base(obj,start):
    if '__tuple__' in obj:
        #OK if __init_args since this means we knew how to construct it (e.g. namedtuples)
        assert('__init_args' in obj), "No support for sub-classing tuple"
    elif '__list__' in obj:
        for v in obj['__list__']:
            start.append(decode_obj(v))
    elif '__set__' in obj:
        for v in obj['__set__']:
            start.add(decode_obj(v))
    elif '__ndict__' in obj:
        for k,v in obj['__ndict__']:
            start[decode_obj(k)] = decode_obj(v)
    elif '__odict__' in obj:
        for k,v in obj['__odict__']:
            start[decode_obj(k)] = decode_obj(v)
    elif '__uuid__' in obj:
        assert(False), "No support for sub-classing UUID"
    elif '__ndarray__' in obj:
        assert(False), "No support for sub-classing ndarray"
    elif '__npgeneric__' in obj:
        assert(False), "No support for sub-classing numpy generics"
    elif '__complex__' in obj:
        assert(False), "No support for sub-classing complex"
    elif '__counter__' in obj:
        assert(False), "No support for sub-classing Counter"
        
def decode_std_obj(obj):
    if '__tuple__' in obj:
        return tuple([decode_obj(v) for v in obj['__tuple__']])
    elif '__list__' in obj:
        return list([decode_obj(v) for v in obj['__list__']])
    elif '__set__' in obj:
        return set([decode_obj(v) for v in obj['__set__']])
    elif '__range__' in obj:
        start,stop,step = obj['__range__']
        if _sys.version_info >= (3, 0):
            return range(start,stop,step)
        else:
            return list(xrange(start,stop,step)) #lists in python2
    elif '__ndict__' in obj:
        return dict([(decode_obj(k),decode_obj(v))
                     for k,v in obj['__ndict__']])
    elif '__odict__' in obj:
        return _collections.OrderedDict(
            [(decode_obj(k),decode_obj(v)) for k,v in obj['__odict__']])
    elif '__odict__' in obj:
        return _collections.Counter(
            {decode_obj(k): decode_obj(v) for k,v in obj['__counter__']})
    elif '__uuid__' in obj:
        return _uuid.UUID(hex=obj['__uuid__'])
    elif '__bytes__' in obj:
        return _base64.b64decode(obj['__bytes__'])
    
    # check for numpy
    elif '__ndarray__' in obj:
        # Check if 'kind' is in obj to enable decoding of data
        # serialized with older versions:
        if obj['kind'] == 'V':
            descr = [tuple(tostr(t) if isinstance(t,bytes) else t for t in d)
                     for d in obj['dtype']]
        else:
            descr = obj['dtype']
        return _np.fromstring(_base64.b64decode(obj['__ndarray__']),
                        dtype=_np.dtype(descr)).reshape(obj['shape'])
    elif '__npgeneric__' in obj:
        return _np.fromstring(
            _base64.b64decode(obj['__npgeneric__']),
            dtype=_np.dtype(obj['dtype'])
        )[0]
    elif '__complex__' in obj:
        return complex(obj['__complex__'])

    elif '__function__' in obj:
        modname, fnname = obj['__function__']
        module = _importlib.import_module(modname)
        return getattr(module, fnname)
    
    return obj



def tostr(x):
    if _sys.version_info >= (3, 0):
        if isinstance(x, bytes):
            return x.decode()
        else:
            return str(x)
    else:
        return x        

    
def dumps(obj):
    return _json.dumps(obj, cls=PygstiJSONEncoder)

def dump(obj, f):
    enc = encode_obj(obj) #this shouldn't be needed... bug in json I think.
    return _json.dump(enc, f, cls=PygstiJSONEncoder)

def loads(s):
    decoded_json = _json.loads(s) #load normal JSON
    return decode_obj(decoded_json) #makes pygsti objects

def load(f):
    decoded_json = _json.load(f) #load normal JSON
    return decode_obj(decoded_json) #makes pygsti objects



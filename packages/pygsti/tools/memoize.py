from functools import wraps
import json

def memoize(obj):
    '''
    Memoize any function based on args/kwargs
    (Pretty robust)
    '''
    #cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        '''
        #print(hash(args))
        try:
            if args not in cache:
                cache[args] = obj(*args, **kwargs)
            return cache[args]
        except TypeError:
        '''
        return obj(*args, **kwargs)
    return memoizer

def kwarg_memoize(obj):
    '''
    Memoize any function based on args/kwargs
    (Pretty robust)
    cache = obj.cache = {}
    '''

    @wraps(obj)
    def memoizer(*args, **kwargs):
        '''
        key = str(args) + json.dumps(kwargs, sort_keys=True)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
        '''
        return obj(*args, **kwargs)
    return memoizer

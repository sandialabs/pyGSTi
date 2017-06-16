from functools import partial, wraps

'''
def to_hashable(t):
    try:
        hash(t)
        return t
    except TypeError:
        if isinstance(t, list):
            return tuple(to_hashable(item) for item in t)
        else:
            raise TypeError('Conversion from {} to hashable type not defined'.format(type(t)))
'''

# note that this decorator ignores **kwargs
def memoize(obj):
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError('Cannot currently memoize on kwargs')
        #args = tuple(to_hashable(arg) for arg in args)
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer

from .formatter_helpers import *
_eb_exists = lambda t : t[1] is not None

class EBFormatter(object):
    def __init__(self, f, formatstringA='%s +/- %s', formatstringB='%s'):
        self.f = f
        if hasattr(f, 'specs'):
            self.specs = f.specs
        self.formatstringA = formatstringA
        self.formatstringB = formatstringB

    def __call__(self, t):
        if hasattr(self.f, 'specs'):
            give_specs(self.f, self.specs)
        if _eb_exists(t):
            return self.formatstringA % (self.f(t[0]), self.f(t[1]))
        else:
            return self.formatstringB % self.f(t[0])


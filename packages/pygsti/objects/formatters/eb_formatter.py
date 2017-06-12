from .formatter_helpers import *

class EBFormatter(object):
    def __init__(self, f, formatstringA='%s +/- %s', formatstringB='%s'):
        self.f = f
        if hasattr(f, 'specs'):
            self.specs = f.specs
        self.formatstringA = formatstringA
        self.formatstringB = formatstringB

    def __call__(self, t, specs):
        if hasattr(self.f, 'specs'):
            give_specs(self.f, self.specs)
        if t.has_eb():
            return self.formatstringA % (self.f(t.get_value()), self.f(t.get_err_bar()))
        else:
            return self.formatstringB % self.f(t.get_value())


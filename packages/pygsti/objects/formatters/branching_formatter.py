from .formatter_helpers import *

# Takes two formatters (a and b), and determines which to use based on a predicate
# (Used in building formatter templates)
class BranchingFormatter(object):
    def __init__(self, predicate, a, b):
        self.predicate = predicate
        self.a = a
        self.b = b

        # So that a branching formatter can hold parameterized formatters
        self.specs = {}
        if hasattr(a, 'specs'):
            self.specs.update(a.specs)
        if hasattr(b, 'specs'):
            self.specs.update(b.specs)

    def __call__(self, label):
        if self.predicate(label):
            give_specs(self.a, self.specs)
            return self.a(label)
        else:
            give_specs(self.a, self.specs)
            return self.b(label)

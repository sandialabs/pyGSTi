from collections import OrderedDict

class Basis(object):
    def __init__(self, name, matrices, longname=None):
        self.name = name
        if longname is None:
            self.longname = self.name
        else:
            self.longname = longname

        self.matrices = OrderedDict()
        assert len(matrices) > 0, 'Need at least one matrix in basis'
        for i, mx in enumerate(matrices):
            if isinstance(mx, tuple):
                mx, label = mx
            else:
                label = 'M{}'.format(i)
            self.matrices[label] = mx



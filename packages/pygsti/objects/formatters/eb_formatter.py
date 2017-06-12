from .formatter import Formatter

class EBFormatter(object):
    def __init__(self, custom=None, formatstringA='{} +/- {}', formatstringB='{}'):
        if isinstance(custom, Formatter):
            self.f = custom
        else:
            self.f = Formatter(custom)
        self.formatstringA = formatstringA
        self.formatstringB = formatstringB

    def __call__(self, t, specs):
        if t.has_eb():
            return self.formatstringA.format(self.f(t.get_value(), specs), self.f(t.get_err_bar(), specs))
        else:
            return self.formatstringB.format(self.f(t.get_value(), specs))


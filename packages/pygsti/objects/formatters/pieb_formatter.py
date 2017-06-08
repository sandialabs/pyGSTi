from .formatter_helpers import *
from .eb_formatter      import EBFormatter

class PiEBFormatter(EBFormatter):
    def __call__(self, t):
        if str(t[0]) == '--' or str(t[0]) == '':  return t[0]
        else:
            return super(PiEBFormatter, self).__call__(t)

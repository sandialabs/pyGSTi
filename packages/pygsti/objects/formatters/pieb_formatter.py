from .formatter_helpers import *
from .eb_formatter      import EBFormatter

class PiEBFormatter(EBFormatter):
    def __call__(self, t):
        if str(t.get_val()) == '--' or str(t.get_val()) == '':  return t.get_val()
        else:
            return super(PiEBFormatter, self).__call__(t)

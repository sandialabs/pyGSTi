from .parameterized_formatter import _ParameterizedFormatter

# Gives precision arguments to formatters
class _PrecisionFormatter(_ParameterizedFormatter):
    '''Helper class for Precision Formatting
    Takes a custom function and a dictionary of keyword arguments:
    So, something like _PrecisionFormatter(latex) would pass precision arguments to
      the latex formatter function during table.render() calls
    '''
    def __init__(self, custom, defaults={}, formatstring='%s'):
        super(_PrecisionFormatter, self).__init__(custom, ['precision', 'polarprecision','sciprecision'],
                                                 defaults, formatstring)


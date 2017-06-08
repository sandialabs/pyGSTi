from .parameterized_formatter import ParameterizedFormatter

# Gives precision arguments to formatters
class PrecisionFormatter(ParameterizedFormatter):
    '''Helper class for Precision Formatting
    Takes a custom function and a dictionary of keyword arguments:
    So, something like PrecisionFormatter(latex) would pass precision arguments to
      the latex formatter function during table.render() calls
    '''
    def __init__(self, custom, defaults={}, formatstring='%s'):
        super(PrecisionFormatter, self).__init__(custom, ['precision', 'polarprecision','sciprecision'],
                                                 defaults, formatstring)


from inspect import getargspec as _getargspec

# Helper function to ParameterizedFormatter
def _has_argname(argname, function):
    return argname in _getargspec(function).args

# Gives arguments to formatters
class ParameterizedFormatter(object):
    '''
    Class that will pass down specs (arguments) to functions that need them

    For example, a precision-parameterized latex formatter without the help of the PrecisionFormatter might look like this:
    formatter = ParameterizedFormatter(latex, ['precision', 'polarprecision', 'sciprecision'])
    Which, when used with a FormatSet, would have arguments to table.render() passed down to the latex() function
    '''
    def __init__(self, custom, neededSpecs, defaults={}, formatstring='%s'):
        self.custom       = custom
        self.specs        = { neededSpec : None for neededSpec in neededSpecs }
        self.defaults     = defaults
        self.formatstring = formatstring

    def render(self, label, _):
        # If the formatter is being called, we know that the needed specs have successfully been supplied by FormatSet
        self.defaults.update(self.specs)
        # Supply arguments to the custom formatter (if it needs them)
        for argname in self.defaults:
            if not callable(self.custom): # If some keyword arguments were supplied already
                if _has_argname(argname, self.custom[0]):             # 'if it needs them'
                    # update the argument in custom's existing keyword dictionary
                    self.custom[1][argname] = self.defaults[argname]
            else:
                if _has_argname(argname, self.custom): # If custom is a lone callable (not a tuple)
                # Create keyword dictionary for custom, modifiying it to be a tuple
                #   (function, kwargs)
                    self.custom = (self.custom, {argname : self.defaults[argname]})
        return self.formatstring % self.custom[0](label, **self.custom[1])

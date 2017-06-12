def no_format(s):
    return str(s)

def give_specs(formatter, specs):
    '''
    Pass parameters down to a formatter
    Parameters
    --------
    formatter : callable, takes arguments
    specs : dictionary of argnames : values
    Returns
    ------
    None
    Raises
    ------
    ValueError : If a needed spec is not supplied.
    '''
    # If the formatter requires a setting to do its job, give the setting
    if hasattr(formatter, 'specs'):
        for spec in formatter.specs:
            if spec not in specs or specs[spec] is None:
                raise ValueError(
                        ('The spec %s was not supplied to ' % spec) +
                        ('FormatSet, but is needed by an active formatter'))
            formatter.specs[spec] = specs[spec]

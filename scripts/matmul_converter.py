import re


def make_dotdot_pattern(dotarg):
    spaces = '[ ]*'  # one or more spaces
    arg_left = f'(?P<left>{spaces}{dotarg}{spaces})'
    arg_right = f'(?P<right>{dotarg}{spaces})'
    # ^ spaces right before arg_right shouldn't be captured 
    pattern_spec = r"_np\.dot\(" + f"{arg_left},{spaces}{arg_right}" + r"\)"
    dotdot = re.compile(pattern_spec)
    return dotdot

# def make_composed_dotdot_pattern(dotarg):
#     spaces = '[ ]*'  # one or more spaces
#     composed_dotarg = dotarg + spaces + '@' + spaces + dotarg
#     arg_left = 
#     return

def replacer(match):
    return f"{match.group('left')} @ {match.group('right')}"


if __name__ == '__main__':
    dotarg = r'[A-Za-z0-9]+(\.T)?(\.conj\(\))?'
    # ^ expressions given by <variable name>, with optional transpose or conjugation.
    npdotdot = make_dotdot_pattern(dotarg)

    print('\nInstances where we want to replace ... ')
    out = npdotdot.sub(replacer, '_np.dot(aAa, B11), nice to see you. _np.dot(aAa, B12)'); print(out)
    out = npdotdot.sub(replacer, '_np.dot(aAa, B11), nice to see you. _np.dot( aAa, B12 )'); print(out)
    out = npdotdot.sub(replacer, '_np.dot(aAa, B11), nice to see you. _np.dot(aAa, B12.T)'); print(out)
    out = npdotdot.sub(replacer, '_np.dot(aAa, B11), nice to see you. _np.dot(aAa.T, B12)'); print(out)

    out = npdotdot.sub(replacer, '_np.dot(aAa.T, B11), nice to see you. _np.dot(aAa, B12)'); print(out)
    out = npdotdot.sub(replacer, '_np.dot(aAa.T, B11.T), nice to see you. _np.dot(aAa, B12.T)'); print(out)
    out = npdotdot.sub(replacer, '_np.dot(aAa.conj(), B11), nice to see you. _np.dot(aAa.T, B12)'); print(out)

    print('\nInstances where we do NOT expect to replace ...')
    out = npdotdot.sub(replacer, '_np.dot(aAa.I, B11), nice to see you. _np.dot(aAa, B12.J)'); print(out)
    out = npdotdot.sub(replacer, '_np.dot(aAa.H, B11.K), nice to see you. _np.dot(aAa.T, B12.A)'); print(out)


    print(0)
    
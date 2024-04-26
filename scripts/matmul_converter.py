import re
import os


def make_simple_dotdot_pattern(dotarg):
    spaces = '[ ]*'  # one or more spaces
    arg_left = f'(?P<left>{spaces}{dotarg}{spaces})'
    arg_right = f'(?P<right>{dotarg}{spaces})'
    # ^ spaces right before arg_right shouldn't be captured 
    pattern_spec = r"_np\.dot\(" + f"{arg_left},{spaces}{arg_right}" + r"\)"
    dotdot = re.compile(pattern_spec)
    return dotdot


def simple_replacer(match):
    return f"{match.group('left')} @ {match.group('right')}"


def demo_dotdot_simple_replacer():
    dotarg = r'[A-Za-z0-9]+(\.T)?(\.conj\(\))?'
    # ^ expressions given by <variable name>, with optional transpose or conjugation.
    npdotdot = make_simple_dotdot_pattern(dotarg)

    print('\nInstances where we want to replace ... ')
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa, B11), nice to see you. _np.dot(aAa, B12)'); print(out)
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa, B11), nice to see you. _np.dot( aAa, B12 )'); print(out)
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa, B11), nice to see you. _np.dot(aAa, B12.T)'); print(out)
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa, B11), nice to see you. _np.dot(aAa.T, B12)'); print(out)

    out = npdotdot.sub(simple_replacer, '_np.dot(aAa.T, B11), nice to see you. _np.dot(aAa, B12)'); print(out)
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa.T, B11.T), nice to see you. _np.dot(aAa, B12.T)'); print(out)
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa.conj(), B11), nice to see you. _np.dot(aAa.T, B12)'); print(out)

    print('\nInstances where we do NOT expect to replace ...')
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa.I, B11), nice to see you. _np.dot(aAa, B12.J)'); print(out)
    out = npdotdot.sub(simple_replacer, '_np.dot(aAa.H, B11.K), nice to see you. _np.dot(aAa.T, B12.A)'); print(out)
    print(0)
    return


def process_file(file_path, line_transformer):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    transformed_lines = [line_transformer(line) for line in lines]
    with open(file_path, 'w') as file:
        file.writelines(transformed_lines)

def traverse_directory(directory, line_transformer):
    # Walk through all directories and files starting from the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a Python file
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Process the file
                process_file(file_path, line_transformer)

def simple_pass():
    dotarg = r'[A-Za-z0-9]+(\.T)?(\.conj\(\))?'
    # ^ expressions given by <variable name>, with optional transpose or conjugation.
    npdotdot = make_simple_dotdot_pattern(dotarg)
    transformer = lambda line: npdotdot.sub(simple_replacer, line)
    traverse_directory('/Users/rjmurr/Documents/pygsti-general/pyGSTi/pygsti', transformer)


if __name__ == '__main__':
    
    print(0)
    
#! /usr/bin/env python
"""Execute project-wide name changes as described in a changeset YAML"""

import ast
import itertools
import argparse
from pathlib import Path
import warnings
import typing as T

import rope.base
import rope.refactor
import yaml

try:
    # colorize output where available
    from colors import color
except ImportError:
    def color(string, **kwargs):
        return string

try:
    # use BPDB where available
    import bpdb as debugger
except ImportError:
    import pdb as debugger


def _custom_warning(message, category, filename, lineno, line=None) -> str:
    return f"{color(filename, fg='yellow')}:{color(lineno, fg='white')}: {message}\n"


warnings.formatwarning = _custom_warning


class ValidatePath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = Path(values)
        if not path.exists():
            raise AttributeError(f"{option_string} {path} not found")
        setattr(namespace, self.dest, path)


# Parse args globally for script
DEFAULT_PROJECT_ROOT = Path(__file__).absolute().parent.parent
DEFAULT_CHANGESET_YAML = DEFAULT_PROJECT_ROOT / 'scripts' / 'api_names.yaml'
DEFAULT_PROJECT_SOURCE = 'pygsti'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--skip-unsure', action='store_true',
                    help="skip any changes that may not be correct.")
parser.add_argument('-f', '--force-unsure', action='store_true',
                    help="make changes that may not be correct without prompting for confirmation.")
parser.add_argument('-D', '--skip-docs', action='store_true',
                    help="skip making changes in comments and strings.")
parser.add_argument('-P', '--project-root', default=DEFAULT_PROJECT_ROOT, action=ValidatePath,
                    help="path to the root of the project. Will be used as the rope project path.")
parser.add_argument('-Y', '--changeset-yaml', default=DEFAULT_CHANGESET_YAML, action=ValidatePath,
                    help="path to a changeset YAML describing the names in the project to change.")
parser.add_argument('-S', '--project-source', default=DEFAULT_PROJECT_SOURCE, action=ValidatePath,
                    help="relative path from the project root to the source tree.")

args = parser.parse_args()
if args.skip_unsure and args.force_unsure:
    warnings.warn("Arguments `--skip-unsure' and `--force-unsure' conflict; ignoring `--force-unsure'.")
    args.force_unsure = False

project = rope.base.project.Project(str(args.project_root))

occurrence_cache = {}

_unsure_actions = {
    'y': "rename this occurrence",
    'Y': "rename this occurrence, and all other occurrences of {name} without prompting",
    'n': "skip this occurrence",
    'N': "skip this occurrence and all other occurrences of {name} without prompting",
    'm': "skip any unsure changes for names from {module}",
    'M': "make all unsure changes for names from {module} without prompting",
    'o': "skip any unsure changes in {occurrence_module}",
    'O': "make all unsure changes in {occurrence_module} without prompting"
}


def _prompt_for_unsure_action(prompt: str, on_eof: str = 'N', **context) -> str:
    # Interactively prompt for a decision on how to proceed
    print(prompt)
    action_keys = ''.join(_unsure_actions.keys())
    while True:
        try:
            choice = input(f"What action should be taken? [{action_keys}?] "
                           + color('=> ', fg='yellow', style='blink'))
        except EOFError:
            return on_eof

        if choice in _unsure_actions:
            return choice
        elif choice == '?':
            # show help
            print("The following actions are available:")
            for k, msg in _unsure_actions.items():
                print("    " + color(k, fg='white') + ": " + msg.format(**context))
            print("or '?' to display this help message.")
        else:
            print(color(f"Please choose one of [{action_keys}], or '?' for help.", fg='red'))


def _highlight(string: str, start: int, end: int, **kwargs):
    return string[:start] + color(string[start:end], **kwargs) + string[end:]


def _make_unsure_action_factory(module: str):
    if args.skip_unsure:
        return lambda: lambda o: False
    elif args.force_unsure:
        return lambda: lambda o: True

    instance_fastforward = None

    def make_unsure_action(name: str, new_name: str):
        if instance_fastforward is not None:
            return lambda o: instance_fastforward
        fastforward = None

        def action(occurrence: rope.refactor.occurrences.Occurrence) -> bool:
            nonlocal fastforward, instance_fastforward
            if fastforward is not None:
                return fastforward
            elif occurrence.resource.real_path in occurrence_cache:
                return occurrence_cache[occurrence.resource.real_path]
            else:
                occurrence_module = occurrence.resource.path
                prompt = f"Possible instance of `{name}' at {occurrence_module}:{occurrence.offset}"
                # Include occurrence source in prompt
                clip_source = occurrence.resource.read()
                o_start, o_end = occurrence.get_primary_range()
                clip_start = clip_source.rfind('\n', 0, o_start)
                if clip_start == -1:
                    clip_start = None
                clip_end = clip_source.find('\n', o_end)
                if clip_end == -1:
                    clip_end = None

                clip = _highlight(clip_source[clip_start:clip_end],
                                  o_start - clip_start, o_end - clip_end,
                                  fg='cyan', style='bold')

                choice = _prompt_for_unsure_action(
                    prompt + "\n" + clip + "\n",
                    name=name,
                    new_name=new_name,
                    module=module,
                    occurrence_module=color(occurrence_module, fg='green')
                )

                if choice == 'Y':
                    fastforward = True
                elif choice == 'N':
                    fastforward = False
                    return False
                elif choice == 'm':
                    instance_fastforward = fastforward = False
                elif choice == 'M':
                    instance_fastforward = fastforward = True
                elif choice == 'o':
                    occurrence_cache[occurrence.resource.real_path] = False
                elif choice == 'O':
                    occurrence_cache[occurrence.resource.real_path] = True

                return choice in 'yYMO'
        return action
    return make_unsure_action


class SkipName(Exception):
    """Raised when processing a name-change which should be skipped."""
    pass


def _get_offset(module_source: str, name: T.Union[str, T.Iterable[str]]) -> T.Optional[int]:
    """Get the character offset of a name's definition in a module.

    Parameters
    ----------
    module_source : str
        Source code of the module.
    name : str or iterable of str
        Name to find, or an iterable of the names of enclosing scopes
        (functions, classes) and the name to find.

    Returns
    -------
    int or None
        Character offset of the definition of `name` in
        `module_source` if it is defined in the module.
    """
    if isinstance(name, str):
        name = [name]

    def iter_defines(tree: ast.AST) -> T.Iterable[T.Tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield child.name, child
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                yield child.target.id, child.target
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        yield target.id, target

    def get_node(tree: ast.AST, enclosing_names: T.Iterable[str]) -> T.Optional[ast.AST]:
        head, *tail = enclosing_names
        for defname, node in iter_defines(tree):
            if defname == head:
                if len(tail) > 0:
                    return get_node(node, enclosing_names=tail)
                else:
                    return node

    node = get_node(ast.parse(module_source), name)
    if node is not None:
        # We want the offset that points to the actual name.
        if len(getattr(node, 'decorator_list', tuple())) > 0:
            lineno = node.decorator_list[-1].lineno + 1
        else:
            lineno = node.lineno
        source_lines = module_source.split(sep='\n')
        node_offset = sum(len(ln) + 1 for ln in source_lines[:lineno - 1]) + node.col_offset
        *_, name = name
        name_offset = module_source.find(name, node_offset)
        if name_offset >= 0:
            return name_offset


def _module_rename(module_path: Path, name_map: dict) -> None:
    """Execute name changes within a Python module.

    Parameters
    ----------
    module_path : Path
        Path to the module source file to change names in.
    name_map : dict
        A map of old names to new names within the module.
    """
    rope_file = project.get_file(str(module_path))
    display_module = color(module_path, fg='yellow')
    make_unsure_action = _make_unsure_action_factory(display_module)

    def iter_changes(change_map: dict):
        for key, value in change_map.items():
            if isinstance(value, str):
                yield [key], value
            elif isinstance(value, dict):
                keyname = value.pop('__name__', None)
                for tail, n in iter_changes(value):
                    yield [key] + tail, n
                if keyname is not None:
                    yield [key], keyname

    for name_parts, new_name in iter_changes(name_map):
        display_name = color('.'.join(name_parts), fg='white')
        display_new_name = color(new_name, fg='cyan')
        try:
            if new_name is not None:
                # Module source is invalidated after every refactoring action
                module_source = rope_file.read()
                offset = _get_offset(module_source, name_parts)
                if offset is None:
                    raise SkipName(f"name not found in {display_module}")
                else:
                    *head, tail = name_parts
                    new_parts = head + [new_name]
                    if _get_offset(module_source, new_parts) is not None:
                        raise SkipName(
                            f"name `{'.'.join(new_parts)}' already exists in {display_module}"
                        )
                    else:
                        print(f"{display_module}: renaming `{display_name}' -> `{display_new_name}'")
                        changes = rope.refactor.rename.Rename(
                            project, rope_file, offset
                        ).get_changes(
                            new_name,
                            unsure=make_unsure_action(display_name, display_new_name),
                            docs=not args.skip_docs
                        )

                        project.do(changes)
                        project.validate()
        except SkipName as e:
            display_msg = color(str(e), fg='red')
            warnings.warn(f"Skipping rename `{display_name}' -> `{display_new_name}':\n    {display_msg}")
        except Exception as e:
            print(e)
            debugger.post_mortem()


def _project_rename(name_map: dict) -> None:
    """Execute name changes within a Python project.

    Parameters
    ----------
    name_map : dict
        A map of old names to new names within the project, organized by module.
    """
    def iter_map_modules(module_map: dict):
        for key, value in module_map.items():
            head = Path(key)
            if head.suffix in ('.py', '.pyx'):
                yield head, value
            elif value is not None:
                for tail, m in iter_map_modules(value):
                    yield head / tail, m

    for module_path, module_map in iter_map_modules(name_map):
        if module_path.suffix == '.pyx':
            warnings.warn(f"Skipping cython extension module `{module_path}' (can't be refactored)")
        else:
            _module_rename(args.project_source / module_path, module_map)


with open(args.changeset_yaml, 'r') as f:
    name_map = yaml.safe_load(f.read())

_project_rename(name_map)

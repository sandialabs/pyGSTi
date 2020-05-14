#! /usr/bin/env python
"""Execute project-wide name changes as described in a changeset YAML"""

import ast
import itertools
import argparse
from pathlib import Path
import warnings
import typing as T

import rope.base
from rope.refactor import rename
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


def _module_rename(rope_project: rope.base.project.Project, module_path: Path, name_map: dict) -> None:
    """Execute name changes within a Python module.

    Parameters
    ----------
    rope_project : rope.base.project.Project
        Rope project instance.
    module_path : Path
        Path to the module source file to change names in.
    name_map : dict
        A map of old names to new names within the module.
    """
    rope_file = rope_project.get_file(str(module_path))
    display_module = color(module_path, fg='yellow')

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
                        changes = rename.Rename(rope_project, rope_file, offset).get_changes(new_name)
                        rope_project.do(changes)
                        rope_project.validate()
        except SkipName as e:
            display_msg = color(str(e), fg='red')
            warnings.warn(f"Skipping rename `{display_name}' -> `{display_new_name}':\n    {display_msg}")
        except Exception as e:
            print(e)
            debugger.post_mortem()


def _project_rename(rope_project: rope.base.project.Project, name_map: dict) -> None:
    """Execute name changes within a Python project.

    Parameters
    ----------
    rope_project : rope.base.project.Project
        Rope project instance.
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
            _module_rename(rope_project, module_path, module_map)


with open('scripts/api_names.yaml', 'r') as f:
    name_map = yaml.safe_load(f.read())

project = rope.base.project.Project('pygsti')

_project_rename(project, name_map)

#! /usr/bin/env python3
""" A simple script to list members of a module missing docstrings """

import argparse
import ast
from pathlib import Path
from typing import Iterable, Union, Optional

try:
    from colors import color
except Exception:
    def color(s: str, *args, **kwargs) -> str:
        return s


_DOCUMENTABLE = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('file', help="python module source file to search for missing docstrings")
parser.add_argument('-p', '--private', action='store_true', help="include private members")
parser.add_argument('-m', '--magic', action='store_true', help="include magic members")
parser.add_argument('-q', '--quiet', action='store_true', help="output nothing if no members are missing docstrings")
args = parser.parse_args()


def allow_name(name: str) -> bool:
    if name is not None:
        if name.startswith('__') and name.endswith('__'):
            return args.magic
        elif name.startswith('_'):
            return args.private
        else:
            return True


def iter_fields(node: ast.AST) -> Iterable[ast.AST]:
    for fieldname in node._fields:
        yield getattr(node, fieldname)


def iter_missing_docstrings(node: ast.AST) -> Iterable:
    if 'name' not in node._fields or allow_name(node.name):
        if isinstance(node, _DOCUMENTABLE) and ast.get_docstring(node) is None:
            yield node

        for field in iter_fields(node):
            if isinstance(field, list):
                for item in field:
                    if isinstance(item, ast.AST):
                        yield from iter_missing_docstrings(item)
            elif isinstance(field, ast.AST):
                yield from iter_missing_docstrings(field)


def format_node(node: ast.AST):
    if isinstance(node, ast.Module):
        name_str = Path(args.file).stem
        help_str = f"{args.file}"
        type_str = "module"
    else:
        name_str = node.name
        help_str = f"{args.file} L{color(str(node.lineno), fg='white')}:{color(str(node.col_offset), fg='white')}"
        if isinstance(node, ast.ClassDef):
            type_str = "class"
        elif isinstance(node, ast.FunctionDef):
            type_str = "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            type_str = "coroutine"

    return f"{type_str} '{color(name_str, fg='cyan')}' ({help_str})"


with open(args.file, 'r') as f:
    source = f.read()

tree = ast.parse(source, mode='exec')

nodes = list(iter_missing_docstrings(tree))
if len(nodes) == 0:
    if not args.quiet:
        print("No members are missing docstrings.")
else:
    print("Members missing docstrings:")
    for node in nodes:
        print("    " + format_node(node))

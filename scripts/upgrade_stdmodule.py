""" A simple CLI script to upgrade a legacy modelpack to the new style.
Note that the resulting source code may require minor adjustment and reformatting.
"""

import argparse
import importlib
from pprint import pformat
import astor
import ast as _ast

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stdmodule', action="store",
                    help="Fully-qualified import path of the legacy modelpack to convert (e.g. pygsti.modelpacks.legacy.std1Q_XY)")
parser.add_argument('-r', '--relative', action="store_true", help="Use import paths relative to `pygsti.modelpacks`")
args = parser.parse_args()

# Prepare static analysis tools for upgrade
std = importlib.import_module(args.stdmodule)
ast = astor.code_to_ast.parse_file(std.__file__)


def name_or_none(name):
    return getattr(std, name) if hasattr(std, name) else None


def single_line(results):
    """ Pretty source generator for astor.to_source which will produce output on a single line, to be formatted at will """
    return ''.join(results)


def stringy_ast(obj):
    """ Build an ast for a stringly-typable object """
    return _ast.parse(str(obj)).body[0].value


mdl = std.target_model()
if mdl.dim == 4:
    sslbls = [0]
    find_replace_state_space = {
        'Q0': 0
    }
    find_replace_labels = {
        'Gi': (),
        'Gx': ('Gx', 0),
        'Gy': ('Gy', 0),
        'Gz': ('Gz', 0),
        'Gn': ('Gn', 0)
    }
elif mdl.dim == 16:
    sslbls = [0, 1]
    find_replace_state_space = {
        'Q0': 0,
        'Q1': 1,
    }
    find_replace_labels = {
        'Gii': (),
        'Gxi': ('Gx', 0),
        'Gyi': ('Gy', 0),
        'Gzi': ('Gz', 0),
        'Gix': ('Gx', 1),
        'Giy': ('Gy', 1),
        'Giz': ('Gz', 1),
        'Gxx': ('Gxx', 0, 1),
        'Gxy': ('Gxy', 0, 1),
        'Gyx': ('Gxy', 0, 1),
        'Gyy': ('Gyy', 0, 1),
        'Gcnot': ('Gcnot', 0, 1),
        'Gcphase': ('Gcphase', 0, 1)
    }
else:
    raise ValueError(f"Unsupported model dimension: {mdl.dim}")


class DependencyWalker(astor.TreeWalk):
    """ Walks the AST, building a set of names on which the tree depends """
    def __init__(self, *args, **kwargs):
        self.dependencies = set()
        astor.TreeWalk.__init__(self, *args, **kwargs)

    def pre_Name(self):
        self.dependencies.add(self.cur_node.id)
        return False


class StateSpaceLabelWalker(astor.TreeWalk):
    """ Walks the AST, upgrading all state-space label strings """
    def pre_Str(self):
        upgrade = find_replace_state_space.get(self.cur_node.s, None)
        if upgrade is not None:
            # hacky
            self.replace(stringy_ast(upgrade))
        return False


class LabelWalker(astor.TreeWalk):
    """ Walks the AST, upgrading all operation labels """
    def pre_Str(self):
        upgrade = find_replace_labels.get(self.cur_node.s, None)
        if upgrade is not None:
            # hacky
            self.replace(stringy_ast(upgrade))
        return False


class ExpressionWalker(astor.TreeWalk):
    """ Walks the AST, upgrading all state space labels embedded in expressions """
    def pre_Str(self):
        for old, new in find_replace_state_space.items():
            # hyper-hacky
            self.cur_node.s = self.cur_node.s.replace(old, str(new))


class UpgradeWalker(astor.TreeWalk):
    """ Walks the AST, building a map of upgraded names and their dependencies """
    def __init__(self, *args, **kwargs):
        self.upgrades = {}
        self.dependencies = {}
        astor.TreeWalk.__init__(self, *args, **kwargs)

    def pre_Assign(self):
        value = self.cur_node.value
        for target in self.cur_node.targets:
            # ignore Subscripts for now
            if isinstance(target, _ast.Name):
                if target.id == '_target_model':
                    assert(isinstance(value, _ast.Call))
                    # done in three separate phases for safety with stringly-typed arguments
                    StateSpaceLabelWalker(value.args[0])
                    LabelWalker(value.args[1])
                    ExpressionWalker(value.args[2])
                else:
                    # "hey, it can't hurt..."
                    LabelWalker(value)

                self.upgrades[target.id] = value

                # Build dependencies from the rhs
                # We don't consider the function name a dependency so bypass that
                critical_ast = value.args if isinstance(value, _ast.Call) else value
                self.dependencies[target.id] = DependencyWalker(critical_ast).dependencies
        return False

    def pre_Call(self):
        func = self.cur_node.func
        if isinstance(func, _ast.Attribute) and func.attr == 'circuit_list':
            # explicitly specify `line_labels` for calls to `_strc.circuit_list`
            if not any([kw.arg == 'line_labels' for kw in self.cur_node.keywords]):
                self.cur_node.keywords.append(_ast.keyword('line_labels', stringy_ast(sslbls)))

        return False


# Build variables for filling in the template
modelpack_import_path = "" if args.relative else "pygsti.modelpacks"
root_import_path = "." if args.relative else "pygsti"

# Handle Clifford compilation separately, it's defined strangely
# XXX weirdly enough, it looks like the original dynamic SMQ converter just drops Clifford compilations...
if hasattr(std, 'clifford_compilation'):
    cc_pairs = [(k, [find_replace_labels.get(lbl, lbl) for lbl in v]) for k, v in std.clifford_compilation.items()]
    cc_src = f"OrderedDict({cc_pairs})"
else:
    cc_src = None

# Don't upgrade these names
description = name_or_none('description')
global_fidPairs = name_or_none('global_fidPairs')
pergerm_fidPairsDict = name_or_none('pergerm_fidPairsDict')
global_fidPairs_lite = name_or_none('global_fidPairs_lite')
pergerm_fidPairsDict_lite = name_or_none('pergerm_fidPairsDict_lite')

ast_upgrades = UpgradeWalker(ast)


def upgrade_src(name):
    """ Shortcut to generate source for an upgraded name """
    upgrade_ast = ast_upgrades.upgrades.get(name, stringy_ast(None))
    return astor.to_source(upgrade_ast, pretty_source=single_line)


# Include extra names where they're depended upon by the critical names included by default
critical_names = ['germs', 'germs_lite', 'prepStrs', 'effectStrs', 'fiducials', 'clifford_compilation', '_target_model']
extra_names = [ast_upgrades.dependencies.get(name, set()) for name in critical_names]
extra_names = set().union(*extra_names).difference(critical_names)

# this is wild as hell
extra_names_src = ''.join([
    astor.to_source(
        _ast.Assign(
            [_ast.Name(name)],
            ast_upgrades.upgrades.get(name, stringy_ast(None))
        ),
        pretty_source=single_line) for name in extra_names
])

# Dead simple: just fill in a template string for the new-style module source
template_str = f"""\"""{std.__doc__}\"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict
from {root_import_path}.construction import circuitconstruction as _strc
from {root_import_path}.construction import modelconstruction as _setc

from {modelpack_import_path}._modelpack import SMQModelPack

{extra_names_src}

class _Module(SMQModelPack):
    description = "{description}"

    gates = {upgrade_src('gates')}
    germs = {upgrade_src('germs')}
    germs_lite = {upgrade_src('germs_lite')}
    fiducials = {upgrade_src('fiducials')}
    prepStrs = {upgrade_src('prepStrs')}
    effectStrs = {upgrade_src('effectStrs')}
    clifford_compilation = {cc_src}
    global_fidPairs = {global_fidPairs}
    pergerm_fidPairsDict = {pergerm_fidPairsDict}
    global_fidPairs_lite = {global_fidPairs_lite}
    pergerm_fidPairsDict_lite = {pergerm_fidPairsDict_lite}

    @property
    def _target_model(self):
        return {upgrade_src('_target_model')}

import sys
sys.modules[__name__] = _Module()
"""

# Dump it to stdout
print(template_str)

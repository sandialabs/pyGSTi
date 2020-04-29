""" The TreeNode class """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import json as _json
import pathlib as _pathlib
import copy as _copy

from .. import io as _io


class TreeNode(object):
    """ TODO: docstring
    A base class for representing an object that lives "at" a filesystem directory
    """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None):
        raise NotImplementedError("Derived classes should implement from_dir(...)!")
        # *** This is a template function, showing how derived classes should implement from_dir ***
        # ret = cls.__new__(cls)  # create the object to return
        # # <perform derived class initialization, so that the 'parent' passed
        # #  to child nodes' from_dir(...) is as initialized as possible >
        # subdir_between_dirname_and_meta_json = None  # typically this identifies a particular derived class
        # ret._init_children(dirname, subdir_between_dirname_and_meta_json)  # loads child nodes
        # return ret

    def __init__(self, possible_child_name_dirs, child_values=None):
        self._dirs = possible_child_name_dirs  # maps possible child keys -> subdir name
        self._vals = child_values if child_values else {}

    def _init_children(self, dirname, meta_subdir=None, **kwargs):
        dirname = _pathlib.Path(dirname)
        edesign_dir = dirname / 'edesign'  # because subdirs.json is always & only in 'edesign'
        with open(edesign_dir / 'subdirs.json', 'r') as f:
            meta = _json.load(f)

        child_dirs = {}
        for d, nm in meta.get('directories', {}).items():
            if isinstance(nm, list): nm = tuple(nm)  # because json makes tuples->lists
            child_dirs[nm] = d

        self._dirs = child_dirs
        self._vals = {}

        for nm, subdir in child_dirs.items():
            subobj_dir = dirname / subdir
            #if meta_subdir:
            submeta_dir = subobj_dir / meta_subdir
            if submeta_dir.exists():  # It's ok if not all possible sub-nodes exist
                self._vals[nm] = _io.cls_from_meta_json(submeta_dir).from_dir(subobj_dir, parent=self,
                                                                              name=nm, **kwargs)
            #else:  # if meta_subdir is None, we default to the same class as self
            #    self._vals[nm] = self.__class__.from_dir(subobj_dir, parent=self, name=nm)

    def keys(self):
        return self._dirs.keys()

    def __contains__(self, key):
        return key in self._dirs

    def __len__(self):
        return len(self._dirs)

    def items(self):
        for k in self._dirs:
            yield k, self[k]

    def __getitem__(self, key):
        if key not in self._dirs:
            raise KeyError("Invalid key: %s" % key)
        if key not in self._vals:
            self._vals[key] = self._create_childval(key)
        return self._vals[key]

    def _create_childval(self, key):
        raise NotImplementedError("Derived class needs to implement _create_childval to create valid key: %s" % key)

    def get_tree_paths(self):
        """Dictionary paths leading to data objects/nodes beneath this one"""
        paths = [()]  # path to self
        for child_name, child_node in self.items():
            paths.extend([(child_name,) + pth for pth in child_node.get_tree_paths()])
        return paths

    def view(self, keys_to_keep):
        if len(keys_to_keep) == 0: return self
        view = _copy.deepcopy(self)  # is deep copy really needed here??
        view._dirs = {k: self._dirs[k] for k in keys_to_keep}
        view._vals = {k: self[k] for k in keys_to_keep}
        return view

    def filter_paths(self, paths, paths_are_sorted=False):
        sorted_paths = paths if paths_are_sorted else sorted(paths)
        npaths = len(sorted_paths)

        if npaths == 1 and len(sorted_paths[0]) == 0:
            return self  # special case when this TreeNode itself is selected

        i = 0
        children_to_keep = {}
        while i < npaths:
            assert(len(sorted_paths[i]) > 0), \
                "Cannot select a TreeNode *and* some/all of its elements using filter_paths!"
            ky = sorted_paths[i][0]

            paths_starting_with_ky = []
            while i < npaths and sorted_paths[i][0] == ky:
                paths_starting_with_ky.append(sorted_paths[i][1:])
                i += 1
            children_to_keep[ky] = self[ky].filter_paths(paths_starting_with_ky, True)

        #assert(len(children_to_keep) > 0)
        view = _copy.deepcopy(self)  # copies type of this tree node
        view._dirs = {k: self._dirs[k] for k in children_to_keep}
        view._vals = children_to_keep
        return view

    def write(self, dirname, parent=None):
        raise NotImplementedError("Derived classes should implement write(...)!")
        # *** This is a template showing how derived classes should implement write(...) ***
        # # <write derived-class specific data to dirname>
        # #  write_subdir_json = True  # True only for the "master" type that defines the directory keys ('edesign')
        # self.write_children(dirname, write_subdir_json)

    def write_children(self, dirname, write_subdir_json=True):
        dirname = _pathlib.Path(dirname)

        if write_subdir_json:
            subdirs = {}
            subdirs['directories'] = {dirname: subname for subname, dirname in self._dirs.items()}
            # write self._dirs "backwards" b/c json doesn't allow tuple-like keys (sometimes keys are tuples)
            with open(dirname / 'edesign' / 'subdirs.json', 'w') as f:
                _json.dump(subdirs, f)

        for nm, val in self._vals.items():  # only write *existing* values
            subdir = self._dirs[nm]
            outdir = dirname / subdir
            outdir.mkdir(exist_ok=True)
            self._vals[nm].write(outdir, parent=self)

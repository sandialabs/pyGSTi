"""
The TreeNode class
"""
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
    """
    A base class for representing an object that lives "at" a filesystem directory.

    Parameters
    ----------
    possible_child_name_dirs : dict
        A dictionary with string keys and values that maps possible child names
        (keys of this `TreeNode`) to directory names (where those keys are stored).

    child_values : dict, optional
        A dictionary of child values (may be other `TreeNode` objects).
    """

    @classmethod
    def from_dir(cls, dirname, parent=None, name=None):
        """
        Load a :class:`TreeNode` from the data rooted at `dirname`.

        Parameters
        ----------
        dirname : str or Path
            The directory path to load from.

        parent : TreeNode, optional
            This node's parent node, if it's already loaded.

        name : immutable, optional
            The name of this node, usually a string or tuple.  Almost always the key
            within `parent` that refers to the loaded node (this can be different
            from the directory name).
        """
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
        with open(str(edesign_dir / 'subdirs.json'), 'r') as f:
            meta = _json.load(f)

        child_dirs = {}
        for d, nm in meta.get('directories', {}).items():
            if isinstance(nm, list): nm = tuple(nm)  # because json makes tuples->lists
            child_dirs[nm] = d

        self._dirs = child_dirs
        self._vals = {}

        for nm, subdir in child_dirs.items():
            subobj_dir = dirname / subdir
            if not subobj_dir.exists():  # if there's no subdirectory, generate the child value
                continue  # don't load anything - create child value on demand

            if meta_subdir:
                submeta_dir = subobj_dir / meta_subdir
                if submeta_dir.exists():  # then load an object from this directory (even if it's meta.json is missing)
                    # if we can't find a meta.json - default to same class as self
                    classobj = _io.metadir._cls_from_meta_json(submeta_dir) \
                        if (submeta_dir / 'meta.json').exists() else self.__class__
                    self._vals[nm] = classobj.from_dir(subobj_dir, parent=self, name=nm, **kwargs)
                # **If there's no subdirectory, don't load a value here - generate a child value if needed**
            else:
                instance = self.__class__  # no meta.json - default to same class as self
                self._vals[nm] = instance.from_dir(subobj_dir, parent=self, name=nm, **kwargs)

    def keys(self):
        """
        An iterator over the keys (child names) of this tree node.
        """
        return self._dirs.keys()

    def __contains__(self, key):
        return key in self._dirs

    def __len__(self):
        return len(self._dirs)

    def items(self):
        """
        An iterator over the `(child_name, child_node)` pairs of this node.
        """
        for k in self._dirs:
            yield k, self[k]

    def __getitem__(self, key):
        if key not in self._dirs:
            raise KeyError("Invalid key: %s" % str(key))
        if key not in self._vals:
            self._vals[key] = self._create_childval(key)
        return self._vals[key]

    def _create_childval(self, key):
        raise NotImplementedError("Derived class needs to implement _create_childval to create valid key: %s" % key)

    def underlying_tree_paths(self):
        """
        Dictionary paths leading to data objects/nodes beneath this one.

        Returns
        -------
        list
            A list of tuples, each specifying the tree traversal to a child node.
            The first tuple is the empty tuple, referring to *this* (root) node.
        """
        paths = [()]  # path to self
        for child_name, child_node in self.items():
            paths.extend([(child_name,) + pth for pth in child_node.underlying_tree_paths()])
        return paths

    def view(self, keys_to_keep):
        """
        Get a "view" of this tree node that only has a subset of this node's children.

        Parameters
        ----------
        keys_to_keep : iterable
            A sequence of key names to keep.

        Returns
        -------
        TreeNode
        """
        if len(keys_to_keep) == 0: return self
        view = _copy.deepcopy(self)  # is deep copy really needed here??
        view._dirs = {k: self._dirs[k] for k in keys_to_keep}
        view._vals = {k: self[k] for k in keys_to_keep}
        return view

    def prune_tree(self, paths, paths_are_sorted=False):
        """
        Prune the tree rooted here to include only the given paths, discarding all other leaves & branches.

        Parameters
        ----------
        paths : list
            A list of tuples specifying the paths to keep.

        paths_are_sorted : bool, optional
            Whether `paths` is sorted (lexographically).  Setting this to `True` will save
            a little time.

        Returns
        -------
        TreeNode
            A view of this node and its descendants where unwanted children have been removed.
        """
        sorted_paths = paths if paths_are_sorted else sorted(paths)
        npaths = len(sorted_paths)

        if npaths == 1 and len(sorted_paths[0]) == 0:
            return self  # special case when this TreeNode itself is selected

        i = 0
        children_to_keep = {}
        while i < npaths:
            assert(len(sorted_paths[i]) > 0), \
                "Cannot select a TreeNode *and* some/all of its elements using prune_tree!"
            ky = sorted_paths[i][0]

            paths_starting_with_ky = []
            while i < npaths and sorted_paths[i][0] == ky:
                paths_starting_with_ky.append(sorted_paths[i][1:])
                i += 1
            children_to_keep[ky] = self[ky].prune_tree(paths_starting_with_ky, True)

        #assert(len(children_to_keep) > 0)
        view = _copy.deepcopy(self)  # copies type of this tree node
        view._dirs = {k: self._dirs[k] for k in children_to_keep}
        view._vals = children_to_keep
        return view

    def write(self, dirname, parent=None):
        """
        Write this tree node to a directory.

        Parameters
        ----------
        dirname : str or Path
            Directory to write to.

        parent : TreeNode, optional
            This node's parent.

        Returns
        -------
        None
        """
        raise NotImplementedError("Derived classes should implement write(...)!")
        # *** This is a template showing how derived classes should implement write(...) ***
        # # <write derived-class specific data to dirname>
        # #  write_subdir_json = True  # True only for the "master" type that defines the directory keys ('edesign')
        # self.write_children(dirname, write_subdir_json)

    def _write_children(self, dirname, write_subdir_json=True):
        """
        Writes this node's children to directories beneath `dirname`.

        Each child node is written to a sub-directory named according to the
        sub-directory names associated with the child names (keys) of this node.

        Parameters
        ----------
        dirname : str or Path
            The root directory to write to.

        write_subdir_json : bool, optional
            If `True`, a `dirname/edesign/subdirs.json` file is written that
            contains child name information, i.e. the map between directory names
            and child names (it is useful to *not* requires these be the same,
            and sometimes it's useful to name children with a tuple rather than
            just a string).

        Returns
        -------
        None
        """
        dirname = _pathlib.Path(dirname)

        if write_subdir_json:
            subdirs = {}
            subdirs['directories'] = {dirname: subname for subname, dirname in self._dirs.items()}
            # write self._dirs "backwards" b/c json doesn't allow tuple-like keys (sometimes keys are tuples)
            with open(str(dirname / 'edesign' / 'subdirs.json'), 'w') as f:
                _json.dump(subdirs, f)

        for nm, val in self._vals.items():  # only write *existing* values
            subdir = self._dirs[nm]
            outdir = dirname / subdir
            outdir.mkdir(exist_ok=True)
            self._vals[nm].write(outdir, parent=self)

"""
Modelmember dependency graph related utility functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import OrderedDict

from pygsti.models.memberdict import OrderedMemberDict
from pygsti.modelmembers.modelmember import ModelMember

class ModelMemberGraph(object):
    """A directed acyclic graph of dependencies of ModelMembers"""

    @classmethod
    def load_modelmembers_from_serialization_dict(cls, sdict):
        """Create a nested dictionary of model members from a previously serialized graph.
        
        Parameters
        ----------
        sdict: dict
            Flat dict of the ModelMemberGraph that was serialized by a
            prior call to :method:`ModelMemberGraph.create_serialization_dict`.

        Returns
        -------
        dict
        """
        from pygsti.io.metadir import _from_memoized_dict
        from pygsti.circuits.circuitparser import parse_label as _parse_label

        mm_nodes = {}
        mm_serial = {}

        for mm_node_serialized_id_str, mm_node_dict in sdict.items():
            mm_node_serialized_id = int(mm_node_serialized_id_str)  # convert string keys back to integers
            mm_serial[mm_node_serialized_id] = _from_memoized_dict(mm_node_dict, mm_serial, underscore=False)
            if 'memberdict_type' in mm_node_dict and 'memberdict_label' in mm_node_dict:
                mm_type, lbl = mm_node_dict['memberdict_type'], _parse_label(mm_node_dict['memberdict_label'])
                if mm_type not in mm_nodes:
                    mm_nodes[mm_type] = {}
                mm_nodes[mm_type][lbl] = mm_serial[mm_node_serialized_id]
        
        return mm_nodes
    
    def __init__(self, mm_dicts):
        """Generate a directed acyclic graph of ModelMember dependencies for an OpModel.

        All ModelMembers are copied, so a new ModelMemberGraph must be constructed
        after any changes to the model to accurately reflect parameterization type and values.

        Parameters
        ----------
        mm_dicts: dict of OrderedMemberDicts
            Dictionary where keys are attribute names and values are OrderedMemberDicts
            from an OpModel, e.g. {'preps': self.preps, etc.} from an ExplicitOpModel
        """
        if not isinstance(mm_dicts, dict) and not all([isinstance(v, OrderedMemberDict) for v in mm_dicts.values()]):
            raise ValueError("Dependency graph requires a dict of attribute name: OrderedMemberDict")

        # Memo for MMNodes (OrderedDict for insertion-order in pre-3.6, since we still support 3.5)
        self.mm_memo = OrderedDict()
        
        # Dict holding all "roots" of the graph
        self.mm_nodes = OrderedDict()
        for mm_type, mm_dict in mm_dicts.items():
            self.mm_nodes[mm_type] = OrderedDict()
            for lbl, mm in mm_dict.items():
                if id(mm) in self.mm_memo:
                    self.mm_nodes[mm_type][lbl] = self.mm_memo[id(mm)]
                else:
                    self.mm_nodes[mm_type][lbl] = MMGNode(mm, self.mm_memo)

    def is_similar(self, other):
        """Comparator between two ModelMemberGraph objects for structure only.
        
        Parameters
        ----------
        other: ModelMemberGraph
            Dependency graph to compare to
        
        Returns
        -------
        bool
            True if both ModelMemberGraphs have identical modelmember structure/parameterization
        """
        return self._dfs_comparison(other, False)

    def is_equivalent(self, other):
        """Comparator between two ModelMemberGraph objects for structure and values.

        Parameters
        ----------
        other: ModelMemberGraph
            Dependency graph to compare to
        
        Returns
        -------
        bool
            True if similar_to AND parameter vectors match
        """
        return self._dfs_comparison(other, True)

    def _dfs_comparison(self, other, check_params):
        """Helper function for comparators implementing DFS traversal.
    
        Parameters
        ----------
        other: ModelMemberGraph
            Dependency graph to compare to
        check_params: bool
            Whether to check the parameter values (True) or not
        
        Returns
        -------
        bool
            True if graphs match (structure/parameterization must match,
            param values must also match if check_params is True)
        """
        # Recursive check function
        def dfs_compare(node1, node2):
            if check_params:
                if not node1.mm.is_equivalent(node2.mm): return False
            else:
                if not node1.mm.is_similar(node2.mm): return False
            
            # Check children
            if len(node1.children) != len(node2.children): return False

            for c1, c2 in zip(node1.children, node2.children):
                if not dfs_compare(c1, c2): return False
            
            # If here, everything checks out in the subtree
            return True

        # Iterate over all types of model members
        for mm_type in self.mm_nodes.keys():
            if set(self.mm_nodes[mm_type].keys()) != set(other.mm_nodes[mm_type].keys()):
                return False

            # Iterate over all model members per type
            for lbl, node1 in self.mm_nodes[mm_type].items():
                node2 = other.mm_nodes[mm_type][lbl]
                if not dfs_compare(node1, node2):
                    return False
        
        # If here, everything checks out
        return True
    
    def create_serialization_dict(self):
        """Serialize the ModelMemberGraph object.
        
        Calls underlying to_memoized_dict on each ModelMember,
        as well as adding necessary metadata for the collection
        of ModelMembers - e.g., OrderedMemberDict keys for root nodes.

        Returns
        -------
        sdict: dict
            Flat dict of the ModelMemberGraph for serialization.
            Keys are serialize_ids, values are derefernced dicts for each
            individual ModelMember.
        """
        sdict = OrderedDict()
        for mm_id, mm_node in self.mm_memo.items():
            sdict[str(mm_node.serialize_id)] = mm_node.mm.to_memoized_dict(self.mm_memo)
        
        # Tag extra metadata for root nodes
        for mm_type, root_nodes in self.mm_nodes.items():
            for lbl, mm_node in root_nodes.items():
                sdict[str(mm_node.serialize_id)]['memberdict_type'] = mm_type
                sdict[str(mm_node.serialize_id)]['memberdict_label'] = str(lbl)
                # serialized-dictionary keys must be *strings*
        
        return sdict

    def print_graph(self, indent=0):
        def print_subgraph(node, indent=0, name=None):
            if name is not None:
                print(' '*indent + f'{name}: {node.mm.__class__.__name__} ({node.serialize_id})')
            else:
                print(' '*indent + f'{node.mm.__class__.__name__} ({node.serialize_id})')

            for child in node.children:
                print_subgraph(child, indent+2)

        for mm_type, mm_dict in self.mm_nodes.items():
            print(f'Modelmember type: {mm_type}')
            for name, node in mm_dict.items():
                print_subgraph(node, indent=2, name=name)


class MMGNode(object):
    """A basic graph node object for ModelMembers"""
    def __init__(self, mm: ModelMember, mm_memo):
        self.children = []
        for sm in mm.submembers():
            if id(sm) in mm_memo:
                self.children.append(mm_memo[id(sm)])
            else:
                subnode = MMGNode(sm, mm_memo)
                mm_memo[id(sm)] = subnode
                self.children.append(subnode)
        
        # TODO: Don't need this, just need serialized version?
        # BUt needs to be after recursive call so all children are in memo
        self.serialize_id = len(mm_memo)
        self.mm = mm
        mm_memo[id(mm)] = self
        
        # Determine depth (0 if leaf)
        if len(self.children) == 0:
            self.depth = 0
        else:
            self.depth = max([c.depth for c in self.children]) + 1
        
        

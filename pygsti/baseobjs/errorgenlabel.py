"""
Defines the ElementaryErrorgenLabel class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


def _to_int_or_strip(x):  # (same as in slowcircuitparser.py)
    return int(x) if x.strip().isdigit() else x.strip()


class ElementaryErrorgenLabel(object):
    """
    TODO: docstring - for entire module
    """
    pass


class LocalElementaryErrorgenLabel(ElementaryErrorgenLabel):
    """
    Labels an elementary error generator by simply a type and one or two
    basis element labels.
    """
    @classmethod
    def cast(cls, obj, sslbls=None, identity_label='I'):
        if isinstance(obj, LocalElementaryErrorgenLabel):
            return obj
        elif isinstance(obj, GlobalElementaryErrorgenLabel):
            assert(sslbls is not None), "Cannot convert global -> local elementary errogen label without `sslbls`!"
            indices_to_replace = [sslbls.index(sslbl) for sslbl in obj.sslbls]
            local_bels = []
            for global_lbl in obj.basis_element_labels:
                local_bel = [identity_label] * len(sslbls)
                for kk, k in enumerate(indices_to_replace):
                    local_bel[k] = global_lbl[kk]
                local_bels.append(''.join(local_bel))
            return cls(obj.errorgen_type, local_bels)
        elif isinstance(obj, str):
            if obj[1:].startswith('(') and obj.endswith(')'):  # e.g. "H(XX)" or "S(XY,YZ)" as from __str__
                bels = [x.strip() for x in obj[2:-1].split(',')]
                return cls(obj[0], bels)
            else:
                return cls(obj[0], (obj[1:],))  # e.g. "HXX" => ('H','XX')
        elif isinstance(obj, (tuple, list)):
            if len(obj) == 3 and all([isinstance(el, (tuple, list)) for el in obj[1:]]):
                # e.g. ('H', ('X',), (1,)) or other GlobalElementaryErrorgenLabel tuples
                assert(sslbls is not None), "Cannot convert global-like tuples -> local elementary errogen label without `sslbls`!"
                indices_to_replace = [sslbls.index(sslbl) for sslbl in obj[2]]
                local_bels = []
                for global_lbl in obj[1]:
                    local_bel = [identity_label] * len(sslbls)
                    for kk, k in enumerate(indices_to_replace):
                        local_bel[k] = global_lbl[kk]
                    local_bels.append(''.join(local_bel))
                return cls(obj[0], local_bels)
            else:
                return cls(obj[0], obj[1:])  # e.g. ('H','XX') or ('S', 'X', 'Y')
        else:
            raise ValueError("Cannot convert %s to a local elementary errorgen label!" % str(obj))

    def __init__(self, errorgen_type, basis_element_labels):
        """
        Parameters
        ----------
        errorgen_type : str
            A string corresponding to the error generator sector this error generator label is
            an element of. Allowed values are 'H', 'S', 'C' and 'A'.

        basis_element_labels : tuple or list
            A list or tuple of strings labeling basis elements used to label this error generator.
            This is either length-1 for 'H' and 'S' type error generators, or length-2 for 'C' and 'A'
            type.
        """

        self.errorgen_type = str(errorgen_type)
        self.basis_element_labels = tuple(basis_element_labels)

    def __hash__(self):
        return hash((self.errorgen_type, self.basis_element_labels))

    def __eq__(self, other):
        return (self.errorgen_type == other.errorgen_type
                and self.basis_element_labels == other.basis_element_labels)

    def __str__(self):
        return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ")"

    def __repr__(self):
        return str((self.errorgen_type, self.basis_element_labels))
    
    def support_indices(self, identity_label='I'):
        """ 
        Returns a sorted tuple of the elements of indices of the nontrivial basis
        element label entries for this label.
        """
        nonidentity_indices = [i for i in range(len(self.basis_element_labels[0]))
                                   if any([bel[i] != identity_label for bel in self.basis_element_labels])]

        return tuple(nonidentity_indices)


class GlobalElementaryErrorgenLabel(ElementaryErrorgenLabel):
    """
    Labels an elementary error generator on n qubits that includes the state
    space labels on which the generator acts (unlike a "local" label, i.e.
    a :class:`LocalElementaryErrorgenLabel` which doesn't)
    """

    @classmethod
    def cast(cls, obj, sslbls=None, identity_label='I'):
        """ TODO: docstring - lots in this module """
        if isinstance(obj, GlobalElementaryErrorgenLabel):
            return obj
        elif isinstance(obj, LocalElementaryErrorgenLabel):
            assert(sslbls is not None), "Cannot convert local -> global elementary errogen label without `sslbls`!"
            nonidentity_indices = [i for i in range(len(sslbls))
                                   if any([bel[i] != identity_label for bel in obj.basis_element_labels])]
            global_bels = []
            for local_bel in obj.basis_element_labels:
                global_bels.append(''.join([local_bel[i] for i in nonidentity_indices]))

            return cls(obj.errorgen_type, global_bels, [sslbls[i] for i in nonidentity_indices])

        elif isinstance(obj, str):
            if obj[1:].startswith('(') and obj.endswith(')'):
                in_parens = obj[2:-1]
                if ':' in in_parens:  # e.g. "H(XX:Q0,Q1)" or "S(XY,YZ:0,1)" as from __str__
                    bel_str, sslbl_str = in_parens.split(':')
                    bels = [x.strip() for x in bel_str.split(',')]
                    sslbls = [_to_int_or_strip(x) for x in sslbl_str.split(',')]
                    return cls(obj[0], bels, sslbls)
                else:  # treat as a local label
                    return cls.cast(LocalElementaryErrorgenLabel.cast(obj), sslbls, identity_label)
            else:  # no parenthesis, assume of form "HXX:Q0,Q1" or local label, e.g. "HXX"
                if ':' in obj:
                    typ_bel_str, sslbl_str = in_parens.split(':')
                    sslbls = [_to_int_or_strip(x) for x in sslbl_str.split(',')]
                    return cls(typ_bel_str[0], (typ_bel_str[1:],), sslbls)
                else:  # treat as a local label
                    return cls.cast(LocalElementaryErrorgenLabel.cast(obj), sslbls, identity_label)

        elif isinstance(obj, (tuple, list)):
            # Allow a tuple-of-tuples format, e.g. ('S', ('XY', 'YZ'), ('Q0', 'Q1'))
            if isinstance(obj[1], (list, tuple)):  # distinguish vs. local labels
                return cls(obj[0], obj[1], obj[2])  # ('H', ('XX',), ('Q0', 'Q1'))
            else:  # e.g. ('H', 'XX')
                return cls.cast(LocalElementaryErrorgenLabel.cast(obj), sslbls, identity_label)
        else:
            raise ValueError("Cannot convert %s to a global elementary errorgen label!" % str(obj))

    def __init__(self, errorgen_type, basis_element_labels, sslbls, sort=True):
        """
        Parameters
        ----------
        errorgen_type : str
            A string corresponding to the error generator sector this error generator label is
            an element of. Allowed values are 'H', 'S', 'C' and 'A'.

        basis_element_labels : tuple or list
            A list or tuple of strings labeling basis elements used to label this error generator.
            This is either length-1 for 'H' and 'S' type error generators, or length-2 for 'C' and 'A'
            type.
        
        sslbls : tuple or list
            A tuple or list of state space labels corresponding to the qudits upon which this error generator
            is supported.

        sort : bool, optional (default True)
            If True then the input state space labels are first sorted, and then the used basis element labels
            are sorted to match the order to the newly sorted state space labels.
        """
        
        if sort:
            sorted_indices, sslbls = zip(*sorted(enumerate(sslbls), key=lambda x: x[1]))
            basis_element_labels = [''.join([bel[i] for i in sorted_indices]) for bel in basis_element_labels]

        self.errorgen_type = str(errorgen_type)
        self.basis_element_labels = tuple(basis_element_labels)
        self.sslbls = tuple(sslbls)
        # Note: each element of basis_element_labels must be an iterable over
        #  1-qubit basis labels of length len(self.sslbls) (?)

    def __hash__(self):
        return hash((self.errorgen_type, self.basis_element_labels, self.sslbls))

    def __eq__(self, other):
        return (self.errorgen_type == other.errorgen_type
                and self.basis_element_labels == other.basis_element_labels
                and self.sslbls == other.sslbls)

    def __str__(self):
        return self.errorgen_type + "(" + ",".join(map(str, self.basis_element_labels)) + ":" \
            + ",".join(map(str, self.sslbls)) + ")"

    def __repr__(self):
        return str((self.errorgen_type, self.basis_element_labels, self.sslbls))

    @property
    def support(self):
        """ Returns a sorted tuple of the elements of `self.sslbls` """
        return tuple(sorted(self.sslbls))

    def padded_basis_element_labels(self, all_sslbls, identity_label='I'):
        """
        Idle-padded versions of this label's basis element labels based on its state space labels.

        A tuple of strings which positions the non-trivial single-qubit labels within the
        elements of `self.basis_element_labels` into a background of `identity_label` characters.
        For example, if the ordering of `all_sslbls` is `(0, 1, 2)`, `self.sslbls` is `(1,)`, and
        `self.basis_element_labels` is `('X',)` then this method returns `('IXI',)`.

        For this method to work correctly, basis element labels should be composed of single
        characters corresponding to non-trivial single-qubit basis elements, and the total basis
        element should be a product of these along with the identity on the state space labels
        absent from `self.sslbls`.

        Parameters
        ----------
        all_sslbls : tuple
            An ordered list of the entirety of the state space labels to create padded basis
            element labels for.  For example, `(0, 1, 2)` or `('Q0', 'Q1', 'Q2')`.

        identity_label : str, optional
            The single-character label used to indicate the single-qubit identity operation.

        Returns
        -------
        tuple
            A tuple of strings.
        """
        ret = []
        all_sslbls = {lbl: i for i, lbl in enumerate(all_sslbls)}
        sslbl_indices = [all_sslbls[lbl] for lbl in self.sslbls]
        for bel in self.basis_element_labels:
            lbl = [identity_label] * len(all_sslbls)
            for i, char in zip(sslbl_indices, bel):
                lbl[i] = char
            ret.append(''.join(lbl))
        return tuple(ret)

    def map_state_space_labels(self, mapper):
        """
        Creates a new GlobalElementaryErrorgenLabel whose `sslbls` attribute is updated according to a mapping function.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing `self.sslbls` values
            and whose value are the new labels, or a function which takes a
            single existing state space label argument and returns a new state
            space label to replace it with.

        Returns
        -------
        GlobalElementaryErrorgenLabel
        """
        def mapper_func(sslbl): return mapper[sslbl] \
            if isinstance(mapper, dict) else mapper(sslbl)
        mapped_sslbls = tuple(map(mapper_func, self.sslbls))
        return GlobalElementaryErrorgenLabel(self.errorgen_type, self.basis_element_labels, mapped_sslbls)

    def sort_sslbls(self):
        """
        Creates a new GlobalElementaryErrorgenLabel with sorted (potentially reordered) state space labels.

        This puts the label into a canonical form that can be useful for comparison with other labels.

        Returns
        -------
        GlobalElementaryErrorgenLabel
        """
        sorted_indices, sorted_sslbls = zip(*sorted(enumerate(self.sslbls), key=lambda x: x[1]))
        sorted_bels = [''.join([bel[i] for i in sorted_indices]) for bel in self.basis_element_labels]
        return GlobalElementaryErrorgenLabel(self.errorgen_type, sorted_bels, sorted_sslbls)

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
            return cls(obj[0], (obj[1:],))  # e.g. "HXX" => ('H','XX')
        elif isinstance(obj, (tuple, list)):
            return cls(obj[0], obj[1:])  # e.g. ('H','XX') or ('S', 'X', 'Y')
        else:
            raise ValueError("Cannot convert %s to a local elementary errorgen label!" % str(obj))

    def __init__(self, errorgen_type, basis_element_labels):
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
        elif isinstance(obj, (str, tuple, list)):
            # TODO: maybe allow a convenient global format in future, e.g. "HXX:Q0,Q1"
            # that we would need to distinguish here
            return cls.cast(LocalElementaryErrorgenLabel.cast(obj), sslbls, identity_label)
        else:
            raise ValueError("Cannot convert %s to a global elementary errorgen label!" % str(obj))

    def __init__(self, errorgen_type, basis_element_labels, sslbls):
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

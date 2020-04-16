""" Utility functions for creating and acting on lists of operation sequences."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import itertools as _itertools
import numpy as _np
import numpy.random as _rndm

from ..tools import listtools as _lt
from ..objects import circuit as _cir
from ..objects import Model as _Model
from ..objects.label import Label as _Lbl


def _run_expression(str_expression, my_locals):
    exec("result = " + str_expression, {"__builtins__": None}, my_locals)
    return my_locals.get("result", None)


def create_circuit_list(*args, **kwargs):
    """
    Create a list of operation sequences using a nested loop.  Positional arguments
    specify evaluation strings, which are evaluated within the inner-loop
    for a nested loop over all list or tuple type keyword arguments.

    Parameters
    ----------
    args : list of strings
        Positional arguments are strings that python can evaluate into either
        a tuple of operation labels or a Circuit instance.  If evaluation raises
        an AssertionError (an assert statement fails) then that inner loop
        evaluation is skipped and list construction proceeds.

    kwargs : dict
        keys specify variable names that can be used in positional argument
        strings.

    Returns
    -------
    list of Circuit

    Examples
    --------
    >>> from pygsti.construction import create_circuit_list
    >>> As = [('a1',), ('a2',)]
    >>> Bs = [('b1',), ('b2',)]
    >>> list1 = create_circuit_list('a', 'a+b', a=As, b=Bs)
    >>> print(list(map(str, list1)))
    ['a1', 'a2', 'a1b1', 'a1b2', 'a2b1', 'a2b2']

    You can change the order in which the different iterables are advanced.

    >>> list2 = create_circuit_list('a+b', a=As, b=Bs, order=['a', 'b'])
    >>> print(list(map(str, list2)))
    ['a1b1', 'a1b2', 'a2b1', 'a2b2']
    >>> list3 = create_circuit_list('a+b', a=As, b=Bs, order=['b', 'a'])
    >>> print(list(map(str, list3)))
    ['a1b1', 'a2b1', 'a1b2', 'a2b2']


    """
    lst = []

    loopOrder = list(kwargs.pop('order', []))
    loopLists = {}; loopLocals = {'True': True, 'False': False, 'str': str, 'int': int, 'float': float}
    for key, val in kwargs.items():
        if type(val) in (list, tuple):  # key describes a variable to loop over
            loopLists[key] = val
            if key not in loopOrder:
                loopOrder.append(key)
        else:  # callable(val): #key describes a function or variable to pass through to exec
            loopLocals[key] = val

    #print "DEBUG: looplists = ",loopLists
    for str_expression in args:
        if len(str_expression) == 0:
            lst.append(_cir.Circuit(())); continue  # special case

        keysToLoop = [key for key in loopOrder if key in str_expression]
        loopListsToLoop = [loopLists[key] for key in keysToLoop]  # list of lists
        for allVals in _itertools.product(*loopListsToLoop):
            myLocals = {key: allVals[i] for i, key in enumerate(keysToLoop)}
            myLocals.update(loopLocals)
            try:
                result = _run_expression(str_expression, myLocals)
            except AssertionError: continue  # just don't append

            if isinstance(result, _cir.Circuit):
                opStr = result
            elif isinstance(result, list) or isinstance(result, tuple):
                opStr = _cir.Circuit(result)
            elif isinstance(result, str):
                opStr = _cir.Circuit(None, stringrep=result)
            lst.append(opStr)

    return lst


def repeat(x, n_times, assert_at_least_one_rep=False):
    """
    Repeat x n_times times.

    Parameters
    ----------
    x : tuple or Circuit
       the operation sequence to repeat

    n_times : int
       the number of times to repeat x

    assert_at_least_one_rep : bool, optional
       if True, assert that n_times > 0.  This can be useful when used
       within a create_circuit_list inner loop to build a operation sequence
       lists where a string must be repeated at least once to be added
       to the list.

    Returns
    -------
    tuple or Circuit (whichever x was)
    """
    if assert_at_least_one_rep: assert(n_times > 0)
    return x * n_times


def repeat_count_with_max_length(x, max_length, assert_at_least_one_rep=False):
    """
    Compute the number of times a operation sequence x must be repeated such that
    the repeated string has length <= max_length.

    Parameters
    ----------
    x : tuple or Circuit
       the operation sequence to repeat

    max_length : int
       the maximum length

    assert_at_least_one_rep : bool, optional
       if True, assert that number of repetitions is > 0.
       This can be useful when used within a create_circuit_list inner loop
       to build a operation sequence lists where a string must be repeated at
       least once to be added to the list.

    Returns
    -------
    int
      the number of repetitions.
    """
    l = len(x)
    if assert_at_least_one_rep: assert(l <= max_length)
    reps = max_length // l if l > 0 else 0
    return reps


def repeat_with_max_length(x, max_length, assert_at_least_one_rep=False):
    """
    Repeat the operation sequence x an integer number of times such that
    the repeated string has length <= max_length.

    Parameters
    ----------
    x : tuple or Circuit
       the operation sequence to repeat.

    max_length : int
       the maximum length.

    assert_at_least_one_rep : bool, optional
       if True, assert that number of repetitions is > 0.
       This can be useful when used within a create_circuit_list inner loop
       to build a operation sequence lists where a string must be repeated at
       least once to be added to the list.

    Returns
    -------
    tuple or Circuit (whichever x was)
        the repeated operation sequence
    """
    return repeat(x, repeat_count_with_max_length(x, max_length, assert_at_least_one_rep), assert_at_least_one_rep)

#Useful for anything?
#def repeat_empty(x,max_length,assert_at_least_one_rep=False):
#    return ()


def repeat_and_truncate(x, n, assert_at_least_one_rep=False):
    """
    Repeat the operation sequence x so the repeated string has length greater than n,
    then truncate the string to be exactly length n.

    Parameters
    ----------
    x : tuple or Circuit
       the operation sequence to repeat & truncate.

    n : int
       the truncation length.

    assert_at_least_one_rep : bool, optional
       if True, assert that number of repetitions is > 0.
       This is always the case when x has length > 0.

    Returns
    -------
    tuple or Circuit (whichever x was)
        the repeated-then-truncated operation sequence
    """
    reps = repeat_count_with_max_length(x, n, assert_at_least_one_rep) + 1
    return (x * reps)[0:n]


def repeat_remainder_for_truncation(x, n, assert_at_least_one_rep=False):
    """
    Repeat the operation sequence x the fewest number of times such that the repeated
    string has length greater than or equal to n.  Return the portion of this
    repeated string from the n-th position to the end. Note that this corresponds
    to what is truncated in a call to repeateAndTruncate(x,n,assert_at_least_one_rep).

    Parameters
    ----------
    x : tuple or Circuit
       the operation sequence to operate on.

    n : int
       the truncation length.

    assert_at_least_one_rep : bool, optional
       if True, assert that number of repetitions is > 0.
       This is always the case when x has length > 0.

    Returns
    -------
    tuple or Circuit (whichever x was)
        the remainder operation sequence

    """
    reps = repeat_count_with_max_length(x, n, assert_at_least_one_rep)
    return x[0:(n - reps * len(x))]


def simplify_str(circuit_str):
    """
    Simplify a string representation of a operation sequence.  The simplified
      string should evaluate to the same operation label tuple as the original.

    Parameters
    ----------
    circuit_str : string
        the string representation of a operation sequence to be simplified.
        (e.g. "Gx{}", "Gy^1Gx")

    Returns
    -------
    string
        the simplified string representation.
    """
    s = circuit_str.replace("{}", "")
    s = s.replace("^1G", "G")
    s = s.replace("^1(", "(")
    s = s.replace("^1{", "{")
    if s.endswith("^1"): s = s[:-2]
    return s if len(s) > 0 else "{}"


## gate-label-tuple function.  TODO: check if these are still needed.

def list_all_circuits(op_labels, minlength, maxlength):
    """
    List all the operation sequences in a given length range.

    Parameters
    ----------
    op_labels : tuple
        tuple of operation labels to include in operation sequences.

    minlength : int
        the minimum operation sequence length to return

    maxlength : int
        the maximum operation sequence length to return

    Returns
    -------
    list
        A list of Circuit objects.
    """
    opTuples = _itertools.chain(*[_itertools.product(op_labels, repeat=N)
                                  for N in range(minlength, maxlength + 1)])
    return list(map(_cir.Circuit, opTuples))


def gen_all_circuits(op_labels, minlength, maxlength):
    """ Generator version of list_all_circuits """
    opTuples = _itertools.chain(*[_itertools.product(op_labels, repeat=N)
                                  for N in range(minlength, maxlength + 1)])
    for opTuple in opTuples:
        yield _cir.Circuit(opTuple)


def list_all_circuits_onelen(op_labels, length):
    """
    List all the operation sequences of a given length.

    Parameters
    ----------
    op_labels : tuple
        tuple of operation labels to include in operation sequences.

    length : int
        the operation sequence length

    Returns
    -------
    list
        A list of Circuit objects.
    """
    opTuples = _itertools.product(op_labels, repeat=length)
    return list(map(_cir.Circuit, opTuples))


def gen_all_circuits_onelen(op_labels, length):
    """Generator version of list_all_circuits_onelen"""
    for opTuple in _itertools.product(op_labels, repeat=length):
        yield _cir.Circuit(opTuple)


def list_all_circuits_without_powers_and_cycles(op_labels, max_length):
    """
    Generate all distinct operation sequences up to a maximum length that are
    aperiodic, i.e., that are not a shorter gate sequence raised to a power,
    and are also distinct up to cycling (e.g. `('Gx','Gy','Gy')` and
    `('Gy','Gy','Gx')` are considered equivalent and only one would be
    included in the returned list).

    Parameters
    ----------
    op_labels : list
        A list of the operation (gate) labels to form operation sequences from.

    max_length : int
        The maximum length strings to return.  Circuits from length 1
        to `max_length` will be returned.

    Returns
    -------
    list
       Of :class:`Circuit` objects.
    """

    #Are we trying to add a germ that is a permutation of a germ we already have?  False if no, True if yes.
    def _perm_check(test_str, str_list):  # works with python strings, so can use "in" to test for substring inclusion
        return any([test_str in s * 2 for s in str_list])

    #Are we trying to add a germ that is a power of a germ we already have?  False if no, True if yes.
    def _pow_check(test_str, str_list_dict):
        L = len(test_str)
        for k in list(str_list_dict.keys()):
            if L % k == 0:
                rep = L // k
                if any([test_str == s * rep for s in str_list_dict[k]]):
                    return True
        return False

    outputDict = {}
    for length in _np.arange(1, max_length + 1):

        permCheckedStrs = []
        for s in gen_all_circuits_onelen(op_labels, length):
            pys = s.to_pythonstr(op_labels)
            if not _perm_check(pys, permCheckedStrs):  # Sequence is not a cycle of anything in permCheckedStrs
                permCheckedStrs.append(pys)

        outputDict[length] = []
        for pys in permCheckedStrs:
            # Now check to see if any elements of tempList2 are powers of elements already in output
            if not _pow_check(pys, outputDict):  # Seqeunce is not a power of anything in output
                outputDict[length].append(pys)

    output = []
    for length in _np.arange(1, max_length + 1):
        output.extend([_cir.Circuit.from_pythonstr(pys, op_labels) for pys in outputDict[length]])
    return output


def list_random_circuits_onelen(op_labels, length, count, seed=None):
    """
    Create a list of random operation sequences of a given length.

    Parameters
    ----------
    op_labels : tuple
        tuple of operation labels to include in operation sequences.

    length : int
        the operation sequence length.

    count : int
        the number of random strings to create.

    seed : int, optional
        If not None, a seed for numpy's random number generator.


    Returns
    -------
    list of Circuits
        A list of random operation sequences as Circuit objects.
    """
    ret = []
    rndm = _rndm.RandomState(seed)  # ok if seed is None
    op_labels = list(op_labels)  # b/c we need to index it below
    for i in range(count):  # pylint: disable=unused-variable
        r = rndm.random_sample(length) * len(op_labels)
        ret.append(_cir.Circuit([op_labels[int(k)] for k in r]))
    return ret


def list_partial_strings(circuit):
    """
    List the parial strings of circuit, that is,
      the strings that are the slices circuit[0:n]
      for 0 <= l <= len(circuit).

    Parameters
    ----------
    circuit : tuple of operation labels or Circuit
        The operation sequence to act upon.

    Returns
    -------
    list of Circuit objects.
       The parial operation sequences.
    """
    ret = []
    for l in range(len(circuit) + 1):
        ret.append(tuple(circuit[0:l]))
    return ret


def list_lgst_circuits(prep_strs, effect_strs, op_label_src):
    """
    List the operation sequences required for running LGST.

    Parameters
    ----------
    prep_strs,effect_strs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    op_label_src : tuple or Model
        List/tuple of operation labels OR a Model whose gate and instrument
        labels should be used.

    Returns
    -------
    list of Circuit objects
        The list of required operation sequences, without duplicates.
    """
    def tolabel(x): return x if isinstance(x, _Lbl) else _Lbl(x)
    if isinstance(op_label_src, _Model):
        opLabels = list(op_label_src.operations.keys()) + \
            list(op_label_src.instruments.keys())
    else: opLabels = list(map(tolabel, op_label_src))

    line_labels = prep_strs[0].line_labels if len(prep_strs) > 0 else 'auto'
    if line_labels is None or len(line_labels) == 0: line_labels = ('*',)
    singleOps = [_cir.Circuit((gl,), line_labels=line_labels)**1 for gl in opLabels]  # **1 adds parens to stringrep
    ret = create_circuit_list('eStr', 'prepStr', 'prepStr+eStr', 'prepStr+g+eStr',
                              eStr=effect_strs, prepStr=prep_strs, g=singleOps,
                              order=['g', 'prepStr', 'eStr'])  # LEXICOGRAPHICAL VS MATRIX ORDER
    return _lt.remove_duplicates(ret)


def list_strings_lgst_can_estimate(dataset, prep_strs, effect_strs):
    """
      Compute the operation sequences that LGST is able to estimate
      given a set of fiducial strings.

      Parameters
      ----------
      dataset : DataSet
          The data used to generate the LGST estimates

      prep_strs,effect_strs : list of Circuits
          Fiducial Circuit lists used to construct a informationally complete
          preparation and measurement.

      Returns
      -------
      list of lists of tuples
         each list of tuples specifyies a operation sequence that LGST can estimate.

    """

    estimatable = []
    circuits = list(dataset.keys())
    pre = tuple(effect_strs[0]); l0 = len(pre)  # the first effect string
    post = tuple(prep_strs[0]); l1 = len(post)  # the first prep string

    def _root_is_ok(root_str):
        for estr in effect_strs:
            for rhostr in prep_strs:
                if tuple(rhostr) + tuple(root_str) + tuple(estr) not in circuits:  # LEXICOGRAPHICAL VS MATRIX ORDER
                    return False
        return True

    #check if string has first fiducial at beginning & end, and if so
    # strip that first fiducial off, leaving a 'root' string that we can test
    for s in circuits:
        if s[0:l0] == pre and s[len(s) - l1:] == post:
            root = s[l0:len(s) - l1]
            if _root_is_ok(root):
                estimatable.append(root)

    return circuit_list(estimatable)


def circuit_list(list_of_op_label_tuples_or_strings, line_labels="auto"):
    """
    Converts a list of operation label tuples or strings to
     a list of Circuit objects.

    Parameters
    ----------
    list_of_op_label_tuples_or_strings : list
        List which may contain a mix of Circuit objects, tuples of gate
        labels, and strings in standard-text-format.

    line_labels : "auto" or tuple, optional
        The line labels to use when creating Circuit objects from *non-Circuit*
        elements of `list_of_op_label_tuples_or_strings`.  If `"auto"` then the
        line labels are determined automatically based on the line-labels which
        are present in the layer labels.

    Returns
    -------
    list of Circuit objects
        Each item of list_of_op_label_tuples_or_strings converted to a Circuit.
    """
    ret = []
    for x in list_of_op_label_tuples_or_strings:
        if isinstance(x, _cir.Circuit):
            ret.append(x)
        elif isinstance(x, tuple) or isinstance(x, list):
            ret.append(_cir.Circuit(x, line_labels))
        elif isinstance(x, str):
            ret.append(_cir.Circuit(None, line_labels, stringrep=x))
        else:
            raise ValueError("Cannot convert type %s into a Circuit" % str(type(x)))
    return ret


def translate_circuit(circuit, alias_dict):
    """
    Creates a new Circuit object from an existing one by replacing
    operation labels in `circuit` by (possibly multiple) new labels according
    to `alias_dict`.

    Parameters
    ----------
    circuit : Circuit
        The operation sequence to use as the base for find & replace
        operations.

    alias_dict : dict
        A dictionary whose keys are single operation labels and whose values are
        lists or tuples of the new operation labels that should replace that key.
        If `alias_dict is None` then `circuit` is returned.

    Returns
    -------
    Circuit
    """
    if alias_dict is None:
        return circuit
    else:
        return _cir.Circuit(tuple(_itertools.chain(
            *[alias_dict.get(lbl, (lbl,)) for lbl in circuit])),
            line_labels=circuit.line_labels)


def translate_circuit_list(circuit_list, alias_dict):
    """
    Creates a new list of Circuit objects from an existing one by replacing
    operation labels in `circuit_list` by (possibly multiple) new labels according
    to `alias_dict`.

    Parameters
    ----------
    circuit_list : list of Circuits
        The list of operation sequences to use as the base for find & replace
        operations.

    alias_dict : dict
        A dictionary whose keys are single operation labels and whose values are
        lists or tuples of the new operation labels that should replace that key.
        If `alias_dict is None` then `circuit_list` is returned.

    Returns
    -------
    list of Circuits
    """
    if alias_dict is None:
        return circuit_list
    else:
        new_circuits = [_cir.Circuit(tuple(_itertools.chain(
            *[alias_dict.get(lbl, (lbl,)) for lbl in opstr])),
            line_labels=opstr.line_labels)  # line labels aren't allowed to change
            for opstr in circuit_list]
        return new_circuits


def compose_alias_dicts(alias_dict_1, alias_dict_2):
    """
    Composes two alias dicts.

    Assumes `alias_dict_1` maps "A" labels to "B" labels and `alias_dict_2` maps
    "B" labels to "C" labels.  The returned dictionary then maps "A" labels
    directly to "C" labels, and satisfies:

    `returned[A_label] = alias_dict_2[ alias_dict_1[ A_label ] ]`

    Parameters
    ----------
    alias_dict_1, alias_dict_2 : dict
        The two dictionaries to compose.

    Returns
    -------
    dict
    """
    ret = {}
    for A, Bs in alias_dict_1.items():
        ret[A] = tuple(_itertools.chain(*[alias_dict_2[B] for B in Bs]))
    return ret


def manipulate_circuit(circuit, sequence_rules, line_labels="auto"):
    """
    Manipulates a Circuit object according to `sequence_rules`.

    Each element of `sequence_rules` is of the form `(find,replace)`,
    and specifies a replacement rule.  For example,
    `('A',), ('B','C')` simply replaces each `A` with `B,C`.
    `('A', 'B'), ('A', 'B2'))` replaces `B` with `B2` when it follows `A`.
    `('B', 'A'), ('B2', 'A'))` replaces `B` with `B2` when it precedes `A`.

    Parameters
    ----------
    circuit : Circuit or tuple
        The operation sequence to manipulate.

    sequence_rules : list
        A list of `(find,replace)` 2-tuples which specify the replacement
        rules.  Both `find` and `replace` are tuples of operation labels
        (or `Circuit` objects).  If `sequence_rules is None` then
        `circuit` is returned.

    line_labels : "auto" or tuple, optional
        The line labels to use when creating a the output Circuit objects.
        If `"auto"` then the line labels are determined automatically based
        on the line-labels which are present in the corresponding layer labels.

    Returns
    -------
    list of Circuits
    """
    if sequence_rules is None:
        return circuit  # avoids doing anything to circuit

    # flag labels as modified so signal they cannot be processed
    # by any further rules
    circuit = tuple(circuit)  # make sure this is a tuple
    modified = _np.array([False] * len(circuit))
    actions = [[] for i in range(len(circuit))]

    #Step 0: compute prefixes and postfixes of rules
    ruleInfo = []
    for rule, replacement in sequence_rules:
        n_pre = 0  # length of shared prefix btwn rule & replacement
        for a, b in zip(rule, replacement):
            if a == b: n_pre += 1
            else: break
        n_post = 0  # length of shared prefix btwn rule & replacement (if no prefix)
        if n_pre == 0:
            for a, b in zip(reversed(rule), reversed(replacement)):
                if a == b: n_post += 1
                else: break
        n = len(rule)
        ruleInfo.append((n_pre, n_post, n))
        #print("Rule%d " % k, rule, "n_pre = ",n_pre," n_post = ",n_post) #DEBUG

    #print("Circuit = ",circuit) #DEBUG

    #Step 1: figure out which actions (replacements) need to be performed at
    # which indices.  During this step, circuit is unchanged, but regions
    # of it are marked as having been modified to avoid double-modifications.
    for i in range(len(circuit)):
        #print(" **** i = ",i) #DEBUG
        for k, (rule, replacement) in enumerate(sequence_rules):
            n_pre, n_post, n = ruleInfo[k]

            #if there's a match that doesn't double-modify
            if rule == circuit[i:i + n] and not any(modified[i + n_pre:i + n - n_post]):
                # queue this replacement action
                actions[i].append(k)
                #print("MATCH! ==> acting rule %d at index %d" % (k,i)) #DEBUG

                # and mark the modified region of the original string
                modified[i + n_pre:i + n - n_post] = True
        i += 1

    #Step 2: perform the actions (in reverse order so indices don't get messed up!)
    N = len(circuit)
    for i in range(N - 1, -1, -1):
        for k in actions[i]:
            #apply rule k at index i of circuit
            rule, replacement = sequence_rules[k]
            n_pre, n_post, n = ruleInfo[k]

            begin = circuit[:i + n_pre]
            repl = replacement[n_pre:len(replacement) - n_post]
            end = circuit[i + n - n_post:]

            circuit = begin + repl + end
            #print("Applied rule %d at index %d: " % (k,i), begin, repl, end, " ==> ", circuit) #DEBUG

    return _cir.Circuit(circuit, line_labels)


def manipulate_circuit_list(circuit_list, sequence_rules, line_labels="auto"):
    """
    Creates a new list of Circuit objects from an existing one by performing
    replacements according to `sequence_rules` (see :func:`manipulate_circuit`).

    Parameters
    ----------
    circuit_list : list of Circuits
        The list of operation sequences to use as the base for find & replace
        operations.

    sequence_rules : list
        A list of `(find,replace)` 2-tuples which specify the replacement
        rules.  Both `find` and `replace` are tuples of operation labels
        (or `Circuit` objects).  If `sequence_rules is None` then
        `circuit_list` is returned.

    line_labels : "auto" or tuple, optional
        The line labels to use when creating output Circuit objects.
        If `"auto"` then the line labels are determined automatically based on
        the line-labels which are present in the corresponding layer labels.

    Returns
    -------
    list of Circuits
    """
    if sequence_rules is None:
        return circuit_list
    else:
        return [manipulate_circuit(opstr, sequence_rules, line_labels) for opstr in circuit_list]


def filter_circuits(circuits, sslbls_to_keep, new_sslbls=None, drop=False, idle=()):
    """
    Removes any labels from `circuits` whose state-space labels are not
    entirely in `sslbls_to_keep`.  If a gates label's state-space labels
    (its `.sslbls`) is `None`, then the label is retained in the returned
    string.

    Furthermore, by specifying `new_sslbls` one can map the state-space
    labels in `sslbls_to_keep` to new labels (useful for "re-basing" a
    set of qubit strings.

    Parameters
    ----------
    circuits : list
        A list of operation sequences to act on.

    sslbls_to_keep : list
        A list of state space labels specifying which operation labels should
        be retained.

    new_sslbls : list, optional
        If not None, a list of the same length as `sslbls_to_keep` specifying
        a new set of state space labels to replace those in `sslbls_to_keep`.

    drop : bool, optional
        If True, then non-empty operation sequences which become empty after
        filtering are not included in (i.e. dropped from) the returned list.
        If False, then the returned list is always the same length as the
        input list.

    idle : string or Label, optional
        The operation label to be used when there are no kept components of a
        "layer" (element) of a circuit.

    Returns
    -------
    list
        A list of Circuits
    """
    if drop:
        ret = []
        for s in circuits:
            fs = filter_circuit(s, sslbls_to_keep, new_sslbls, idle)
            if len(fs) > 0 or len(s) == 0: ret.append(fs)
        return ret
    else:  # drop == False (the easy case)
        return [filter_circuit(s, sslbls_to_keep, new_sslbls, idle) for s in circuits]


def filter_circuit(circuit, sslbls_to_keep, new_sslbls=None, idle=()):
    """
    Removes any labels from `circuit` whose state-space labels are not
    entirely in `sslbls_to_keep`.  If a gates label's state-space labels
    (its `.sslbls`) is `None`, then the label is retained in the returned
    string.

    Furthermore, by specifying `new_sslbls` one can map the state-space
    labels in `sslbls_to_keep` to new labels (useful for "re-basing" a
    set of qubit strings.

    Parameters
    ----------
    circuit : Circuit
        The operation sequence to act on.

    sslbls_to_keep : list
        A list of state space labels specifying which operation labels should
        be retained.

    new_sslbls : list, optional
        If not None, a list of the same length as `sslbls_to_keep` specifying
        a new set of state space labels to replace those in `sslbls_to_keep`.

    idle : string or Label, optional
        The operation label to be used when there are no kept components of a
        "layer" (element) of `circuit`.

    Returns
    -------
    Circuit
    """
    if new_sslbls is not None:
        sslbl_map = {old: new for old, new in zip(sslbls_to_keep, new_sslbls)}
    else: sslbl_map = None

    lbls = []
    for lbl in circuit:
        sublbls = []; pintersect = False  # btwn lbl's sslbls & to-keep
        for sublbl in lbl.components:
            if sublbl.sslbls is None \
               or set(sublbl.sslbls).issubset(sslbls_to_keep):  # then keep this comp

                if sslbl_map:  # update state space labels
                    new_sslbls = None if (sublbl.sslbls is None) else \
                        tuple((sslbl_map[x] for x in sublbl.sslbls))
                    sublbls.append(_Lbl(sublbl.name, new_sslbls))
                else:  # leave labels as-is
                    sublbls.append(sublbl)

            elif len(set(sublbl.sslbls).intersection(sslbls_to_keep)) > 0:
                pintersect = True  # partial intersection w/to-keep!

        if pintersect:
            # there was partial intersection with at least one component,
            # so there's no way to cleanly cast this gate sequence as
            # just a sequence of labels in "sslbls_to_keep"
            return None

        if len(sublbls) > 0:
            if len(sublbls) == 1:  # necessary?
                lbls.append(sublbls[0])
            else:
                lbls.append(sublbls)
        else:  # if len(lbl.components) > 0:
            # no mention of any of sslbls_to_keep in all components (otherwise
            # either pintersect would be set or len(sublbls) > 0), so this layer
            # is just an idle: add idle placeholder.
            if idle is not None: lbls.append(_Lbl(idle))

    return _cir.Circuit(lbls, line_labels=sslbls_to_keep)

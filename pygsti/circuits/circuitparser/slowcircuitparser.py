""" Native python implementation of a text parser for reading GST input files. """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.baseobjs import label as _lbl


def _to_int_or_strip(x):
    return int(x) if x.strip().isdigit() else x.strip()


def parse_circuit(code, create_subcircuits=True, integerize_sslbls=True):
    if '@' in code:  # format:  <string>@<line_labels>[@<occurrence_id>]
        code, *extras = code.split('@')
        labels = extras[0].strip("( )")  # remove opening and closing parenthesis
        if len(labels) > 0:
            labels = tuple(map(_to_int_or_strip, labels.split(',')))
        else:
            labels = ()  # no labels

        if len(extras) > 1:
            occurrence_id = _to_int_or_strip(extras[1])
        else:
            occurrence_id = None
    else:
        labels = None
        occurrence_id = None

    compilable_joins_exist = bool('~' in code)
    barrier_joins_exist = bool('|' in code)
    if compilable_joins_exist and barrier_joins_exist:
        raise ValueError("Circuit string '%s' contains both barrier and compilable layer joining!" % code)
    elif compilable_joins_exist: interlayer_marker = '~'
    elif barrier_joins_exist: interlayer_marker = '|'
    else: interlayer_marker = u''  # matches nothing

    result = []; interlayer_marker_inds = []
    code = code.replace('*', '')  # multiplication is implicit (no need for '*' ops)
    i = 0; end = len(code); segment = 0

    while i < end:
        if code[i] == interlayer_marker:
            interlayer_marker_inds.append(len(result) - 1); i += 1
            if i == end: break

        lbls_list, i, segment, marker_inds = _get_next_lbls(code, i, end, create_subcircuits, integerize_sslbls,
                                                            segment, interlayer_marker)
        interlayer_marker_inds.extend([len(result) + k for k in marker_inds])
        result.extend(lbls_list)

    # construct list of compile-able circuit layer indices (indices of result)
    if compilable_joins_exist:  # marker is '~', so marker indices == compilable-layer indices
        compilable_indices = tuple(interlayer_marker_inds)
    elif barrier_joins_exist:  # marker is '|', so invert marker indices to get compilable-layer indices
        compilable_indices = tuple(sorted(set(range(len(result))) - set(interlayer_marker_inds)))
    else:
        compilable_indices = None

    return tuple(result), labels, occurrence_id, compilable_indices


def parse_label(code, integerize_sslbls=True):
    create_subcircuits = False
    segment = 0  # segment for gates/instruments vs. preps vs. povms: 0 = *any*
    interlayer_marker = u''  # matches nothing - no interlayer markerg

    lbls_list, _, _, _ = _get_next_lbls(code, 0, len(code), create_subcircuits, integerize_sslbls,
                                        segment, interlayer_marker)
    if len(lbls_list) != 1:
        raise ValueError("Invalid label, could not parse: '%s'" % str(code))
    return lbls_list[0]


def _get_next_lbls(s, start, end, create_subcircuits, integerize_sslbls, segment, interlayer_marker):
    if s[start] == "(":
        i = start + 1
        lbls_list = []; interlayer_marker_inds = []
        while i < end and s[i] != ")":
            if s[i] == interlayer_marker:
                interlayer_marker_inds.append(len(lbls_list) - 1); i += 1
                if i == end or s[i] == ")": break

            lbls, i, segment, _ = _get_next_lbls(s, i, end, create_subcircuits, integerize_sslbls, segment,
                                                 interlayer_marker)  # don't recursively look for interlayer markers
            lbls_list.extend(lbls)

        if i == end: raise ValueError("mismatched parenthesis")
        i += 1
        exponent, i = _parse_exponent(s, i, end)

        if exponent != 1 and len(interlayer_marker_inds) > 0:
            if exponent == 0: interlayer_marker_inds = ()
            else:  # exponent > 1
                base_marker_inds = interlayer_marker_inds[:]  # a new list
                for k in range(1, exponent):
                    offset = len(lbls_list) * k
                    interlayer_marker_inds.extend(map(lambda x: x + offset, base_marker_inds))

        if create_subcircuits:
            if len(lbls_list) == 0:  # special case of {}^power => remain empty
                return [], i, segment, ()
            else:
                tmp = _lbl.Label(lbls_list)  # just for total sslbs - should probably do something faster
                return [_lbl.CircuitLabel('', lbls_list, tmp.sslbls, exponent)], i, segment, ()
        else:
            return lbls_list * exponent, i, segment, interlayer_marker_inds

    elif s[start] == "[":  # layer
        i = start + 1
        lbls_list = []
        while i < end and s[i] != "]":
            lbls, i, segment, _ = _get_next_lbls(s, i, end, create_subcircuits, integerize_sslbls, segment,
                                                 interlayer_marker)  # but don't actually look for marker
            lbls_list.extend(lbls)
        if i == end: raise ValueError("mismatched parenthesis")
        i += 1
        exponent, i = _parse_exponent(s, i, end)

        if len(lbls_list) == 0:
            to_exponentiate = _lbl.LabelTupTup(())
        elif len(lbls_list) > 1:
            time = max([l.time for l in lbls_list])
            # create a layer label - a label of the labels within square brackets
            to_exponentiate = _lbl.LabelTupTup(tuple(lbls_list)) if (time == 0.0) \
                else _lbl.LabelTupTupWithTime(tuple(lbls_list), time)
        else:
            to_exponentiate = lbls_list[0]
        return [to_exponentiate] * exponent, i, segment, ()

    else:
        lbls, i, segment = _get_next_simple_lbl(s, start, end, integerize_sslbls, segment)
        exponent, i = _parse_exponent(s, i, end)
        return lbls * exponent, i, segment, ()


def _get_next_simple_lbl(s, start, end, integerize_sslbls, segment):
    i = start
    c = s[i]
    if segment == 0 and s[i:i + 3] == 'rho':
        i += 3; segment = 1
    elif segment <= 1:
        if (c == 'G' or c == 'I'):
            i += 1; segment = 1
        elif c == 'M':
            i += 1; segment = 2  # marks end - no more labels allowed
        elif c == '{':
            i += 1
            if i < end and s[i] == '}':  # empty-label special case
                i += 1
                return [], i, segment
        else:
            raise ValueError("Invalid prefix at: %s..." % s[i:i + 5])
    else:
        raise ValueError("Invalid prefix at: %s..." % s[i:i + 5])

    if s[start] == '{':
        while i < end and s[i] != '}':
            c = s[i]
            if not ('a' <= c <= 'z' or '0' <= c <= '9' or c == '_' or c == '(' or c == ')'):
                raise ValueError("Invalid character '%s' in gate name: %s..." % (c, s[start - 1:start + 4]))
            i += 1
        if s[i] != '}' or start + 1 >= i - 1:
            raise ValueError("Invalid '{' at: %s..." % s[start - 1:start + 4])
        i += 1
    else:
        while i < end:
            c = s[i]
            if 'a' <= c <= 'z' or '0' <= c <= '9' or c == '_':
                i += 1
            else:
                break
    name = s[start:i]; last = i

    args = []
    while i < end and s[i] == ';':
        i += 1
        last = i
        while i < end:
            c = s[i]
            if 'a' <= c <= 'z' or '0' <= c <= '9' or c == '_' or c == 'Q' or c == '.' or c == '/' or c == '-':
                i += 1
            else:
                break
        try:
            arg = float(s[last:i])
        except:
            arg = s[last:i]
        args.append(arg); last = i

    sslbls = []
    while i < end and s[i] == ':':
        i += 1
        last = i
        while i < end:
            c = s[i]
            if 'a' <= c <= 'z' or '0' <= c <= '9' or c == '_' or c == 'Q':
                i += 1
            else:
                break
        if integerize_sslbls:
            try:
                val = int(s[last:i])
            except:
                val = s[last:i]
            sslbls.append(val); last = i
        else:
            sslbls.append(s[last:i]); last = i

    if i < end and s[i] == '!':
        i += 1
        last = i
        while i < end:
            c = s[i]
            if '0' <= c <= '9' or c == '.':
                i += 1
            else:
                break
        time = float(s[last:i])
    else:
        time = 0.0

    if len(args) == 0:
        if len(sslbls) == 0:
            return [_lbl.LabelStr(name, time)], i, segment
        elif time == 0.0:
            return [_lbl.LabelTup((name,) + tuple(sslbls))], i, segment
        else:
            return [_lbl.LabelTupWithTime((name,) + tuple(sslbls), time)], i, segment
    else:
        return [_lbl.LabelTupWithArgs((name, 2 + len(args)) + tuple(args) + tuple(sslbls), time)], i, segment


def _parse_exponent(s, i, end):
    #z = re.match("\^([0-9]+)", s[i:end])
    exponent = 1
    if i < end and s[i] == '^':
        i += 1
        last = i
        while i < end:
            c = s[i]
            if '0' <= c <= '9':
                i += 1
            else:
                break
        exponent = int(s[last:i])
    return exponent, i

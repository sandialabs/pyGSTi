# encoding: utf-8
# cython: profile=False
# cython: linetrace=False
# filename: fastcircuitparser.pyx

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import sys
import time as pytime
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport log10, sqrt, log
from libc cimport time
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.algorithm cimport sort as stdsort
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference as deref, preincrement as inc
cimport cython


#from cpython.ref cimport PyObject
#cdef extern from "Python.h":
#    Py_UCS4* PyUnicode_4BYTE_DATA(PyObject* o)

from ...baseobjs import label as _lbl


#Use 64-bit integers
ctypedef long long INT
ctypedef unsigned long long UINT


def _to_int_or_strip(x):
    return int(x) if x.strip().isdigit() else x.strip()


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def parse_circuit(unicode code, bool create_subcircuits, bool integerize_sslbls):
    if '@' in code:  # format:  <string>@<line_labels>
        code, *extras = code.split(u'@')
        labels = extras[0].strip(u"( )")  # remove opening and closing parenthesis
        if len(labels) > 0:
            labels = tuple(map(_to_int_or_strip, labels.split(u',')))
        else:
            labels = ()  # no labels

        if len(extras) > 1:
            occurrence_id = _to_int_or_strip(extras[1])
        else:
            occurrence_id = None
    else:
        labels = None
        occurrence_id = None

    compilable_joins_exist = (u'~' in code)
    barrier_joins_exist = (u'|' in code)
    if compilable_joins_exist and barrier_joins_exist:
        raise ValueError("Circuit string '%s' contains both barrier and compilable layer joining!" % code)
    elif compilable_joins_exist: interlayer_marker = u'~'
    elif barrier_joins_exist: interlayer_marker = u'|'
    else: interlayer_marker = u''  # matches nothing

    result = []; interlayer_marker_inds = []
    code = code.replace(u'*',u'')  # multiplication is implicit (no need for '*' ops)
    i = 0; end = len(code); segment = 0
    #print "DB -FASTPARSE: ", code

    #cdef Py_UCS4* codep = PyUnicode_4BYTE_DATA(<PyObject*>code)

    while i < end:
        if code[i] == interlayer_marker:
            interlayer_marker_inds.append(len(result) - 1); i += 1
            if i == end: break

        #print "TOP at:",code[i:]
        lbls_list,i,segment, marker_inds = get_next_lbls(code, i, end, create_subcircuits, integerize_sslbls,
                                                         segment, interlayer_marker)
        interlayer_marker_inds.extend([len(result) + k for k in marker_inds])
        result.extend(lbls_list)
        #print "Labels = ",result

    # construct list of compile-able circuit layer indices (indices of result)
    if compilable_joins_exist:  # marker is '~', so marker indices == compilable-layer indices
        compilable_indices = tuple(interlayer_marker_inds)
    elif barrier_joins_exist:  # marker is '|', so invert marker indices to get compilable-layer indices
        compilable_indices = tuple(sorted(set(range(len(result))) - set(interlayer_marker_inds)))
    else:
        compilable_indices = None

    return tuple(result), labels, occurrence_id, compilable_indices


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def parse_label(unicode code, bool integerize_sslbls=True):
    create_subcircuits = False
    segment = 0  # segment for gates/instruments vs. preps vs. povms: 0 = *any*
    interlayer_marker = u''  # matches nothing - no interlayer markerg

    lbls_list, _, _, _ = get_next_lbls(code, 0, len(code), create_subcircuits, integerize_sslbls,
                                       segment, interlayer_marker)
    if len(lbls_list) != 1:
        raise ValueError("Invalid label, could not parse: '%s'" % str(code))
    return lbls_list[0]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef get_next_lbls(unicode s, INT start, INT end, bool create_subcircuits, bool integerize_sslbls, INT segment,
                   unicode interlayer_marker):

    cdef INT i
    cdef INT k
    cdef INT exponent
    cdef INT offset
    cdef INT last

    if s[start] == u"(":
        i = start+1
        lbls_list = []; interlayer_marker_inds = []
        while i < end and s[i] != u")":
            if s[i] == interlayer_marker:
                interlayer_marker_inds.append(len(lbls_list) - 1); i += 1
                if i == end or s[i] == u")": break
            lbls,i,segment,_ = get_next_lbls(s,i,end, create_subcircuits, integerize_sslbls, segment,
                                           interlayer_marker)  # don't recursively look for interlayer markers
            lbls_list.extend(lbls)
        if i == end: raise ValueError("mismatched parenthesis")
        i += 1
        exponent, i = parse_exponent(s,i,end)

        if exponent != 1 and len(interlayer_marker_inds) > 0:
            if exponent == 0: interlayer_marker_inds = ()
            else:  # exponent > 1
                base_marker_inds = interlayer_marker_inds[:]  # a new list
                for k in range(1,exponent):
                    offset = len(lbls_list) * k
                    interlayer_marker_inds.extend(map(lambda x: x + offset, base_marker_inds))

        if create_subcircuits:
            if len(lbls_list) == 0: # special case of {}^power => remain empty
                return [], i, segment, ()
            else:
                tmp = _lbl.Label(lbls_list)  # just for total sslbs - should probably do something faster
                return [_lbl.CircuitLabel('', lbls_list, tmp.sslbls, exponent)], i, segment, ()
        else:
            return lbls_list * exponent, i, segment, interlayer_marker_inds

    elif s[start] == u"[":  #layer
        i = start+1
        lbls_list = []
        while i < end and s[i] != u"]":
            #lbls,i,segment = get_next_simple_lbl(s,i,end, integerize_sslbls, segment)  #ONLY SIMPLE LABELS in [] (no parens)
            lbls,i,segment,_ = get_next_lbls(s,i,end, create_subcircuits, integerize_sslbls, segment,
                                           interlayer_marker)  # but don't actually look for marker
            lbls_list.extend(lbls)
        if i == end: raise ValueError("mismatched parenthesis")
        i += 1
        exponent, i = parse_exponent(s,i,end)

        if len(lbls_list) == 0:
            to_exponentiate = _lbl.LabelTupTup( () )
        elif len(lbls_list) > 1:
            time = max([l.time for l in lbls_list])
            to_exponentiate = _lbl.LabelTupTup(tuple(lbls_list)) if (time == 0.0) \
                else _lbl.LabelTupTupWithTime(tuple(lbls_list), time)  # create a layer label - a label of the labels within square brackets
        else:
            to_exponentiate = lbls_list[0]
        return [to_exponentiate] * exponent, i, segment, ()

    else:
        lbls,i,segment = get_next_simple_lbl(s,start,end, integerize_sslbls, segment)
        exponent, i = parse_exponent(s,i,end)
        return lbls*exponent, i, segment, ()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef get_next_simple_lbl(unicode s, INT start, INT end, bool integerize_sslbls, INT segment):
    cdef INT  i
    cdef INT last
    cdef Py_UCS4 c
    cdef double time
    cdef bool is_int
    #cdef Py_UCS4* sp = PyUnicode_4BYTE_DATA(<PyObject*>s)

    i = start
    c = s[i]
    if segment == 0 and s[i] == u'r':
        i += 1
        if s[i] == u'h':
            i += 1
            if s[i] == u'o':
                i += 1
                segment = 1
            else:
                raise ValueError("Invalid prefix at: %s..." % s[i-2:i+3])
        else:
            raise ValueError("Invalid prefix at: %s..." % s[i-1:i+4])
    elif segment <= 1:
        if (c == u'G' or c == u'I'):
            i += 1; segment = 1
        elif c == u'M':
            i += 1; segment = 2 #marks end - no more labels allowed
        elif c == u'{':
            i += 1
            if i < end and s[i] == u'}':
                i += 1
                return [], i, segment
            else:
                raise ValueError("Invalid '{' at: %s..." % s[i-1:i+4])
        else:
            raise ValueError("Invalid prefix at: %s..." % s[i:i+5])
    else:
        raise ValueError("Invalid prefix at: %s..." % s[i:i+5])

    #z = re.match("([a-z0-9_]+)((?:;[a-zQ0-9_\./]+)*)((?::[a-zQ0-9_]+)*)(![0-9\.]+)?", s[i:end])
    tup = []
    while i < end:
        c = s[i]
        if u'a' <= c <= u'z' or u'0' <= c <= u'9' or c == u'_':
            i += 1
        else:
            break
    name = sys.intern(s[start:i]); last = i

    args = []
    while i < end and s[i] == u';':
        i += 1
        last = i; is_float = True
        while i < end:
            c = s[i]
            if u'a' <= c <= u'z' or c == u'_' or c == u'Q' or c == u'/':
                i += 1; is_float = False
            elif u'0' <= c <= u'9' or c == u'.' or c == u'-':
                i += 1
            else:
                break
        args.append(float(s[last:i]) if is_float else s[last:i]); last = i

    sslbls = []
    while i < end and s[i] == u':':
        i += 1
        last = i; is_int = True
        while i < end:
            c = s[i]
            if u'0' <= c <= u'9':
                i += 1
            elif u'a' <= c <= u'z' or c == u'_' or c == u'Q':
                i += 1; is_int = False
            else:
                break
        if integerize_sslbls and is_int:
            sslbls.append(int(s[last:i])); last = i
        else:
            sslbls.append(sys.intern(s[last:i])); last = i

    if i < end and s[i] == u'!':
        i += 1
        last = i
        while i < end:
            c = s[i]
            if u'0' <= c <= u'9' or c == u'.':
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
    #return _Label(name,sslbls,time,args), i

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef parse_exponent(unicode s, INT i, INT end):
    #z = re.match("\^([0-9]+)", s[i:end])
    cdef Py_UCS4 c
    cdef INT last
    cdef INT exponent = 1
    #cdef Py_UCS4* sp = PyUnicode_4BYTE_DATA(<PyObject*>s)

    if i < end and s[i] == u'^':
        i += 1
        last = i
        #exponent = 0
        while i < end:
            c = s[i]
            if u'0' <= c <= u'9':
                i += 1
                #exponent = exponent * 10 + (<INT>c-<INT>u'0')
            else:
                break
        exponent = int(s[last:i])
    return exponent, i

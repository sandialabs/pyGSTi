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

from ...objects import label as _lbl


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

    result = []
    code = code.replace(u'*',u'')  # multiplication is implicit (no need for '*' ops)
    i = 0; end = len(code); segment = 0
    #print "DB -FASTPARSE: ", code

    #cdef Py_UCS4* codep = PyUnicode_4BYTE_DATA(<PyObject*>code)

    while(True):
        if i == end: break
        #print "TOP at:",code[i:]
        lbls_list,i,segment = get_next_lbls(code, i, end, create_subcircuits, integerize_sslbls, segment)
        result.extend(lbls_list)
        #print "Labels = ",result

    return result, labels, occurrence_id

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef get_next_lbls(unicode s, INT start, INT end, bool create_subcircuits, bool integerize_sslbls, INT segment):

    cdef INT i
    cdef INT last

    if s[start] == u"(":
        i = start+1
        lbls_list = []
        while i < end and s[i] != u")":
            lbls,i,segment = get_next_lbls(s,i,end, create_subcircuits, integerize_sslbls, segment)
            lbls_list.extend(lbls)
        if i == end: raise ValueError("mismatched parenthesis")
        i += 1
        exponent, i = parse_exponent(s,i,end)

        if create_subcircuits:
            if len(lbls_list) == 0: # special case of {}^power => remain empty
                return [], i, segment
            else:
                tmp = _lbl.Label(lbls_list)  # just for total sslbs - should probably do something faster
                return [_lbl.CircuitLabel('', lbls_list, tmp.sslbls, exponent)], i, segment
        else:
            return lbls_list * exponent, i, segment

    elif s[start] == u"[":  #layer
        i = start+1
        lbls_list = []
        while i < end and s[i] != u"]":
            #lbls,i,segment = get_next_simple_lbl(s,i,end, integerize_sslbls, segment)  #ONLY SIMPLE LABELS in [] (no parens)
            lbls,i,segment = get_next_lbls(s,i,end, create_subcircuits, integerize_sslbls, segment)
            lbls_list.extend(lbls)
        if i == end: raise ValueError("mismatched parenthesis")
        i += 1
        exponent, i = parse_exponent(s,i,end)

        if len(lbls_list) == 0:
            to_exponentiate = _lbl.LabelTupTup( () )
        elif len(lbls_list) > 1:
            time = max([l.time for l in lbls_list])
            to_exponentiate = _lbl.LabelTupTup(tuple(lbls_list), time) #create a layer label - a label of the labels within square brackets
        else:
            to_exponentiate = lbls_list[0]
        return [to_exponentiate] * exponent, i, segment

    else:
        lbls,i,segment = get_next_simple_lbl(s,start,end, integerize_sslbls, segment)
        exponent, i = parse_exponent(s,i,end)
        return lbls*exponent, i, segment

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
    name = s[start:i]; last = i

    args = []
    while i < end and s[i] == u';':
        i += 1
        last = i
        while i < end:
            c = s[i]
            if u'a' <= c <= u'z' or u'0' <= c <= u'9' or c == u'_' or c == u'Q' or c == u'.' or c == u'/':
                i += 1
            else:
                break
        args.append(s[last:i]); last = i


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
            sslbls.append(s[last:i]); last = i

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
        else:
            return [_lbl.LabelTup((name,) + tuple(sslbls), time)], i, segment
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

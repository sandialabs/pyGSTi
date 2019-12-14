""" A text parser for reading GST input files. """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Encapsulates a text parser for reading GST input files.

** Grammar **

expop   :: '^'
multop  :: '*'
integer :: '0'..'9'+
real    :: ['+'|'-'] integer [ '.' integer [ 'e' ['+'|'-'] integer ] ]
reflbl  :: (alpha | digit | '_')+

nop       :: '{}'
opname    :: 'G' [ lowercase | digit | '_' ]+
instrmtnm :: 'I' [ lowercase | digit | '_' ]+
povmnm    :: 'M' [ lowercase | digit | '_' ]+
prepnm    :: 'rho' [ lowercase | digit | '_' ]+
gate      :: opname [':' integer ]*
instrmt   :: instrmt [':' integer ]*
povm      :: povmnm [':' integer ]*
prep      :: prepnm [':' integer ]*
strref    :: 'S' '<' reflbl '>'
slcref    :: strref [ '[' integer ':' integer ']' ]
subcircuit:: '[' layer ']' | '(' string ')'
layerable :: gate | instrmt | subcircuit [ expop integer ]*
layer     :: layerable [ [ multop ] layerable ]*
expable   :: gate | instrmt | '[' layer ']' | slcref | '(' string ')' | nop
expdstr   :: expable [ expop integer ]*
string    :: expdstr [ [ multop ] expdstr ]*
pstring   :: [ prep ] string
ppstring  :: pstring [ povm ]
"""


import warnings as _warnings
from ply import lex, yacc
from ..objects import label as _lbl

try:
    from .fastcircuitparser import fast_parse_circuit as _fast_parse_circuit
except ImportError:
    _fast_parse_circuit = None


class CircuitLexer:
    """ Lexer for matching and interpreting text-format operation sequences """

    # List of token names.   This is always required
    tokens = (
        'EXPOP',
        'MULTOP',
        'INTEGER',
        'NOP',
        'GATE',
        'INSTRMT',
        'PREP',
        'POVM',
        'REFLBL',
        'OPENBR',
        'CLOSEBR',
        'LPAREN',
        'COLON',
        'SEMICOLON',
        'EXCLAM',
        'RPAREN',
        'STRINGIND'
    )

    @staticmethod
    def makeLabel(s):
        if '!' in s:
            s, time = s.split('!')  # must be only two parts (only 1 exclamation pt)
            time = float(time)
        else:
            time = 0.0

        if ';' in s:
            parts = s.split(';')
            parts2 = parts[-1].split(':')
            nm = parts[0]
            args = parts[1:-1] + [parts2[0]]
            sslbls = parts2[1:]
        else:
            parts = s.split(':')
            nm = parts[0]
            args = None
            sslbls = parts[1:]

        if len(sslbls) == 0:
            sslbls = None

        return _lbl.Label(nm, sslbls, time, args)

    @staticmethod
    def t_GATE(t):
        r'G[a-z0-9_]+(;[a-zQ0-9_\./]+)*(:[a-zQ0-9_]+)*(![0-9\.]+)?'
        #Note: Q is only capital letter allowed in qubit label
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.makeLabel(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_INSTRMT(t):
        r'I[a-z0-9_]+(![0-9\.]+)?'
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.makeLabel(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_PREP(t):
        r'rho[a-z0-9_]+(![0-9\.]+)?'
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.makeLabel(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_POVM(t):
        r'M[a-z0-9_]+(![0-9\.]+)?'
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.makeLabel(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_STRINGIND(t):
        r'S(?=\s*\<)'
        return t

    @staticmethod
    def t_REFLBL(t):
        r'<\s*[a-zA-Z0-9_]+\s*>'
        t.value = t.value[1:-1].strip()
        return t

    # Regular expression rules for simple tokens
    t_EXPOP = r'\^'
    t_MULTOP = r'\*'
    t_OPENBR = r'\['
    t_CLOSEBR = r'\]'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_COLON = r'\:'
    t_SEMICOLON = r'\;'
    t_EXCLAM = r'\!'

    @staticmethod
    def t_NOP(t):
        r'\{\}'
        t.value = tuple()
        return t

    @staticmethod
    def t_INTEGER(t):
        r'\d+'
        t.value = int(t.value)
        return t

    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

    # Error handling rule
    @staticmethod
    def t_error(t):
        if t is not None:
            raise ValueError("Illegal character '{}' at position {} of string '{}'".format(
                t.value[0], t.lexpos, t.lexer.lexdata))
        raise ValueError("Lexer error")  # pragma: no cover


class CircuitParser(object):
    """ Parser for text-format operation sequences """
    tokens = CircuitLexer.tokens
    mode = "simple"

    def __init__(self, lexer_object=None, lookup={}):
        if self.mode == "ply":
            self._lookup = lookup
            self._lexer = lex.lex(object=lexer_object if lexer_object else CircuitLexer())
            self._parser = yacc.yacc(module=self, start="ppstring", debug=False,
                                     tabmodule='pygsti.io.parsetab_string')
            self.parse = self.ply_parse
        else:
            self._parser = SimpleCircuitParser()
            self.parse = self._parser.parse

    @property
    def lookup(self):
        """ The lookup dictionary for expanding references """
        return self._lookup

    @lookup.setter
    def lookup(self, newdict):
        """ The lookup dictionary for expanding references """
        self._lookup = newdict

    def p_strref(self, p):
        '''strref  : STRINGIND REFLBL'''
        match = self._lookup.get(str(p[2]))
        if match is None:
            raise ValueError("Lookup, key '{}' not found".format(str(p[2])))
        p[0] = tuple(match)

    @staticmethod
    def p_slcref_slice(p):
        '''slcref : strref OPENBR INTEGER COLON INTEGER CLOSEBR'''
        _, ref, _, lower, _, upper, _ = p
        p[0] = ref[lower:upper]

    @staticmethod
    def p_slcref(p):
        '''slcref : strref'''
        p[0] = p[1]

    @staticmethod
    def p_subcircuit_singlelayer(p):
        '''subcircuit : OPENBR layer CLOSEBR'''
        p[0] = p[2],  # subcircuit should be a tuple of layers - and p[2] is a *single* layer

    @staticmethod
    def p_subcircuit_string(p):
        '''subcircuit : LPAREN layer RPAREN'''
        p[0] = p[2]

    @staticmethod
    def p_layerable(p):
        '''layerable : GATE
                     | INSTRMT '''
        p[0] = p[1]

    @staticmethod
    def p_layerable_subcircuit(p):
        '''layerable : subcircuit '''
        plbl = _lbl.Label((p[1],))  # just for total sslbls
        p[0] = _lbl.CircuitLabel('', p[1], plbl.sslbls, 1),

    @staticmethod
    def p_layerable_subcircuit_expop(p):
        '''layerable : subcircuit EXPOP INTEGER'''
        plbl = _lbl.Label(p[1])  # just for total sslbls
        p[0] = _lbl.CircuitLabel('', p[1], plbl.sslbls, p[3]),

    @staticmethod
    def p_layer_layerable(p):
        '''layer : layerable'''
        p[0] = p[1]

    @staticmethod
    def p_layer_str(p):
        '''layer : layer layerable'''
        p[0] = p[1] + p[2]  # tuple concatenation

    @staticmethod
    def p_layer(p):
        '''layer : layer MULTOP layerable'''
        p[0] = p[1] + p[3]  # tuple concatenation

    @staticmethod
    def p_expable_paren(p):
        '''expable : LPAREN string RPAREN'''
        p[0] = p[2]

    @staticmethod
    def p_expable_layer(p):
        '''expable : OPENBR layer CLOSEBR'''
        p[0] = p[2],  # -> tuple

    @staticmethod
    def p_expable_empty_layer(p):
        '''expable : OPENBR CLOSEBR'''
        p[0] = ((),)  # -> empty layer tuple

    @staticmethod
    def p_expable_single(p):
        '''expable : GATE
                   | INSTRMT
                   | NOP '''
        p[0] = p[1]

    @staticmethod
    def p_expable(p):
        '''expable : slcref'''
        p[0] = p[1]

    @staticmethod
    def p_expdstr_expop(p):
        '''expdstr : expable EXPOP INTEGER'''
        plbl = _lbl.Label(p[1])  # just for total sslbls
        if len(p[1]) > 0:
            p[0] = _lbl.CircuitLabel('', p[1], plbl.sslbls, p[3]),
        else:
            p[0] = ()  # special case of {}^power => remain empty
        #OLD (before subcircuits) p[0] = p[1] * p[3]  # tuple repetition

    @staticmethod
    def p_expdstr(p):
        '''expdstr : expable'''
        p[0] = p[1]

    @staticmethod
    def p_string_expdstr(p):
        '''string : expdstr'''
        p[0] = p[1]

    @staticmethod
    def p_string_str(p):
        '''string : string expdstr'''
        p[0] = p[1] + p[2]  # tuple concatenation

    @staticmethod
    def p_string(p):
        '''string : string MULTOP expdstr'''
        p[0] = p[1] + p[3]  # tuple concatenation

    @staticmethod
    def p_pstring(p):
        '''pstring : string'''
        p[0] = p[1]

    @staticmethod
    def p_pstring_prep(p):
        '''pstring : PREP string'''
        p[0] = p[1] + p[2]

    @staticmethod
    def p_ppstring(p):
        '''ppstring : pstring'''
        p[0] = p[1]

    @staticmethod
    def p_ppstring_povm(p):
        '''ppstring : pstring POVM'''
        p[0] = p[1] + p[2]

    @staticmethod
    def p_error(p):
        message = "Syntax error"
        if p is not None:
            message += " at pos {} of input {}".format(p.lexpos, p.lexer.lexdata)
        raise ValueError(message)

    def ply_parse(self, code, create_subcircuits=True):
        """
        Perform lexing and parsing of `code`.

        Parameters
        ----------
        code : str
            A circuit encoded as a single-line string

        Returns
        -------
        layer_labels : tuple
            A tuple of the layer-labels of the circuit
        line_labels : tuple
            A tuple of the line labels of the circuit.
        """
        if '@' in code:  # format:  <string>@<line_labels>
            code, labels = code.split('@')
            labels = labels.strip("( )")  # remove opening and closing parenthesis
            def process(x): return int(x) if x.strip().isdigit() else x.strip()
            labels = tuple(map(process, labels.split(',')))
        else:
            labels = None

        self._lexer.input(code)
        result = self._parser.parse(lexer=self._lexer)
        return result, labels


class SimpleCircuitParser(object):
    """ Parser for text-format operation sequences """

    def __init__(self):
        if _fast_parse_circuit is not None:
            self.parse = self._fast_parse
        else:
            self.parse = self._slow_parse
        self.warned = False

    def _fast_parse(self, code, create_subcircuits=True, integerize_sslbls=True):
        return _fast_parse_circuit(code, create_subcircuits, integerize_sslbls)

    def _slow_parse(self, code, create_subcircuits=True, integerize_sslbls=True):

        if not self.warned:
            _warnings.warn(("Because you don't have Cython installed, pyGSTi is using"
                            " a pure-Python parser.  This is fine, but it means loading"
                            " text file will be somewhat slower."))
            self.warned = True

        if '@' in code:  # format:  <string>@<line_labels>
            code, labels = code.split('@')
            labels = labels.strip("( )")  # remove opening and closing parenthesis
            def process(x): return int(x) if x.strip().isdigit() else x.strip()
            labels = tuple(map(process, labels.split(',')))
        else:
            labels = None

        result = []
        code = code.replace('*', '')
        i = 0; end = len(code); segment = 0

        while(True):
            if i == end: break
            lbls_list, i, segment = self.get_next_lbls(code, i, end, create_subcircuits, integerize_sslbls, segment)
            result.extend(lbls_list)

        return result, labels

    def get_next_lbls(self, s, start, end, create_subcircuits, integerize_sslbls, segment):

        if s[start] == "(":
            i = start + 1
            lbls_list = []
            while i < end and s[i] != ")":
                lbls, i, segment = self.get_next_lbls(s, i, end, create_subcircuits, integerize_sslbls, segment)
                lbls_list.extend(lbls)
            if i == end: raise ValueError("mismatched parenthesis")
            i += 1
            exponent, i = self.parse_exponent(s, i, end)

            if create_subcircuits:
                if len(lbls_list) == 0:  # special case of {}^power => remain empty
                    return [], i, segment
                else:
                    tmp = _lbl.Label(lbls_list)  # just for total sslbs - should probably do something faster
                    return [_lbl.CircuitLabel('', lbls_list, tmp.sslbls, exponent)], i, segment
            else:
                return lbls_list * exponent, i, segment

        elif s[start] == "[":  # layer
            i = start + 1
            lbls_list = []
            while i < end and s[i] != "]":
                lbls, i, segment = self.get_next_lbls(s, i, end, create_subcircuits, integerize_sslbls, segment)
                lbls_list.extend(lbls)
            if i == end: raise ValueError("mismatched parenthesis")
            i += 1
            exponent, i = self.parse_exponent(s, i, end)

            if len(lbls_list) == 0:
                to_exponentiate = _lbl.LabelTupTup(())
            elif len(lbls_list) > 1:
                time = max([l.time for l in lbls_list])
                # create a layer label - a label of the labels within square brackets
                to_exponentiate = _lbl.LabelTupTup(tuple(lbls_list), time)
            else:
                to_exponentiate = lbls_list[0]
            return [to_exponentiate] * exponent, i, segment

        else:
            lbls, i, segment = self.get_next_simple_lbl(s, start, end, integerize_sslbls, segment)
            exponent, i = self.parse_exponent(s, i, end)
            return lbls * exponent, i, segment

    def get_next_simple_lbl(self, s, start, end, integerize_sslbls, segment):
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
                if i < end and s[i] == '}':
                    i += 1
                    return [], i, segment
                else:
                    raise ValueError("Invalid '{' at: %s..." % s[i - 1:i + 4])
            else:
                raise ValueError("Invalid prefix at: %s..." % s[i:i + 5])
        else:
            raise ValueError("Invalid prefix at: %s..." % s[i:i + 5])

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
                if 'a' <= c <= 'z' or '0' <= c <= '9' or c == '_' or c == 'Q' or c == '.' or c == '/':
                    i += 1
                else:
                    break
            args.append(s[last:i]); last = i

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
            else:
                return [_lbl.LabelTup((name,) + tuple(sslbls), time)], i, segment
        else:
            return [_lbl.LabelTupWithArgs((name, 2 + len(args)) + tuple(args) + tuple(sslbls), time)], i, segment

    def parse_exponent(self, s, i, end):
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

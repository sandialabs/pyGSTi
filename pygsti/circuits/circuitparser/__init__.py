"""Implementations of a text parser for reading GST input files."""
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

**Grammar**

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

warn_msg = """
An optimized Cython-based implementation of `{module}` is available as
an extension, but couldn't be imported. This might happen if the
extension has not been built. `pip install cython`, then reinstall
pyGSTi to build Cython extensions. Alternatively, setting the
environment variable `PYGSTI_NO_CYTHON_WARNING` will suppress this
message.
""".format(module=__name__)

try:
    # Import cython implementation if it's been built...
    from .fastcircuitparser import parse_circuit, parse_label
except ImportError:
    # ... If not, fall back to the python implementation, with a warning.
    import os as _os
    import warnings as _warnings

    if 'PYGSTI_NO_CYTHON_WARNING' not in _os.environ:
        _warnings.warn(warn_msg)

    from .slowcircuitparser import parse_circuit, parse_label


from pygsti.baseobjs import label as _lbl


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
    def make_label(s):
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
    def t_GATE(t):                                                                       # noqa
        r"""
        ``'G[a-z0-9_]+(;[a-zQ0-9_\./]+)*(:[a-zQ0-9_]+)*(![0-9\.]+)?'``
        """
        
        #Note: Q is only capital letter allowed in qubit label
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.make_label(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_INSTRMT(t):                                                               # noqa
        r"""
        ``'I[a-z0-9_]+(![0-9\.]+)?'``
        """
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.make_label(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_PREP(t):                                                                       # noqa
        r"""
        ``'rho[a-z0-9_]+(![0-9\.]+)?'``
        """
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.make_label(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_POVM(t):                                                                       # noqa
        r"""
        ``'M[a-z0-9_]+(![0-9\.]+)?'``
        """
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        lbl = CircuitLexer.make_label(t.value)
        t.value = lbl,  # make it a tuple
        return t

    @staticmethod
    def t_STRINGIND(t):                                                                  # noqa
        r"""
        ``'S(?=\s*\<)'``
        """
        return t

    @staticmethod
    def t_REFLBL(t):                                                                     # noqa
        r"""
        ``'<\s*[a-zA-Z0-9_]+\s*>'``
        """
        t.value = t.value[1:-1].strip()
        return t

    # Regular expression rules for simple tokens
    t_EXPOP = r'\^'                                                                      # noqa
    t_MULTOP = r'\*'                                                                     # noqa
    t_OPENBR = r'\['                                                                     # noqa
    t_CLOSEBR = r'\]'                                                                    # noqa
    t_LPAREN = r'\('                                                                     # noqa
    t_RPAREN = r'\)'                                                                     # noqa
    t_COLON = r'\:'                                                                      # noqa
    t_SEMICOLON = r'\;'                                                                  # noqa
    t_EXCLAM = r'\!'                                                                     # noqa

    @staticmethod
    def t_NOP(t):                                                                        # noqa
        r"""
        ``'\{\}'``
        """
        t.value = tuple()
        return t

    @staticmethod
    def t_INTEGER(t):                                                                    # noqa
        r"""
        ``'\d+'``
        """
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

    def __init__(self, lexer_object=None, lookup=None):
        if lookup is None:
            lookup = {}
        if self.mode == "ply":
            from ply import lex, yacc  # these aren't needed for "simple" mode
            self._lookup = lookup
            self._lexer = lex.lex(object=lexer_object if lexer_object else CircuitLexer())
            self._parser = yacc.yacc(module=self, start="ppstring", debug=False,
                                     tabmodule='pygsti.io.parsetab_string')
            self.parse = self.ply_parse
        else:
            self.parse = self._parse

    def _parse(self, code, create_subcircuits=True, integerize_sslbls=True):
        """ Parse a circuit string

        This method will dispatch to the optimized Cython
        implementation, if available. Otherwise, the slower native
        python implementation will be used.
        """
        return parse_circuit(code, create_subcircuits, integerize_sslbls)

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

""" A text parser for reading GST input files. """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Encapsulates a text parser for reading GST input files.

** Grammar **

expop   :: '^'
multop  :: '*'
integer :: '0'..'9'+
real    :: ['+'|'-'] integer [ '.' integer [ 'e' ['+'|'-'] integer ] ]
reflbl  :: (alpha | digit | '_')+

nop     :: '{}'
gate    :: 'G' [ lowercase | digit | '_' ]+
strref  :: 'S' '[' reflbl ']'
slcref  :: strref [ '[' integer ':' integer ']' ]
expable :: gate | slcref | '(' string ')' | nop
expdstr :: expable [ expop integer ]*
string  :: expdstr [ [ multop ] expdstr ]*
"""

from ply import lex, yacc


class GateStringLexer:
    """ Lexer for matching and interpreting text-format gate sequences """
    
    # List of token names.   This is always required
    tokens = (
        'EXPOP',
        'MULTOP',
        'INTEGER',
        'NOP',
        'GATE',
        'REFLBL',
        'OPENBR',
        'CLOSEBR',
        'LPAREN',
        'COLON',
        'RPAREN',
        'STRINGIND'
    )

    @staticmethod
    def t_GATE(t):
        r'G[a-z0-9_]+'
        t.value = t.value,  # make it a tuple
        return t

    @staticmethod
    def t_STRINGIND(t):
        r'S(?=\s*\[)'
        return t

    @staticmethod
    def t_REFLBL(t):
        r'\[\s*[a-zA-Z0-9_]+\s*\]'
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
    t_ignore  = ' \t'

    # Error handling rule
    @staticmethod
    def t_error(t):
        if t is not None:
            raise ValueError("Illegal character '{}' at position {} of string '{}'".format(t.value[0], t.lexpos, t.lexer.lexdata))
        raise ValueError("Lexer error")


class GateStringParser(object):
    """ Parser for text-format gate sequences """
    tokens = GateStringLexer.tokens

    def __init__(self, lexer_object=None, lookup={}):
        self._lookup = lookup
        self._lexer = lex.lex(object=lexer_object if lexer_object else GateStringLexer())
        self._parser = yacc.yacc(module=self, start="string", debug=False, tabmodule='pygsti.baseobjs.parsetab_string')

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
    def p_expable_paren(p):
        '''expable : LPAREN string RPAREN'''
        p[0] = p[2]

    @staticmethod
    def p_expable_single(p):
        '''expable : GATE
                   | NOP '''
        p[0] = p[1]

    @staticmethod
    def p_expable(p):
        '''expable : slcref'''
        p[0] = p[1]

    @staticmethod
    def p_expdstr_expop(p):
        '''expdstr : expable EXPOP INTEGER'''
        p[0] = p[1] * p[3]  # tuple repetition

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
        p[0] = p[1] + p[2]  # tuple conatenation

    @staticmethod
    def p_string(p):
        '''string : string MULTOP expdstr'''
        p[0] = p[1] + p[3]  # tuple concatenation

    @staticmethod
    def p_error(p):
        message = "Syntax error"
        if p is not None:
            message += " at pos {} of input {}".format(p.lexpos, p.lexer.lexdata)
        raise ValueError(message)

    def parse(self, code):
        """ Perform lexing and parsing of `code` """
        self._lexer.input(code)
        result = self._parser.parse(lexer=self._lexer)
        return result


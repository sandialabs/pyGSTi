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

nop       :: '{}'
opname    :: 'G' [ lowercase | digit | '_' ]+ 
instrmtnm :: 'I' [ lowercase | digit | '_' ]+ 
povmnm    :: 'M' [ lowercase | digit | '_' ]+
prepnm    :: 'rho' [ lowercase | digit | '_' ]+
gate      :: opname [':' integer ]*
instrmt   :: instrmt [':' integer ]*
povm      :: povm [':' integer ]*
prep      :: prep [':' integer ]*
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

from ply import lex, yacc
from .label import Label as _Label
from .label import CircuitLabel as _CircuitLabel

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
        'RPAREN',
        'STRINGIND'
    )

    @staticmethod
    def t_GATE(t):
        r'G[a-z0-9_]+(:[a-zQ0-9_]+)*'
        #Note: Q is only capital letter allowed in qubit label
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        parts = t.value.split(':')
        lbl = _Label(t.value) if (len(parts) == 1) else _Label(parts[0],parts[1:]) 
        t.value = lbl, # make it a tuple
        return t 

    @staticmethod
    def t_INSTRMT(t):
        r'I[a-z0-9_]+'
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        parts = t.value.split(':')
        lbl = _Label(t.value) if (len(parts) == 1) else _Label(parts[0],parts[1:]) 
        t.value = lbl, # make it a tuple
        return t

    @staticmethod
    def t_PREP(t):
        r'rho[a-z0-9_]+'
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        parts = t.value.split(':')
        lbl = _Label(t.value) if (len(parts) == 1) else _Label(parts[0],parts[1:]) 
        t.value = lbl, # make it a tuple
        return t

    @staticmethod
    def t_POVM(t):
        r'M[a-z0-9_]+'
        #Note: don't need to convert parts[1],etc, to integers (if possible) as Label automatically does this
        parts = t.value.split(':')
        lbl = _Label(t.value) if (len(parts) == 1) else _Label(parts[0],parts[1:]) 
        t.value = lbl, # make it a tuple
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
        raise ValueError("Lexer error") # pragma: no cover


class CircuitParser(object):
    """ Parser for text-format operation sequences """
    tokens = CircuitLexer.tokens

    def __init__(self, lexer_object=None, lookup={}):
        self._lookup = lookup
        self._lexer = lex.lex(object=lexer_object if lexer_object else CircuitLexer())
        self._parser = yacc.yacc(module=self, start="ppstring", debug=False,
                                 tabmodule='pygsti.baseobjs.parsetab_string')
        
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
        p[0] = p[2], # subcircuit should be a tuple of layers - and p[2] is a *single* layer

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
        plbl = _Label( (p[1],) ) # just for total sslbls
        p[0] = _CircuitLabel('',p[1],plbl.sslbls,1),

    @staticmethod
    def p_layerable_subcircuit_expop(p):
        '''layerable : subcircuit EXPOP INTEGER'''
        plbl = _Label(p[1]) # just for total sslbls
        p[0] = _CircuitLabel('',p[1],plbl.sslbls,p[3]),

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
        p[0] = p[2], # -> tuple

    @staticmethod
    def p_expable_empty_layer(p):
        '''expable : OPENBR CLOSEBR'''
        p[0] = ((),) # -> empty layer tuple
        
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
        plbl = _Label(p[1]) # just for total sslbls
        if len(p[1]) > 0:
            p[0] = _CircuitLabel('',p[1],plbl.sslbls,p[3]),
        else:
            p[0] = () # special case of {}^power => remain empty
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

    def parse(self, code):
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
        if '@' in code: # format:  <string>@<line_labels>
            code,labels = code.split('@')
            labels = labels.strip("( )") #remove opening and closing parenthesis
            process = lambda x: int(x) if x.strip().isdigit() else x.strip()
            labels = tuple(map(process,labels.split(',')))
        else:
            labels = None
            
        self._lexer.input(code)
        result = self._parser.parse(lexer=self._lexer)
        return result,labels


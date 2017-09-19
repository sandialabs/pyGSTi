from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Variables for working with the 2-qubit gate set containing the gates
I*X(pi/2), I*Y(pi/2), X(pi/2)*I, Y(pi/2)*I, and CPHASE.
"""

from . import gatestringconstruction as _strc
from . import gatesetconstruction as _setc
from . import spamspecconstruction as _spamc
from collections import OrderedDict as _OrderedDict

description = "I*I, I*X(pi/2), I*Y(pi/2), X(pi/2)*I, and Y(pi/2)*I gates"

gates = ['Gii', 'Gix','Giy','Gxi','Gyi']

fiducials16 = _strc.gatestring_list(
    [ (), ('Gix',), ('Giy',), ('Gix','Gix'),
      ('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'),
      ('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'),
      ('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix') ] )

fiducials36 = _strc.gatestring_list(
    [ (), ('Gix',), ('Giy',), ('Gix','Gix'), ('Gix','Gix','Gix'), ('Giy','Giy','Giy'),
      ('Gxi',), ('Gxi','Gix'), ('Gxi','Giy'), ('Gxi','Gix','Gix'), ('Gxi','Gix','Gix','Gix'), ('Gxi','Giy','Giy','Giy'),
      ('Gyi',), ('Gyi','Gix'), ('Gyi','Giy'), ('Gyi','Gix','Gix'), ('Gyi','Gix','Gix','Gix'), ('Gyi','Giy','Giy','Giy'),
      ('Gxi','Gxi'), ('Gxi','Gxi','Gix'), ('Gxi','Gxi','Giy'), ('Gxi','Gxi','Gix','Gix'), ('Gxi','Gxi','Gix','Gix','Gix'),
      ('Gxi','Gxi','Giy','Giy','Giy'), ('Gxi','Gxi','Gxi'), ('Gxi','Gxi','Gxi','Gix'), ('Gxi','Gxi','Gxi','Giy'),
      ('Gxi','Gxi','Gxi','Gix','Gix'), ('Gxi','Gxi','Gxi','Gix','Gix','Gix'), ('Gxi','Gxi','Gxi','Giy','Giy','Giy'),
      ('Gyi','Gyi','Gyi'), ('Gyi','Gyi','Gyi','Gix'), ('Gyi','Gyi','Gyi','Giy'), ('Gyi','Gyi','Gyi','Gix','Gix'),
      ('Gyi','Gyi','Gyi','Gix','Gix','Gix'), ('Gyi','Gyi','Gyi','Giy','Giy','Giy') ] )

fiducials = fiducials16
prepStrs = fiducials16

effectStrs = _strc.gatestring_list(
    [(), ('Gix',), ('Giy',), 
     ('Gix','Gix'), ('Gxi',), 
     ('Gyi',), ('Gxi','Gxi'), 
     ('Gxi','Gix'), ('Gxi','Giy'), 
     ('Gyi','Gix'), ('Gyi','Giy')] )

germs = _strc.gatestring_list(
    [('Gii',),
     ('Gxi',),
     ('Gyi',),
     ('Gix',),
     ('Giy',),
     ('Gxi', 'Gyi'),
     ('Gix', 'Giy'),
     ('Giy', 'Gxi'),
     ('Gix', 'Gxi'),
     ('Gii', 'Gix'),
     ('Gxi', 'Gyi', 'Gii'),
     ('Gxi', 'Gii', 'Gyi'),
     ('Gxi', 'Gii', 'Gii'),
     ('Gyi', 'Gii', 'Gii'),
     ('Gix', 'Giy', 'Gii'),
     ('Gix', 'Gii', 'Giy'),
     ('Gix', 'Gii', 'Gii'),
     ('Giy', 'Gii', 'Gii'),
     ('Gii', 'Gyi', 'Giy'),
     ('Gix', 'Gyi', 'Giy'),
     ('Gxi', 'Gxi', 'Gii', 'Gyi'),
     ('Gxi', 'Gyi', 'Gyi', 'Gii'),
     ('Gix', 'Gix', 'Gii', 'Giy'),
     ('Gix', 'Giy', 'Giy', 'Gii'),
     ('Gix', 'Gyi', 'Gix', 'Gyi'),
     ('Gyi', 'Gyi', 'Giy', 'Gyi'),
     ('Gix', 'Gix', 'Gxi', 'Gix'),
     ('Giy', 'Gix', 'Gix', 'Gix'),
     ('Gix', 'Gyi', 'Gyi', 'Gyi'),
     ('Gix', 'Gii', 'Gii', 'Gxi'),
     ('Gix', 'Gii', 'Gyi', 'Gii'),
     ('Giy', 'Giy', 'Gxi', 'Gyi', 'Gxi'),
     ('Giy', 'Giy', 'Giy', 'Gix', 'Gxi'),
     ('Gix', 'Gxi', 'Gyi', 'Gxi', 'Giy'),
     ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'),
     ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'),
     ('Gix', 'Giy', 'Gix', 'Giy', 'Gix', 'Gyi'),
     ('Gyi', 'Gii', 'Giy', 'Gxi', 'Gxi', 'Giy'),
     ('Gxi', 'Gix', 'Giy', 'Gxi', 'Giy', 'Gyi'),
     ('Giy', 'Gii', 'Gii', 'Gxi', 'Giy', 'Gxi'),
     ('Gxi', 'Gix', 'Giy', 'Gix', 'Giy', 'Gix'),
     ('Gix', 'Giy', 'Gix', 'Gyi', 'Gxi', 'Gii', 'Gxi'),
     ('Gxi', 'Gyi', 'Gyi', 'Gix', 'Giy', 'Gxi', 'Giy'),
     ('Giy', 'Gii', 'Gyi', 'Gyi', 'Gix', 'Gxi', 'Giy'),
     ('Giy', 'Gyi', 'Gxi', 'Gyi', 'Gix', 'Gix', 'Giy'),
     ('Gxi', 'Giy', 'Gxi', 'Gyi', 'Gix', 'Gii', 'Gxi'),
     ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gix', 'Gii'),
     ('Giy', 'Gii', 'Gii', 'Giy', 'Giy', 'Gii', 'Giy'),
     ('Gxi', 'Gii', 'Giy', 'Gxi', 'Gyi', 'Giy', 'Gii'),
     ('Giy', 'Gxi', 'Gyi', 'Giy', 'Gyi', 'Gxi', 'Gii'),
     ('Gii', 'Gxi', 'Giy', 'Gyi', 'Gyi', 'Gix', 'Gyi'),
     ('Gix', 'Giy', 'Giy', 'Gyi', 'Gii', 'Gxi', 'Giy'),
     ('Gyi', 'Gix', 'Gxi', 'Gyi', 'Gxi', 'Gii', 'Giy'),
     ('Gix', 'Gyi', 'Gii', 'Gix', 'Gix', 'Gxi', 'Gyi'),
     ('Giy', 'Gxi', 'Gix', 'Giy', 'Gyi', 'Giy', 'Gxi'),
     ('Giy', 'Gii', 'Gxi', 'Gxi', 'Gix', 'Gii', 'Gyi', 'Giy'),
     ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gii', 'Gxi', 'Giy'),
     ('Giy', 'Gix', 'Gii', 'Gyi', 'Gii', 'Gyi', 'Gxi', 'Giy'),
     ('Giy', 'Gyi', 'Gix', 'Gix', 'Gxi', 'Gxi', 'Gxi', 'Gyi'),
     ('Gii', 'Gii', 'Gyi', 'Giy', 'Gix', 'Giy', 'Gix', 'Gxi'),
     ('Gxi', 'Gii', 'Gii', 'Gix', 'Giy', 'Gxi', 'Gyi', 'Gix'),
     ('Gyi', 'Gxi', 'Giy', 'Gxi', 'Gix', 'Gxi', 'Gyi', 'Giy'),
     ('Gyi', 'Gii', 'Gix', 'Gyi', 'Gyi', 'Gxi', 'Gix', 'Giy') ] )
    
#Construct the target gateset
gs_target = _setc.build_gateset(
    [4], [('Q0','Q1')],['Gii','Gix','Giy','Gxi','Gyi'],
    [ "I(Q0):I(Q1)", "I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)" ],
    prepLabels=['rho0'], prepExpressions=["0"],
    effectLabels=['E0','E1','E2'], effectExpressions=["0","1","2"],
    spamdefs={'00': ('rho0','E0'), '01': ('rho0','E1'),
              '10': ('rho0','E2'), '11': ('rho0','remainder') }, basis="pp")


specs16x10 = _spamc.build_spam_specs(
    prepStrs=prepStrs,
    effectStrs=effectStrs,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

specs16 = _spamc.build_spam_specs(
    fiducials16,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

specs36 = _spamc.build_spam_specs(
    fiducials36,
    prep_labels=gs_target.get_prep_labels(),
    effect_labels=gs_target.get_effect_labels() )

specs = specs16x10 #use smallest specs set as "default"

clifford_compilation = _OrderedDict()
clifford_compilation['Gc0c0'] = ['Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c1'] = ['Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c2'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Giy', 'Gii']   
clifford_compilation['Gc0c3'] = ['Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c4'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii']   
clifford_compilation['Gc0c5'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c6'] = ['Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c7'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c8'] = ['Gix', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c9'] = ['Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c10'] = ['Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c11'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c12'] = ['Giy', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c13'] = ['Gix', 'Gix', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c14'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gix']   
clifford_compilation['Gc0c15'] = ['Giy', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c16'] = ['Gix', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c17'] = ['Gix', 'Giy', 'Gix', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c18'] = ['Giy', 'Giy', 'Giy', 'Gix', 'Gix', 'Gii', 'Gii']   
clifford_compilation['Gc0c19'] = ['Gix', 'Giy', 'Giy', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c20'] = ['Gix', 'Giy', 'Giy', 'Giy', 'Gix', 'Gii', 'Gii']   
clifford_compilation['Gc0c21'] = ['Giy', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc0c22'] = ['Gix', 'Gix', 'Gix', 'Giy', 'Giy', 'Gii', 'Gii']   
clifford_compilation['Gc0c23'] = ['Gix', 'Giy', 'Gix', 'Gix', 'Gix', 'Gii', 'Gii']   
clifford_compilation['Gc1c0'] = ['Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc2c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii']   
clifford_compilation['Gc3c0'] = ['Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']     
clifford_compilation['Gc4c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii']   
clifford_compilation['Gc5c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']
clifford_compilation['Gc6c0'] = ['Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc7c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii']    
clifford_compilation['Gc8c0'] = ['Gxi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc9c0'] = ['Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc10c0'] = ['Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc11c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc12c0'] = ['Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']    
clifford_compilation['Gc13c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc14c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gxi']   
clifford_compilation['Gc15c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']    
clifford_compilation['Gc16c0'] = ['Gxi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc17c0'] = ['Gxi', 'Gyi', 'Gxi', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc18c0'] = ['Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gxi', 'Gii', 'Gii']   
clifford_compilation['Gc19c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc20c0'] = ['Gxi', 'Gyi', 'Gyi', 'Gyi', 'Gxi', 'Gii', 'Gii']   
clifford_compilation['Gc21c0'] = ['Gyi', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii', 'Gii']   
clifford_compilation['Gc22c0'] = ['Gxi', 'Gxi', 'Gxi', 'Gyi', 'Gyi', 'Gii', 'Gii']   
clifford_compilation['Gc23c0'] = ['Gxi', 'Gyi', 'Gxi', 'Gxi', 'Gxi', 'Gii', 'Gii']   


global_fidPairs =  [
    (0, 2), (1, 0), (1, 4), (3, 10), (4, 3), (5, 7), (7, 2), 
    (7, 4), (7, 7), (7, 8), (8, 5), (8, 7), (8, 9), (9, 2), (9, 6), 
    (10, 3), (14, 10), (15, 4)]

pergerm_fidPairsDict = {
  ('Gix',): [
        (0, 5), (1, 0), (1, 1), (2, 2), (2, 5), (2, 9), (3, 3), 
        (3, 4), (3, 8), (4, 0), (4, 2), (4, 7), (4, 8), (4, 10), 
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 8), (6, 7), (6, 8), 
        (6, 9), (7, 0), (7, 4), (8, 5), (8, 9), (9, 5), (10, 8), 
        (10, 10), (12, 2), (12, 4), (12, 7), (13, 2), (13, 3), 
        (13, 9), (14, 0), (14, 5), (14, 6), (15, 5), (15, 8), 
        (15, 9)],
  ('Gii',): [
        (0, 8), (1, 0), (1, 1), (1, 3), (1, 10), (2, 5), (2, 9), 
        (3, 3), (3, 9), (4, 3), (4, 8), (5, 0), (5, 5), (5, 7), 
        (6, 4), (6, 6), (6, 8), (6, 10), (7, 0), (7, 2), (7, 3), 
        (7, 4), (7, 6), (7, 10), (8, 3), (8, 5), (9, 3), (9, 4), 
        (9, 5), (9, 6), (9, 8), (9, 9), (10, 3), (10, 9), (10, 10), 
        (11, 1), (11, 5), (12, 5), (12, 7), (12, 9), (13, 0), 
        (13, 10), (14, 0), (14, 1), (14, 2), (14, 6), (15, 0), 
        (15, 5), (15, 6), (15, 7), (15, 8)],
  ('Giy',): [
        (0, 0), (0, 7), (1, 1), (3, 5), (3, 6), (4, 2), (4, 4), 
        (4, 5), (5, 3), (5, 7), (7, 1), (7, 8), (8, 5), (9, 4), 
        (9, 5), (9, 9), (10, 5), (11, 5), (11, 6), (11, 8), (11, 10), 
        (12, 0), (12, 3), (13, 10), (14, 0), (14, 5), (14, 6), 
        (14, 7), (15, 0), (15, 6), (15, 9)],
  ('Gxi',): [
        (0, 7), (1, 1), (1, 7), (2, 7), (3, 3), (4, 9), (5, 4), 
        (7, 2), (7, 10), (8, 2), (9, 2), (9, 8), (9, 9), (10, 1), 
        (10, 10), (11, 2), (11, 5), (11, 6), (13, 2), (14, 7), 
        (15, 2), (15, 3)],
  ('Gyi',): [
        (3, 1), (4, 1), (4, 2), (5, 0), (5, 1), (5, 7), (6, 0), 
        (6, 8), (7, 2), (7, 4), (7, 9), (8, 0), (8, 7), (9, 2), 
        (9, 3), (10, 9), (10, 10), (14, 7), (14, 9), (15, 10)],
  ('Gix', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5), 
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6), 
        (12, 9), (13, 9), (15, 1)],
  ('Gix', 'Gxi'): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1), 
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10), 
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10), 
        (14, 8), (15, 3)],
  ('Gxi', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Giy', 'Gxi'): [
        (1, 1), (2, 8), (3, 0), (3, 2), (3, 6), (4, 7), (7, 2), 
        (8, 6), (9, 1), (9, 7), (9, 9), (10, 2), (10, 10), (11, 8), 
        (12, 6), (13, 2), (13, 7), (14, 2), (15, 5)],
  ('Gii', 'Gix'): [
        (0, 5), (1, 0), (1, 1), (2, 2), (2, 5), (2, 9), (3, 3), 
        (3, 4), (3, 8), (4, 0), (4, 2), (4, 7), (4, 8), (4, 10), 
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 8), (6, 7), (6, 8), 
        (6, 9), (7, 0), (7, 4), (8, 5), (8, 9), (9, 5), (10, 8), 
        (10, 10), (12, 2), (12, 4), (12, 7), (13, 2), (13, 3), 
        (13, 9), (14, 0), (14, 5), (14, 6), (15, 5), (15, 8), 
        (15, 9)],
  ('Gxi', 'Gii', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Gxi', 'Gii', 'Gii'): [
        (0, 7), (1, 1), (1, 7), (2, 7), (3, 3), (4, 9), (5, 4), 
        (7, 2), (7, 10), (8, 2), (9, 2), (9, 8), (9, 9), (10, 1), 
        (10, 10), (11, 2), (11, 5), (11, 6), (13, 2), (14, 7), 
        (15, 2), (15, 3)],
  ('Gyi', 'Gii', 'Gii'): [
        (3, 1), (4, 1), (4, 2), (5, 0), (5, 1), (5, 7), (6, 0), 
        (6, 8), (7, 2), (7, 4), (7, 9), (8, 0), (8, 7), (9, 2), 
        (9, 3), (10, 9), (10, 10), (14, 7), (14, 9), (15, 10)],
  ('Gxi', 'Gyi', 'Gii'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Giy', 'Gii', 'Gii'): [
        (0, 0), (0, 7), (1, 1), (3, 5), (3, 6), (4, 2), (4, 4), 
        (4, 5), (5, 3), (5, 7), (7, 1), (7, 8), (8, 5), (9, 4), 
        (9, 5), (9, 9), (10, 5), (11, 5), (11, 6), (11, 8), (11, 10), 
        (12, 0), (12, 3), (13, 10), (14, 0), (14, 5), (14, 6), 
        (14, 7), (15, 0), (15, 6), (15, 9)],
  ('Gix', 'Giy', 'Gii'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5), 
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6), 
        (12, 9), (13, 9), (15, 1)],
  ('Gix', 'Gii', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5), 
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6), 
        (12, 9), (13, 9), (15, 1)],
  ('Gix', 'Gii', 'Gii'): [
        (0, 5), (1, 0), (1, 1), (2, 2), (2, 5), (2, 9), (3, 3), 
        (3, 4), (3, 8), (4, 0), (4, 2), (4, 7), (4, 8), (4, 10), 
        (5, 0), (5, 1), (5, 2), (5, 6), (5, 8), (6, 7), (6, 8), 
        (6, 9), (7, 0), (7, 4), (8, 5), (8, 9), (9, 5), (10, 8), 
        (10, 10), (12, 2), (12, 4), (12, 7), (13, 2), (13, 3), 
        (13, 9), (14, 0), (14, 5), (14, 6), (15, 5), (15, 8), 
        (15, 9)],
  ('Gix', 'Gyi', 'Giy'): [
        (3, 0), (4, 4), (5, 1), (5, 8), (6, 5), (7, 3), (8, 6), 
        (8, 7), (9, 5), (10, 3), (11, 4), (14, 0), (14, 6), (14, 9), 
        (15, 5)],
  ('Gii', 'Gyi', 'Giy'): [
        (0, 6), (0, 8), (0, 10), (1, 0), (1, 1), (1, 3), (2, 9), 
        (3, 8), (4, 4), (4, 7), (5, 7), (6, 1), (7, 0), (7, 8), 
        (9, 10), (10, 5), (11, 5), (12, 5), (12, 6), (14, 0), 
        (15, 0), (15, 6), (15, 8)],
  ('Gix', 'Gyi', 'Gix', 'Gyi'): [
        (0, 1), (0, 3), (3, 0), (3, 6), (5, 5), (5, 8), (6, 8), 
        (7, 0), (8, 3), (8, 9), (9, 9), (10, 9), (10, 10), (11, 6), 
        (15, 0)],
  ('Gix', 'Gyi', 'Gyi', 'Gyi'): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4), 
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4), 
        (14, 9), (15, 5), (15, 7)],
  ('Gix', 'Gii', 'Gii', 'Gxi'): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1), 
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10), 
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10), 
        (14, 8), (15, 3)],
  ('Gix', 'Gii', 'Gyi', 'Gii'): [
        (0, 5), (0, 9), (1, 6), (3, 1), (3, 2), (5, 0), (5, 4), 
        (6, 0), (6, 8), (9, 7), (10, 9), (11, 1), (11, 4), (14, 4), 
        (14, 9), (15, 5), (15, 7)],
  ('Gyi', 'Gyi', 'Giy', 'Gyi'): [
        (0, 2), (1, 1), (1, 4), (2, 1), (2, 10), (3, 10), (4, 0), 
        (5, 3), (5, 7), (6, 4), (6, 10), (8, 2), (8, 3), (9, 0), 
        (10, 8), (11, 1), (11, 7), (13, 1), (13, 8)],
  ('Gix', 'Giy', 'Giy', 'Gii'): [
        (0, 4), (0, 5), (0, 7), (1, 1), (1, 6), (2, 3), (4, 10), 
        (5, 4), (6, 8), (7, 4), (7, 10), (8, 8), (8, 9), (10, 5), 
        (11, 5), (11, 6), (11, 9), (13, 10), (14, 1), (14, 9)],
  ('Giy', 'Gix', 'Gix', 'Gix'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5), 
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6), 
        (12, 9), (13, 9), (15, 1)],
  ('Gix', 'Gix', 'Gii', 'Giy'): [
        (0, 0), (0, 6), (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), 
        (4, 8), (5, 5), (6, 7), (7, 6), (8, 9), (9, 9), (10, 2), 
        (10, 8), (11, 10), (12, 6), (12, 9), (13, 1), (13, 9), 
        (15, 1)],
  ('Gxi', 'Gyi', 'Gyi', 'Gii'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gix', 'Gix', 'Gxi', 'Gix'): [
        (0, 0), (1, 5), (2, 4), (3, 3), (3, 5), (5, 2), (6, 1), 
        (6, 8), (6, 10), (8, 6), (10, 2), (10, 8), (10, 10), 
        (11, 8), (12, 1), (13, 1), (13, 4), (13, 6), (13, 10), 
        (14, 8), (15, 3)],
  ('Gxi', 'Gxi', 'Gii', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Giy', 'Giy', 'Gix', 'Gxi'): [
        (0, 6), (3, 0), (5, 0), (6, 7), (7, 1), (8, 3), (9, 9), 
        (10, 4), (10, 9), (12, 9), (13, 2), (14, 5), (14, 8), 
        (14, 10), (15, 6)],
  ('Giy', 'Giy', 'Gxi', 'Gyi', 'Gxi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gix', 'Gxi', 'Gyi', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gxi', 'Gix', 'Giy', 'Gix', 'Giy', 'Gix'): [
        (0, 4), (0, 6), (1, 1), (2, 2), (4, 1), (4, 3), (5, 1), 
        (5, 3), (6, 10), (8, 2), (8, 8), (9, 4), (10, 7), (12, 1), 
        (13, 2), (15, 6), (15, 9)],
  ('Gxi', 'Gix', 'Giy', 'Gxi', 'Giy', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gxi', 'Gxi', 'Gyi', 'Gxi', 'Gyi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gii', 'Gii', 'Gxi', 'Giy', 'Gxi'): [
        (0, 4), (3, 7), (5, 7), (7, 3), (8, 1), (9, 3), (9, 4), 
        (9, 6), (9, 9), (10, 9), (11, 2), (12, 5), (12, 9), (13, 1), 
        (15, 7)],
  ('Gix', 'Gix', 'Giy', 'Gix', 'Giy', 'Giy'): [
        (1, 0), (1, 10), (4, 0), (4, 4), (4, 7), (4, 8), (5, 5), 
        (7, 6), (8, 9), (9, 9), (10, 2), (10, 8), (11, 10), (12, 6), 
        (12, 9), (13, 9), (15, 1)],
  ('Gix', 'Giy', 'Gix', 'Giy', 'Gix', 'Gyi'): [
        (2, 2), (2, 6), (3, 5), (4, 9), (5, 3), (7, 10), (8, 0), 
        (8, 5), (9, 0), (10, 4), (10, 10), (11, 1), (12, 1), 
        (12, 10), (14, 6), (15, 0), (15, 2)],
  ('Gyi', 'Gii', 'Giy', 'Gxi', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gii', 'Gyi', 'Gyi', 'Gix', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gxi', 'Gix', 'Giy', 'Gyi', 'Giy', 'Gxi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gxi', 'Gyi', 'Giy', 'Gyi', 'Gxi', 'Gii'): [
        (0, 2), (1, 1), (1, 4), (4, 2), (5, 1), (5, 6), (5, 7), 
        (6, 4), (6, 10), (7, 1), (8, 2), (9, 0), (11, 7), (13, 1), 
        (14, 1)],
  ('Gxi', 'Giy', 'Gxi', 'Gyi', 'Gix', 'Gii', 'Gxi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Gxi', 'Gyi', 'Gyi', 'Gix', 'Giy', 'Gxi', 'Giy'): [
        (0, 9), (1, 1), (1, 3), (2, 4), (3, 7), (4, 7), (5, 5), 
        (6, 0), (7, 1), (7, 9), (10, 9), (11, 0), (12, 9), (14, 9), 
        (15, 0)],
  ('Gix', 'Gyi', 'Gii', 'Gix', 'Gix', 'Gxi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gix', 'Giy', 'Gix', 'Gyi', 'Gxi', 'Gii', 'Gxi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gii', 'Gii', 'Giy', 'Giy', 'Gii', 'Giy'): [
        (0, 0), (0, 1), (0, 7), (1, 1), (1, 5), (1, 7), (2, 3), 
        (2, 5), (2, 8), (4, 4), (5, 2), (5, 4), (5, 7), (6, 2), 
        (6, 8), (7, 6), (7, 7), (7, 9), (7, 10), (8, 0), (8, 3), 
        (9, 9), (10, 5), (10, 9), (11, 5), (12, 8), (13, 5), 
        (13, 9), (14, 0), (14, 2), (14, 7), (14, 8), (15, 4)],
  ('Giy', 'Gyi', 'Gxi', 'Gyi', 'Gix', 'Gix', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gix', 'Gii'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Gix', 'Giy', 'Giy', 'Gyi', 'Gii', 'Gxi', 'Giy'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Gyi', 'Gix', 'Gxi', 'Gyi', 'Gxi', 'Gii', 'Giy'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Gxi', 'Gii', 'Giy', 'Gxi', 'Gyi', 'Giy', 'Gii'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gii', 'Gxi', 'Giy', 'Gyi', 'Gyi', 'Gix', 'Gyi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Gyi', 'Gxi', 'Giy', 'Gxi', 'Gix', 'Gxi', 'Gyi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gxi', 'Gii', 'Gii', 'Gix', 'Giy', 'Gxi', 'Gyi', 'Gix'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gii', 'Gii', 'Gyi', 'Giy', 'Gix', 'Giy', 'Gix', 'Gxi'): [
        (0, 1), (0, 2), (0, 5), (1, 3), (1, 9), (2, 4), (2, 10), 
        (3, 8), (5, 5), (7, 0), (9, 3), (9, 9), (9, 10), (10, 8), 
        (12, 2), (12, 6), (14, 6), (15, 0), (15, 5)],
  ('Giy', 'Gii', 'Gxi', 'Gxi', 'Gix', 'Gii', 'Gyi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gix', 'Gii', 'Gyi', 'Gii', 'Gyi', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gyi', 'Gii', 'Gix', 'Gyi', 'Gyi', 'Gxi', 'Gix', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Giy', 'Gyi', 'Gix', 'Gix', 'Gxi', 'Gxi', 'Gxi', 'Gyi'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
  ('Gxi', 'Giy', 'Gix', 'Gyi', 'Gix', 'Gii', 'Gxi', 'Giy'): [
        (0, 1), (0, 5), (1, 3), (3, 8), (5, 5), (7, 0), (9, 3), 
        (9, 9), (9, 10), (10, 8), (12, 2), (12, 6), (14, 6), 
        (15, 0), (15, 5)],
}

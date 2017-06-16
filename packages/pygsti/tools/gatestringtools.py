from collections import namedtuple, OrderedDict

def apply_seq(gatestring, sequenceRules):
    rightShifted = (None,) + gatestring[:-1]
    leftShifted = gatestring[1:] + (None,)
    ret = tuple()
    for prevStr, currentStr, nextStr in zip(rightShifted, gatestring, leftShifted):
        broke = False
        for rule, replacement in sequenceRules:
            if rule == (prevStr, currentStr):
                ret += (replacement[1],)
                broke = True
                break
            if rule == (currentStr, nextStr) and replacement[0] != currentStr:
                ret += (replacement[0],) 
                broke = True
                break
        if not broke:
            ret += (currentStr,)
    return ret

'''
Rules:

    AB -> AB' (if B follows A, prime B)

    BA -> B''A (if B precedes A, double-prime B)

    CA -> CA' (if A follows C, prime A)

    BC -> BC' (if C follows B, prime C)


     Desired output:

     BAB ==> B''AB'

     ABA ==> AB'A  (frustrated, so do first replacement: B' not B'')

     CAB ==> CA'B'

     ABC ==> AB'C'
'''

sequenceRules = [
        (('A', 'B'), ('A', 'B\'')),
        (('B', 'A'), ('B\'\'', 'A')),
        (('C', 'A'), ('C', 'A\'')),
        (('B', 'C'), ('B', 'C\''))]

run = lambda s : print(''.join(apply_seq(tuple(s), sequenceRules)))
run('BAB')
run('ABA')
run('CAB')
run('ABC')

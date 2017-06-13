from collections import namedtuple, OrderedDict

def apply_replace(gatestring, replaceRules):
    for rule in replaceRules:
        temp = tuple()
        for item in gatestring:
            if item == rule:
                temp += replaceRules[item]
            else:
                temp += (item,)
        gatestring = temp
    return gatestring

def apply_seq(gatestring, sequenceRules):
    if len(gatestring) <= 1:
        return gatestring
    ret = (gatestring[0],)
    for pair in zip(gatestring[:-1], gatestring[1:]):
        if pair in sequenceRules:
            ret += sequenceRules[pair]
        else:
            ret += (pair[1],) # Current item
    return ret

def pre_process_gatestring(gatestring, sequenceRules, replaceRules):
    '''
    Apply sequencing and replacement rules to a gatestring

    Sequencing
        if B after A
            B -> B'
        if A after B
            A -> A'
        Delayed: BAB -> BA'B'

        General rule form:
            if X after Y
                X -> Z
    Replacement
        A -> C
        C -> C'
        Applied in order: A -> C'

        General rule form:
            X -> Y
    '''
    gatestring = apply_seq(gatestring, sequenceRules)
    gatestring = apply_replace(gatestring, replaceRules)
    return gatestring

sequenceRules = {
    ('A', 'B') : ('B\'',),
    ('B', 'A') : ('A\'',)}

replaceRules = OrderedDict()
replaceRules['A'] = ('C', 'C')
replaceRules['C'] = ('C\'',)

print(pre_process_gatestring(('A', 'B', 'A', 'C'), sequenceRules, replaceRules))

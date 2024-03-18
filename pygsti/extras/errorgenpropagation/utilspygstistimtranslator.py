import stim



'''
returns a dictionary capable of translating pygsti standard gate labels to stim tablue representations of gates
'''
def Gate_Translate_Dict_p_2_s():
    pyGSTi_to_stim_GateDict={
    'Gi'    : stim.Tableau.from_named_gate('I'),
    'Gxpi'  : stim.Tableau.from_named_gate('X'),
    'Gypi'  : stim.Tableau.from_named_gate('Y'),
    'Gzpi'  : stim.Tableau.from_named_gate('Z'),
    'Gxpi2' : stim.Tableau.from_named_gate('SQRT_X'),
    'Gypi2' : stim.Tableau.from_named_gate('SQRT_Y'),
    'Gzpi2' : stim.Tableau.from_named_gate('SQRT_Z'),
    'Gxmpi2': stim.Tableau.from_named_gate('SQRT_X_DAG'),
    'Gympi2': stim.Tableau.from_named_gate('SQRT_Y_DAG'),
    'Gzmpi2': stim.Tableau.from_named_gate('SQRT_Z_DAG'),
    'Gh'    : stim.Tableau.from_named_gate('H'),
    'Gxx'   : stim.Tableau.from_named_gate('SQRT_XX'),
    'Gzz'   : stim.Tableau.from_named_gate('SQRT_ZZ'),
    'Gcnot' : stim.Tableau.from_named_gate('CNOT'),
    'Gswap' : stim.Tableau.from_named_gate('SWAP')
    }
    return pyGSTi_to_stim_GateDict


'''
returns a dict translating the stim tableu (gate) key to pyGSTi gate keys
TODO: change the stim tablues to tablues keys
'''
def Gate_Translate_Dict_s_2_p():
    dict = Gate_Translate_Dict_p_2_s()
    return {v: k for k, v in dict.items()}

'''
Takes a layer of pyGSTi gates and composes them into a single stim Tableu
'''
def pyGSTiLayer_to_stimLayer(player,qubits,MultiGateDict={},MultiGate=False):
    slayer=stim.Tableau(qubits)
    stimDict=Gate_Translate_Dict_p_2_s()
    for sub_lbl in player:
        if not MultiGate:
            temp = stimDict[sub_lbl.name]
        else:
            temp = stimDict[MultiGateDict[sub_lbl.name]]
        slayer.append(temp,sub_lbl.qubits)
    return slayer

'''
Takes the typical pygsti label for paulis and returns a stim PauliString object
'''
def pyGSTiPauli_2_stimPauli(pauli):
    return stim.PauliString(pauli)


'''
Converts a stim paulistring to the string typically used in pysti to label paulis
warning: stim ofter stores a pauli phase in the string (i.e +1,-1,+i,-i) this is assumed positive 
in this function, if the weight is needed please store paulistring::weight prior to applying this function
'''
def stimPauli_2_pyGSTiPauli(pauliString):
    return str(pauliString)[1:].replace('_',"I")
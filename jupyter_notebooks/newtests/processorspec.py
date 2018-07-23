from pygsti.objects import ProcessorSpec

def test_processorspec():
    
    # Tests init a pspec using standard gatenames, and all standards.
    n = 3
    gate_names = ['Gi','Gh','Gp','Gxpi','Gypi','Gzpi','Gpdag','Gcphase']
    ps = ProcessorSpec(n,gate_names=gate_names)

    # Tests init a pspec containing 1 qubit (as special case which could break)
    n = 1
    gate_names = ['Gi','Gh','Gp','Gxpi','Gypi','Gzpi','Gpdag','Gcphase']
    ps = ProcessorSpec(n,gate_names=gate_names)
    
    # Note: More complex pspec objects are created for testing clifford compilers.
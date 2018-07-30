from pygsti.extras import rb
import pygsti

def test_rb_simulate():

    n = 3
    glist = ['Gi','Gxpi','Gypi','Gzpi','Gh','Gp','Gcphase']
    availability = {'Gcphase':[(0,1),(1,2)]}
    pspec = pygsti.obj.ProcessorSpec(n,glist,availability=availability,verbosity=0)

    errormodel = rb.simulate.create_iid_pauli_error_model(pspec, oneQgate_errorrate=0.01, twoQgate_errorrate=0.05, 
                                                          idle_errorrate=0.005, measurement_errorrate=0.05, 
                                                          ptype='uniform')
    errormodel = rb.simulate.create_iid_pauli_error_model(pspec, oneQgate_errorrate=0.001, twoQgate_errorrate=0.01, 
                                                          idle_errorrate=0.005, measurement_errorrate=0.05, 
                                                          ptype='X')

    out = rb.simulate.rb_with_pauli_errors(pspec,errormodel,[0,2,4],2,3,filename='testfiles/simtest.txt',rbtype='CRB',
                                    returndata=True, verbosity=0)

    errormodel = rb.simulate.create_locally_gate_independent_pauli_error_model(pspec, [0.0,0.01,0.02], 
                                                                               [0.0,0.1,0.01],ptype='uniform')

    out = rb.simulate.rb_with_pauli_errors(pspec,errormodel,[0,10,20],2,2,filename='testfiles/simtest.txt',rbtype='DRB',
                                    returndata=True, verbosity=0)
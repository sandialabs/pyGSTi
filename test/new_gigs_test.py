import pygsti
import numpy as np
from pygsti.construction import std1Q_XY as std

print "================================================================"
print "================================================================"
print "================================================================"
print "================================================================"

g0 = std.gs_target.depolarize(max_gate_noise=0.01,seed=12)
g0.set_rhovec( np.array([[1/np.sqrt(2)],[0.001],[0.001],[1/np.sqrt(2)]],'d'))

ggi = pygsti.objects.GaugeInvGateSet()
ggi.from_gateset( g0, verbosity=10 )
g1 = ggi.to_gateset(verbosity=0)

np.random.seed(1234)
B = np.random.rand(4,4)
g2 = g1.copy(); g2.transform(B)
ggi2 = pygsti.objects.GaugeInvGateSet()
ggi2.from_gateset( g2, debug=B, verbosity=10 )
g3 = ggi2.to_gateset(verbosity=0)

if False:
    print "Comparisons:"
    print "rho:"
    pygsti.print_mx(g0.rhoVecs[0])
    print "vs"
    pygsti.print_mx(g1.rhoVecs[0])
    
    
    print "E:"
    pygsti.print_mx(g0.EVecs[0])
    print "vs"
    pygsti.print_mx(g1.EVecs[0])
    
    
    for gl in g0:
        print "Gate %s:" % gl
        pygsti.print_mx(g0[gl])
        print "vs"
        pygsti.print_mx(g1[gl])
print "Tesing g0 => ggi (gauge-inv) => g1"
print "Checking that g0 == g1: ",
assert(g0.frobeniusdist(g1) < 1e-6)
print "OK"

print "Tesing g2 => ggi2 (gauge-inv) => g3"
assert(g2.frobeniusdist(g3) < 1e-6)
print "OK"

print "ggi has %d params" % ggi.num_params()
print "g0 has %d params (%d non-gauge)" % (g0.num_params(),
                                           g0.num_nongauge_params())

print "EParams:"
print ggi.E_params - ggi2.E_params

print "\nDParams:"
for i,(D1,D2) in enumerate(zip(ggi.D_params,ggi2.D_params)):
    print "%d: " % i, D1-D2

print "\nBParams:"
for i,(B1,B2) in enumerate(zip(ggi.B0_params[1:],ggi2.B0_params[1:]), start=1):
    print "%d: " % i, B1-B2

print "Dist between ggi and ggi2 = ", np.linalg.norm( ggi.to_vector() - ggi2.to_vector() ) 

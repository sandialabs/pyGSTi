import pygsti
from pygsti.construction import std1Q_XY as std

g0 = std.gs_target
ggi = pygsti.objects.GaugeInvGateSet()
ggi.from_gateset( g0, verbosity=0 )
g1 = ggi.to_gateset(verbosity=0)

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

print "ggi has %d params" % ggi.num_params()
print "g0 has %d params (%d non-gauge)" % (g0.num_params(),
                                           g0.num_nongauge_params())

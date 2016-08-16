import pygsti
import numpy as np
from pygsti.construction import std1Q_XYI as std
# DISABLED: out of date
'''
print("================================================================")
print("================================================================")
print("================================================================")
print("================================================================")

g0 = std.gs_target.depolarize(max_gate_noise=0.1,seed=12)
g0.set_rhovec( np.array([[1/np.sqrt(2)],[0.1],[0.1],[1/np.sqrt(2)]],'d'))

Gi_mx = np.array( [[1.0,   0,    0,    0],
                   [0,  0.99,    0,    0],
                   [0,     0, 0.98,    0],
                   [0,     0,    0, 0.97]], 'd')
#g0.set_gate("Gi", pygsti.objects.FullyParameterizedGate( Gi_mx ))
#g0 = g0.kick(0.01, seed=1211)

ggi = pygsti.objects.GaugeInvGateSet()
print("====== FROM GATESET => GIG ============================================")
ggi.from_gateset( g0, verbosity=10 )

print("====== GIG => TO GATESET ============================================")
g1 = ggi.to_gateset(verbosity=0)

if False:
    print("Comparisons:")
    print("rho:")
    pygsti.print_mx(g0.rhoVecs[0])
    print("vs")
    pygsti.print_mx(g1.rhoVecs[0])


    print("E:")
    pygsti.print_mx(g0.EVecs[0])
    print("vs")
    pygsti.print_mx(g1.EVecs[0])


    for gl in g0:
        print("Gate %s:" % gl)
        pygsti.print_mx(g0[gl])
        print("vs")
        pygsti.print_mx(g1[gl])

print("Tesing g0 => ggi (gauge-inv) => g1")
print("Checking that g0 == g1: |g0-g1| = ", g0.frobeniusdist(g1))
assert(g0.frobeniusdist(g1) < 1e-6)
print("OK")

print("ggi has %d params" % ggi.num_params())
print("g0 has %d params (%d non-gauge)" % (g0.num_params(),
                                           g0.num_nongauge_params()))



np.random.seed(1234)
B = np.random.rand(4,4)
g2 = g0.copy(); g2.transform(B)
Bi = np.linalg.inv(B)
#print " -> PRE g0 matrix:"; pygsti.print_mx(g0['Gx'])
#print " -> PRE debug matrix:"; pygsti.print_mx(np.dot(B,np.dot(g2['Gx'],Bi)))
#print " -> PRE test matrix:"; pygsti.print_mx(np.dot(Bi,np.dot(g0['Gx'],B)))
ggi2 = pygsti.objects.GaugeInvGateSet()
print("====== FROM B-GAUGED-GATESET => GIG2 ============================================")
print(g2)
ggi2.from_gateset( g2, debug=B, verbosity=10, fix=0.265 )

print("====== GIG2 => TO GATESET ============================================")
g3 = ggi2.to_gateset(verbosity=0)


print("Tesing g2 => ggi2 (gauge-inv) => g3")
print("|g2-g3| = ",g2.frobeniusdist(g3))
#print g2.EVecs[0] - g3.EVecs[0], "\n"
#print g2.rhoVecs[0] - g3.rhoVecs[0], "\n"
#print g2['Gi'] - g3['Gi'], "\n"
#print g2['Gx'] - g3['Gx'], "\n"
#print g2['Gy'] - g3['Gy'], "\n"
##print g2['Gx'],"\n"
##print g3['Gx'],"\n"
assert(g2.frobeniusdist(g3) < 1e-6)
print("OK")

print("Dist between ggi and ggi2 = ", np.linalg.norm( ggi.to_vector() - ggi2.to_vector() ))

if True:
    print("EParams:")
    print(ggi.E_params)

    print("\nDParams:")
    for i,(D1,D2) in enumerate(zip(ggi.D_params,ggi2.D_params)):
        print("%d: " % i, D1)

    print("\nBParams:")
    for i,(B1,B2) in enumerate(zip(ggi.B0_params[1:],ggi2.B0_params[1:]), start=1):
        print("%d: " % i, B1)

    print("diff(EParams):")
    print(ggi.E_params - ggi2.E_params)

    print("\ndiff(DParams):")
    for i,(D1,D2) in enumerate(zip(ggi.D_params,ggi2.D_params)):
        print("%d: " % i, D1-D2)

    print("\ndiff(BParams):")
    for i,(B1,B2) in enumerate(zip(ggi.B0_params[1:],ggi2.B0_params[1:]), start=1):
        print("%d: " % i, B1-B2)

assert(np.linalg.norm( ggi.to_vector() - ggi2.to_vector() ) < 1e-6 )

print("DONE")
'''

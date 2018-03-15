from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
import copy as _copy
import time as _time

from . import sampler as _samp

import matplotlib.pyplot as _plt
from scipy.optimize import minimize as _minimize


def simulate_full_grb_experiment(ds, mname, lengths, circuits_per_length, sampler='weights', 
                                 sampler_args = {'two_qubit_weighting' :0.5},
                                 twirled=True, stabilizer=True, verbosity=1, plot=False, 
                                 store=True):
   
    """
    A function for simulating a full generator RB experiment, with only the minimal necessary
    user input.
    
    """
    
    start = _time.time()
    if verbosity > 0:
        print("*********************************************************************")
        print("******************* generator RB simulator **************************")
        print("*********************************************************************")
        print("")
    
    # Lists of average survival probabilities and average circuit depths, at each sequence length
    ASPs = []
    ACDs = []
    
    # Dictionaries which will contain the full set of probabilities and the circuits sampled
    probabilities = {}
    circuits = {}
    
    # Main loop of the simulation
    for i in range(0,len(lengths)):
        
        if verbosity > 0:
            print("--------------------------------------------------")
            print("Starting simulations for length {} sequences...".format(lengths[i]))
            print("--------------------------------------------------")
            print("")
        
        # Temporary lists for keep the survival probabilities and circuit depths at this length, 
        # to average and append to the ASPs and ACDs arrays.
        SPs = []
        CDs = []
        
        # A list of the probabilities and circuits at this length.
        circuits[lengths[i]] = []
        probabilities[lengths[i]] = []
        
        # Do the simulations for all circuits at the jth length
        for j in range(0,circuits_per_length[i]):
            
            if verbosity > 0:
                print("- Starting simulation circuit {} of {} at length {}".format(j+1, circuits_per_length[i], 
                                                                                         lengths[i]))
            
            start_for_this_circuit = _time.time()
            
            # Sample a generator RB circuit, and add the circuit to the circuit list
            if verbosity > 0:
                print(" - Sampling and compiling an RB circuit...",end='')
            sampled_circuit = _samp.construct_grb_circuit(ds, lengths[i], sampler=sampler, 
                                                          sampler_args=sampler_args, twirled=twirled,
                                                          stabilizer=stabilizer)
            circuits[lengths[i]].append(sampled_circuit)
            if verbosity > 0:    
                print("Complete.")
                print("   * Circuit size: {}".format(sampled_circuit.size()))
                print("   * Circuit depth: {}".format(sampled_circuit.depth()))
                print("   * Circuit two-qubit gate count: {}".format(sampled_circuit.twoqubit_gatecount()))
            if verbosity > 0:
                print(" - Beginning circuit simulation...",end='')
            # Simulate the circuit, using the built-in simulator of the Circuit object
            probabilities[lengths[i]].append(sampled_circuit.simulate(ds.models[mname],store=store))
            if verbosity > 0:    
                print("Complete.")    
            # Store the simulated survival prob, and the circuit depth.
            SPs.append(probabilities[lengths[i]][j][tuple(_np.zeros((ds.number_of_qubits),int))])
            CDs.append(circuits[lengths[i]][j].depth())
            
            end = _time.time()
            
            if verbosity > 0:
                print("   * Success probabilities :  ",end='')
                print(SPs)
                print("   * Total time elapsed : ", end - start)
                print("   * This circuit took : ", end - start_for_this_circuit)

            #
            # This doesn't work and needs fixing!
            #
            #if plot:
            #    print("   * The decay obtained so far is:") 
            #    _plt.plot(ASPs + _np.mean(_np.array(SPs)),'o-')
            #    _plt.show()
            
        ASPs.append(_np.mean(_np.array(SPs)))
        ACDs.append(_np.mean(_np.array(CDs)))
        
        def obj_func(params,lengths,ASPs):
            A,Bs,f = params
            return _np.sum((A+(Bs-A)*f**lengths-ASPs)**2)
        
        #print(lengths[:i+1],ASPs)
        p0 = [1/2**(ds.number_of_qubits),0.5,0.9]        
        fit_out = _minimize(obj_func, p0, args=(lengths[:i+1],ASPs),method='L-BFGS-B')
        A = fit_out.x[0]
        B = fit_out.x[1] - fit_out.x[0]
        p = fit_out.x[2]
        
        if verbosity > 0:
            print("**** The current fit parameters are :  A = {}, B = {}, f = {} ****".format(A,B,p))
            print("")
                
        fit_parameters = [A,B,p]
        auxillary_out = {}
        auxillary_out['ACDs'] = ACDs
        auxillary_out['circuits'] = circuits
        auxillary_out['probabilities'] = probabilities
            
    return ASPs, fit_parameters, auxillary_out 


def simulate_full_cliffordrb_experiment(ds, mname, lengths, circuits_per_length, verbosity=1,
                                        plot=False, store=True):
   
    """
    A function for simulating a full generator RB experiment, with only the minimal necessary
    user input.
    
    """
    
    start = _time.time()
    if verbosity > 0:
        print("*********************************************************************")
        print("******************** Clifford RB simulator **************************")
        print("*********************************************************************")
        print("")
    
    # Lists of average survival probabilities and average circuit depths, at each sequence length
    ASPs = []
    ACDs = []
    
    # Dictionaries which will contain the full set of probabilities and the circuits sampled
    probabilities = {}
    circuits = {}
    
    # Main loop of the simulation
    for i in range(0,len(lengths)):
        
        if verbosity > 0:
            print("--------------------------------------------------")
            print("Starting simulations for length {} sequences...".format(lengths[i]))
            print("--------------------------------------------------")
            print("")
        # Temporary lists for keep the survival probabilities and circuit depths at this length, 
        # to average and append to the ASPs and ACDs arrays.
        SPs = []
        CDs = []
        
        # A list of the probabilities and circuits at this length.
        circuits[lengths[i]] = []
        probabilities[lengths[i]] = []
        
        # Do the simulations for all circuits at the jth length
        for j in range(0,circuits_per_length[i]):
            
            start_for_this_circuit = _time.time()
            
            if verbosity > 0:
                print(" - Sampling and compiling an RB circuit...")
            # Sample a generator RB circuit, and add the circuit to the circuit list
            sampled_circuit = _samp.construct_cliffordrb_circuit(ds, lengths[i])
            circuits[lengths[i]].append(sampled_circuit)
            if verbosity > 0:    
                print("Complete.")
                print("   * Circuit size: {}".format(sampled_circuit.size()))
                print("   * Circuit depth: {}".format(sampled_circuit.depth()))
                print("   * Circuit two-qubit gate count: {}".format(sampled_circuit.twoqubit_gatecount()))
            if verbosity > 0:
                print(" - Beginning circuit simulation...",end='')
                
            # Simulate the circuit, using the built-in simulator of the Circuit object
            probabilities[lengths[i]].append(sampled_circuit.simulate(ds.models[mname],store=store))
            SPs.append(probabilities[lengths[i]][j][tuple(_np.zeros((ds.number_of_qubits),int))])
            # Store the simulated survival prob, and the circuit depth.
            CDs.append(circuits[lengths[i]][j].depth())
            
            end = _time.time()
                       
            if verbosity > 0:
                print(" - completed simulation for {} of {} circuits at this length".format(j+1,circuits_per_length[i]))
                print("   * Total time elapsed", end - start)
                print("   * This circuit took", end - start_for_this_circuit)
            
            #
            # This doesn't work and needs fixing!
            #
            #if plot:
            #    print("   * The decay obtained so far is:") 
            #    _plt.plot(ASPs + _np.mean(_np.array(SPs)),'o-')
            #    _plt.show()
            
        ASPs.append(_np.mean(_np.array(SPs)))
        ACDs.append(_np.mean(_np.array(CDs)))
        
        def obj_func(params,lengths,ASPs):
            A,Bs,f = params
            return _np.sum((A+(Bs-A)*f**lengths-ASPs)**2)
        
        #print(lengths[:i+1],ASPs)
        p0 = [1/2**(ds.number_of_qubits),0.5,0.9]        
        fit_out = _minimize(obj_func, p0, args=(lengths[:i+1],ASPs),method='L-BFGS-B')
        A = fit_out.x[0]
        B = fit_out.x[1] - fit_out.x[0]
        p = fit_out.x[2]
        
        if verbosity > 0:
            print("   * The generator RB decay rate is",p)
            print("   * The generator RB error rate is",(2**(ds.number_of_qubits)-1)*(1-p)/2**(ds.number_of_qubits))
            print("   * The full fit parameters are A = {}, B = {}, f = {}".format(A,B,p))
            print("")
          
        fit_parameters = [A,B,p]
        auxillary_out = {}
        auxillary_out['ACDs'] = ACDs
        auxillary_out['circuits'] = circuits
        auxillary_out['probabilities'] = probabilities
            
    return ASPs, fit_parameters, auxillary_out 
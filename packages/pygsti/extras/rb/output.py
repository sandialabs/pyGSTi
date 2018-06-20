def write_empty_rb_files(filename, m_list, K_m, group_or_gateset, 
                         inverse=True, interleaved_list=None, alias_maps=None, 
                         seed=None, randState=None):
    """
    A wrapper for list_random_rb_clifford_strings which also writes output
    to disk.

    This function returns the same value as list_random_rb_clifford_strings,
    and also:

    - saves the clifford strings in an empty data set file by adding ".txt"
      to `filename`.
    - saves each set of strings to a gatestring list text file by adding
      "_<gate-label-set-name>.txt" to `filename`.  
      
    For example, if "primitive" is the only key of `alias_maps`, and 
    `filename` is set to "test", then the following files are created:

    - "test.txt" (empty dataset with clifford-labelled strings)
    - "test_clifford.txt" (gate string list with clifford-label strings)
    - "test_primitive.txt" (gate string list with primitive-label strings)

    Parameters
    ----------
    filename : str
        The base name of the files to create (see above).

    m_min : integer
        Smallest desired Clifford sequence length.
    
    m_max : integer
        Largest desired Clifford sequence length.
    
    Delta_m : integer
        Desired Clifford sequence length increment.

    clifford_group : MatrixGroup
        Which Clifford group to use.

    K_m_sched : int or dict
        If an integer, the fixed number of Clifford sequences to be sampled at
        each length m.  If a dictionary, then a mapping from Clifford
        sequence length m to number of Cliffords to be sampled at that length.
    
    alias_maps : dict of dicts, optional
        If not None, a dictionary whose keys name other gate-label-sets, e.g.
        "primitive" or "canonical", and whose values are "alias" dictionaries 
        which map the clifford labels (defined by `clifford_group`) to those
        of the corresponding gate-label-set.  For example, the key "canonical"
        might correspond to a dictionary "clifford_to_canonical" for which 
        (as one example) clifford_to_canonical['Gc1'] == ('Gy_pi2','Gy_pi2').
            
    seed : int, optional
        Seed for random number generator; optional.

    randState : numpy.random.RandomState, optional
        A RandomState object to generate samples from. Can be useful to set
        instead of `seed` if you want reproducible distribution samples across
        multiple random function calls but you don't want to bother with
        manually incrementing seeds between those calls.
    

    Returns
    -------
    dict or list
        If `alias_maps` is not None, a dictionary of lists-of-gatestring-lists
        whose keys are 'clifford' and all of the keys of `alias_maps` (if any).
        Values are lists of `GateString` lists, one for each K_m value.  If
        `alias_maps` is None, then just the list-of-lists corresponding to the 
        clifford gate labels is returned.
    """
    base_filename = filename
    if interleaved_list is not None:
        base_filename = filename+'_baseline' 
        
    # line below ensures random_string_lists is *always* a dictionary
    alias_maps_mod = {} if (alias_maps is None) else alias_maps      
    random_string_lists = \
        create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = None, alias_maps = alias_maps_mod, 
                                  seed=seed, randState=randState)
    #always write uncompiled gates to empty dataset (in future have this be an arg?)
    _io.write_empty_dataset(base_filename+'.txt', list(
            _itertools.chain(*random_string_lists['uncompiled'])))
    for gstyp,strLists in random_string_lists.items():
        _io.write_gatestring_list(base_filename +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
        
    if interleaved_list is None:
        if alias_maps is None: 
            return random_string_lists['uncompiled'] 
            #mimic list_random_rb_clifford_strings return value
        else: return random_string_lists
        
    else:
        all_random_string_lists = {}
        if alias_maps is None: 
            all_random_string_lists['baseline'] = random_string_lists['uncompiled']
        else:
            all_random_string_lists['baseline'] = random_string_lists
        
        for interleaved in interleaved_list:
            # No seed allowed here currently, as currently no way to make it different to
            # the seed for the baseline decay
            filename_interleaved = filename+'_interleaved_'+interleaved
            random_string_lists = \
                       create_random_gatestrings(m_list, K_m, group_or_gateset,inverse,
                                  interleaved = interleaved, alias_maps = alias_maps_mod)
            _io.write_empty_dataset(filename_interleaved+'.txt', list(
                _itertools.chain(*random_string_lists['uncompiled'])))
            for gstyp,strLists in random_string_lists.items():
                _io.write_gatestring_list(filename_interleaved +'_%s.txt' % gstyp,
                                  list(_itertools.chain(*strLists)))
                
            if alias_maps is None: 
                all_random_string_lists[interleaved] = random_string_lists['uncompiled']
            else:
                all_random_string_lists[interleaved] = random_string_lists
            
        return all_random_string_lists   


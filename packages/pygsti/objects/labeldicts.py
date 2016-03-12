import collections as _collections
import numpy as _np

import spamvec as _sv
import gate as _gate

class PrefixOrderedDict(_collections.OrderedDict):
    def __init__(self, prefix, items=[]):
        #** Note: if change __init__ signature, update __reduce__ below
        self._prefix = prefix
        super(PrefixOrderedDict,self).__init__(items)

    def __setitem__(self, key, val):
        if not key.startswith(self._prefix):
            raise ValueError("All keys must be strings, " +
                             "beginning with the prefix '%s'" % self._prefix)
        super(PrefixOrderedDict,self).__setitem__(key, val)

    def __reduce__(self):
        items = [(k,v) for k,v in self.iteritems()]
        return (PrefixOrderedDict, (self._prefix, items), None) 



class OrderedSPAMVecDict(PrefixOrderedDict):
    def __init__(self, parent, default_param, remainderLabel, prefix, items=[]):
        #** Note: if change __init__ signature, update __reduce__ below
        self.parent = parent # dimension == parent.dim
        self.default_param = default_param  # "tp" or "full"
        self.remainderLabel = remainderLabel
        super(OrderedSPAMVecDict,self).__init__(prefix, items)

    def _check_dim(self, vec):
        if self.parent is None: return
        if self.parent.dim is None:
            self.parent.dim = len(vec)
        elif self.parent.dim != len(vec):
            raise ValueError("Cannot add vector with dimension" +
                             "%d to gateset of dimension %d"
                             % (len(vec),self.parent.dim))

    def __getitem(self, key):
        if key == self.remainderLabel:
            if self.parent is None or self.parent.identityVec is None:
                raise ValueError("Cannot compute remainder vector because "
                                 + " identity vector is not set!")
            return parent.identityVec - sum(self)
        return super(OrderedSPAMVecDict,self).__getitem__(key)


    def __setitem__(self, key, vec):
        self._check_dim(vec)

        if isinstance(vec, _sv.SPAMVec):  #if we're given a SPAMVec object...
            #just replace or create vector
            if vec.transposed:
                raise ValueError("I store only *column* SPAM vectors - please" +
                                 " transpose this vector first")
            super(OrderedSPAMVecDict,self).__setitem__(key, vec) 

        elif key in self: #if a SPAMVec object already exists...
            #try to set its value
            super(OrderedSPAMVecDict,self).__getitem__(key).set_vector(vec)

        else:
            #otherwise, we've been given a non-SPAMVec-object that doesn't 
            # exist yet, so use default creation flags to make one:
            if self.default_param == "tp":
                vecObj = _sv.TPParameterizedSPAMVec(vec)
            elif self.default_param == "full":
                vecObj = _sv.FullyParameterizedSPAMVec(vec)
            elif self.default_param == "static":
                vecObj = _sv.StaticSPAMVec(vec)
            else: 
                raise ValueError("Invalid default_param: %s" % self.default_param)

            super(OrderedSPAMVecDict,self).__setitem__(key, vecObj)

    def copy(self):
        return OrderedSPAMVecDict(self.parent, self.default_param,
                           self.remainderLabel, self._prefix,
                           [(lbl,val.copy()) for lbl,val in self.iteritems()])

    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v) for k,v in self.iteritems()]
        return (OrderedSPAMVecDict, 
                (None, self.default_param,
                 self.remainderLabel, self._prefix, items), None)

        



class OrderedGateDict(PrefixOrderedDict):
    def __init__(self, parent, default_param, prefix, items=[]):
        #** Note: if change __init__ signature, update __reduce__ below
        self.parent = parent # dimension == parent.dim
        self.default_param = default_param  # "tp" or "full"
        super(OrderedGateDict,self).__init__(prefix, items)


    def _check_dim(self, M):
        if isinstance(M, _gate.Gate): 
            gate_dim = M.dim
        else:
            try:
                d1 = len(M)
                d2 = len(M[0])
            except:
                raise ValueError("%s doesn't look like a 2D array/list" % M)
            if any([len(row) != d1 for row in M]):
                raise ValueError("%s is not a *square* 2D array" % M)
            gate_dim = d1

        #Dimension check
        if self.parent is None: return
        if self.parent.dim is None:
            self.parent.dim = gate_dim
        elif self.parent.dim != gate_dim:
            raise ValueError("Cannot add gate with dimension " +
                             "%d to gateset of dimension %d"
                             % (gate_dim,self.parent.dim))


    def __setitem__(self, key, M):
        self._check_dim(M)

        if isinstance(M, _gate.Gate):  #if we're given a Gate object...
            #just replace or create vector
            super(OrderedGateDict,self).__setitem__(key, M) 

        elif key in self: #if a Gate object already exists...
            #try to set its value
            super(OrderedGateDict,self).__getitem__(key).set_matrix(_np.asarray(M))

        else:
            #otherwise, we've been given a non-Gate-object that doesn't 
            # exist yet, so use default creation flags to make one:
            if self.default_param == "tp":
                gateObj = _gate.TPParameterizedGate(M)
            elif self.default_param == "full":
                gateObj = _gate.FullyParameterizedGate(M)
            elif self.default_param == "static":
                gateObj = _gate.StaticGate(M)
            else: 
                raise ValueError("Invalid default_param: %s" % 
                                 self.default_param)

            super(OrderedGateDict,self).__setitem__(key, gateObj)

    def copy(self):
        return OrderedGateDict(self.parent, self.default_param, self._prefix,
                           [(lbl,val.copy()) for lbl,val in self.iteritems()])

    def __reduce__(self):
        #Call constructor to create object, but with parent == None to avoid
        # circular pickling of GateSets.  Must set parent separately.
        items = [(k,v) for k,v in self.iteritems()]
        return (OrderedGateDict, 
                (None, self.default_param, self._prefix, items), None)




class OrderedSPAMLabelDict(_collections.OrderedDict):
    def __init__(self, remainderLabel, items=[]):
        #** Note: if change __init__ signature, update __reduce__ below
        self.remainderLabel = remainderLabel
        super(OrderedSPAMLabelDict,self).__init__(items)

    def __setitem__(self, key, val):
        if type(key) != str:
            raise ValueError("SPAM labels must be strings!")
        if type(val) != tuple or len(val) != 2:
            raise ValueError("SPAM label values must be 2-tuples!")

        rhoLabel, eLabel = val        
        if rhoLabel == self.remainderLabel:
            if eLabel != self.remainderLabel: 
                raise ValueError("POVM label must always ==" +
                                 "%s when preparation " % self.remainderLabel +
                                 "label does")

            # This spam label generates probabilities which equal
            #  1 - sum(other spam label probs)
            _warnings.warn( "You have choosen to use a gate set in a sum-to-one"
                            + "mode in which the spam label %s " % key
                            + "will generate probabilities equal to 1.0 minus "
                            + "the sum of all the probabilities of the other "
                            + "spam labels.")
        
        # if inserted *value* already exists, clobber so values are unique
        for k,v in self.iteritems():
            if val == v: del self[k]

        super(OrderedSPAMLabelDict,self).__setitem__(key,val)

    def copy(self):
        return OrderedSPAMLabelDict(self.remainderLabel,
                                    [(lbl,val) for lbl,val in self.iteritems()])

    def __reduce__(self):
        items = [(k,v) for k,v in self.iteritems()]
        return (OrderedSPAMLabelDict, (self.remainderLabel, items), None)

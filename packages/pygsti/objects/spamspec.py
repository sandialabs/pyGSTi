import gatestring as _gatestring

class SpamSpec(object):
    """ 
    Encapsulates a rho- or E-vector index paired with
    a gate string, which serves as either a "rho-specifier" or
    an "E-specifier" respectively.  A rho-specifier means a state
    preparation followed by the gate string, while an E-specifier
    means a gate string followed by a measurement (Note: gate
    strings are performed in left-to-right order!)
    """
    
    def __init__(self,index,gatestring):
        """
        Create a new SpamSpec object 

        Parameters
        ----------
        index : integer
          rho- or E-vector index
          
        gatestring : tuple or GateString
          gate string, evaluated in left-to-right order, which 
          precedes or follows the E- or rho-vector indexed by
          index, respectively.
        """
        self.i = index
        self.str = _gatestring.GateString(gatestring) 
          #this makes sure self.str is always a gatestring object

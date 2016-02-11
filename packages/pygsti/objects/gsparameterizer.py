#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation              
#    This Software is released under the GPL license detailed    
#    in the file "license.txt" in the top-level pyGSTi directory 
#*****************************************************************
""" Defines gate set parameterizer classes """

class GateSetParameterizer(object):
    """ 
    Base class specifying the "parameterizer" interface for GateSets.

    A parameterizer object is held by each GateSet, and the GateSet uses
    it to translate between "gateset parameters" and raw gate matrices.  Thus,
    a parameterizer object abstracts away the details of particular ways of 
    parameterizing a set of gates (along with SPAM ops).

    As such, a parameterizer object typically holds Gate and SPAMOp objects,
    to which it can defer some of the gate-set-parameterization details to.
    However, it is important to note that there can exist parameters "global"
    to the entire gate set that do not fall within any particular gate or
    SPAM operation (e.g. explicitly gauge-transforming parameters - which
    include gate-set basis changes), and these belong directly to the
    parameterizer object.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")



#Note: perhaps contain a "basis" string here too - to specify what 
# basis ("std", "gm", "pp") the gates should be interpreted as being in??
class StandardParameterizer(GateSetParameterizer):
    """ 
    Parameterizes gates and SPAM ops as generic Gate and SPAMOp objects,
    and contains addition of "global" gauge parameters.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")



class FullParameterizer(StandardParameterizer):
    """ 
    Special case of StandardParameterizer where each gate is and SPAM
    operation are fully parameterized.  Because of this, additional
    "global gauge" parameters are not needed.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")


class GaugeInvParameterizer(GateSetParameterizer):
    """ 
    Parametrizes a gate set using a minimal set of gauge invariant parameters.
    """

    def __init__(self):
        """
        Create a new parameterizer object.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def get_gate_matrix(self, gateLabel):
        """
        Build and return the raw gate matrix for the given gate label.
        
        Parameters
        ----------
        gateLabel : string
            The gate label
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")

    def set_gate_matrix(self, gateLabel, mx):
        """
        Attempts to modify gate set parameters so that the specified raw
        gate matrix becomes mx.  Will raise ValueError if this operation
        is not possible.
        
        Parameters
        ----------
        gateLabel : string
            The gate label.

        mx : numpy array
            Desired raw gate matrix.
            
        Returns
        -------
        numpy array
            The 2-dimensional gate matrix.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def from_vector(self, v):
        """
        Set the gate set parameters using an array of values.

        Parameters
        ----------
        v : numpy array
            A 1D array of parameter values.

        Returns
        -------
        None
        """
        raise NotImplementedError("Should be implemented by derived class")


    def to_vector(self):
        """
        Get the gate set parameters as an array of value.

        Returns
        -------
        numpy array
            The gate set parameters as a 1D array.
        """
        raise NotImplementedError("Should be implemented by derived class")


    def deriv_wrt_params(self):
        """
        Construct a matrix of derivatives whose columns correspond
        to gate set parameters and whose rows correspond to elements
        of the raw gate matrices and raw SPAM vectors.

        Each column is the length of a vectorizing the raw gateset elements
        and there are num_params(...) columns.  If the gateset is fully 
        parameterized (i.e. gate-set-parameters <==> gate-set-elements) then
        the resulting matrix will be the (square) identity matrix.

        Returns
        -------
        numpy array
        """
        raise NotImplementedError("Should be implemented by derived class")

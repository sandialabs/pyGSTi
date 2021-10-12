"""
Defines OverdispersionModel objects
"""

import numpy as np
from warnings import warn


class OverdispersionModel():

"""
    A factory class for overdispersion models for GST experiments.
    
    This class is carries with it a set of overdispersion parameters and a set of rules for
    determining how to assign an overdispersion parameter to a particular circuit.
    The class can batch process a full-list of circuits and produce a mapping between the 
    overdispersion parameters and the circuits.
    
    This class rather than acting as a container for the actual values of the parameters
    serves to build up a skeleton/template for the model. Once this skeleton/template is
    constructed we can create a frozen version of the object in which we can initialize
    and update the values of these parameters.
    
    I think for now it makes sense to separate out the class that generates the mappings
    and the class which stores the values of the parameters. I envision using this object
    to create a vector of parameters and the mapping between circuits and parameters and
    then when you want to optimize/initialize the parameters in the overdispersion model
    you initialize a "frozen" version of the model object and then set the values there.
    This sidesteps the issue of how you'd dynamically handle ckts you hadn't seen/mapped
    before and allow for a fixed layout to be constructed which will make things more
    performant when evaluating objective functions.
    
    The object can take as input a circuit layout, e.g. a COPALayout object and then
    use this to construct parameter layouts in the same form those used by the model objects
    that the overdispersion model is associates with so as to make the objective functions
    calculations more efficient.
    

    Parameters
    ----------
    
    mdl_rules: list of str and/or functions
        The mdl_rules is a list of rule specifications used to perform filtering
        operations on one or more circuits in order to assign a parameter to
        that circuit. The list can be a mix of either strings from a predefined list of
        filtering options or function handles. These function handles will be called on
        circuits and are expected to return a Boolean value corresponding to whether
        the circuit matched that criterion. as well as a tuple of subconditions if appropriate
        (and an empty tuple if there are no subconditions). Conditions are evaluated in the order
        in which they appear in the list. If the filtering is being performed on a list
        of ckts then we apply a condition to every ckt in that list before moving on to the next
        one. ckts are associated with the first condition/filtering they match with and so
        are removed from the list of ckts when checking subsequent conditions.
        
    ckt_list: Circuit or CircuitList
        A single circuit or a list of circuits to apply a set of mdl_rules to in order
        to construct an overdispersion model.
    
    """

    def __init__(self, mdl_rules, ckt_list):
        self.mdl_rules=mdl_rules
        self.ckt_list=ckt_list
        
        #this is an attribute which when initialized stores the matching criterion for
        #the list of circuits.
        self.ckt_matching=None
        self.unique_ckt_matching=None
        #attribute for storing a list of overdispersion parameter objects. Will be initialized later.
        self.mdl_parameters=None
    
    
    def apply_mdl_rules(self)
        """
        This method applies the list of rules specified in the user specified mdl_rules list.
        It loops through the list of circuits and applies the matching rules to each of the circuits in sequence. 
        The matching information is then stored in a list of the same length as the circuit list.
        We finally will gather up all of the like matching circuits and sort them by the ordering
        of the mdl_rules list. The circuit is assigned to the first criterion it matches.        
        """
        
        #turn the list of model rules, which may include string arguments into a list of callable
        #objects
        self.callable_mdl_rules=[]
        for rule in self.mdl_rules:
            if isinstance(rule,str):
                #if a string create a callable class object from the list of built in model rules.
                self.callable_mdl_rules.append(BuiltInOverdispersionModelConditions(rule))
            #if the rule is a callable object directly append it to the list of callable rules.
            elif callable(rule):
                self.callable_mdl_rules.append(rule)
            #if neither is true then raise a value error.
            else:
                raise ValueError('The elements of the mdl_rule list must either be a string corresponding to a built-in rule or \
                                   a callable object. Other options not supported now.')
            
        #loop through the list of ckts and then for each circuit evaluate the model rule to check for compatibility.
        #initialize a list for storing the matching info for each ckt.
        self.ckt_matching=[]
        for ckt in self.ckt_list:
            for i, rule in enumerate(self.callable_mdl_rules):
                compatibility, subcondition, name= rule(ckt)
                if compatibility
                    #I want to append a tuple with (condition,subcondition,name) for each ckt once we find a match,
                    #but for the condition I want to include the original condition from the mdl_rules list, rather than the callable
                    #versions. i.e. if a string was passed in that'll be included instead of the corresponding class.
                    self.ckt_matching.append((self.mdl_rules[i],subcondition,name))
                    #then break out of the inner loop to go to the next ckt.
                    break
                    
    #Next create a method which turns the list of ckt_matching criterion into a list of overdisperion parameters.
    def construct_parameters(self):
        #check that the ckt_matching attribute has been initialized
        if self.ckt_matching is None:
            raise ValueError('The circuits haven\'t been run against the model rules yet. Call apply_mdl_rules first.')
        #If so, then start by gathering the ckts together according to their matching criterion.
        #Start by constructing a list of the unique ckt matching criterion in ckt_matching
        #easiest way is by casting the list as a set which removes duplicates.
        self.unique_ckt_matching=list(set(self.ckt_matching))
        #For each of these unique conditions we construct an overdispersion parameter passing in a list of the ckts
        #which correspond to this mapping. Using fancy list comprehension.
        self.mdl_parameters=[]
        for unique_matching in self.unique_ckt_matching:
            self.mdl_parameters.append(OverdispersionParameter(name_label=unique_ckt_matching[2],condition=unique_ckt_matching[0], subcondition=unique_ckt_matching[1] , \
                                       ckt_list=[ckt for i,ckt in enumerate(self.ckt_list) if self.ckt_matching[i]==unique_matching] ))
        
        

        
    #define a function for adding a ckt (or CircuitList or list of Circuits) to the overdisperion model.     
    def add_ckt(self,ckts):
        #loop through the list of ckts and then for each circuit evaluate the model rule to check for compatibility.
        #initialize a list for storing the matching info for each ckt.
        if self.ckt_matching is None:
            self.ckt_matching=[]
        if self.callable_mdl_rules is None:
        #turn the list of model rules, which may include string arguments into a list of callable
        #objects
            self.callable_mdl_rules=[]
            for rule in self.mdl_rules:
                if isinstance(rule,str):
                    #if a string create a callable class object from the list of built in model rules.
                    self.callable_mdl_rules.append(BuiltInOverdispersionModelConditions(rule))
                #if the rule is a callable object directly append it to the list of callable rules.
                elif callable(rule):
                    self.callable_mdl_rules.append(rule)
                #if neither is true then raise a value error.
                else:
                    raise ValueError('The elements of the mdl_rule list must either be a string corresponding to a built-in rule or \
                                       a callable object. Other options not supported now.')
        new_ckt_matches=[]
        for ckt in ckts:
            for i, rule in enumerate(self.callable_mdl_rules):
                compatibility, subcondition, name= rule(ckt)
                if compatibility
                    #I want to append a tuple with (condition,subcondition,name) for each ckt once we find a match,
                    #but for the condition I want to include the original condition from the mdl_rules list, rather than the callable
                    #versions. i.e. if a string was passed in that'll be included instead of the corresponding class.
                    new_ckt_matches.append((self.mdl_rules[i],subcondition,name))
                    self.ckt_matching.append((self.mdl_rules[i],subcondition,name))
                    #then break out of the inner loop to go to the next ckt.
                    break
   
        #now loop through the new matching criterion splitting them into two parts, those already in the unique ckt matching set and those that are new.
        if self.unique_ckt_matching is None:
            self.unique_ckt_matching= list(set(self.ckt_matching))
        if self.mdl_parameters is None:
            self.mdl_parameters=[]
    
        for i,match in enumerate(new_ckt_matches):
            #if the matching criterion tuple is already in our set then find the corresponding parameter and add the circuit to that parameter.
            if match in self.unique_ckt_matching:
                #get the index that this match corresponds to in the unique_ckt_matching list, get the overdisperion parameter at that index and then
                #use the add_ckt method of the overdisperion parameter object.
                self.mdl_parameters[self.unique_ckt_matching.index(match)].add_ckt(ckts[i])
            #otherwise initialize a new overdisperion parameter and add this circuit to it. Append the criterion to the unique ckt_matching list
            else:
                self.mdl_parameters.append(OverdispersionParameter(name_label=match[2],condition=match[0], subcondition=match[1] , \
                                       ckt_list=[ckts[i]] ))
                self.unique_ckt_matching.append(match)
                
    
    #define a function for removing a ckt from the overdispersion model.
    #I don't have a good way to do this at the moment aside from searching through the internal lists of all of the overdisperion parameter objects, so pass on this for now.
    def remove_ckt(self):
        raise NotImplementedError('Not implemented yet. Sorry.')
    #define a function for checking if a ckt is part of the overdispersion model.
    
    #define a method for getting the number of overdisperion parameters in this model.
    def num_params(self):
        if self.mdl_parameters is None:
            num=0
        else:
            num= len(self.mdl_parameters)
        return num
        
    #define a method for setting the values of the overdisperion parameters in this model.
    #takes as input a list or vector of values and this must be the same length as the number of parameters.
    def set_params(self, values):
        if len(values)!= self.num_params():
            raise ValueError('The number of values specified should be equal to the number of parameters in the overdisperion model.')
        else:
            for i, value in enumerate(values):
                self.mdl_parameters[i].set_value(value)
                
                
    #define a version of the parameter setting function which does a specific value.
    def set_param(self, value, index):
        if self.mdl_parameters is None:
            raise ValueError('Sorry, the model parameters have not been initialized yet. No values to set.')
        elif index> len(self.mdl_parameters)-1:
            raise IndexError('Provided index is out of range.')
        else:
            self.mdl_parameters[index].set_value(value)
            
    #define analogous functions for getting the parameter values.
        #define a method for setting the values of the overdisperion parameters in this model.
    #takes as input a list or vector of values and this must be the same length as the number of parameters.
    def get_params(self):
                
                
    #define a version of the parameter setting function which does a specific value.
    def get_param(self, index):   
    
    
    #Thinking forward to the evaluation of the     
    
    
    #define a function for "freezing" an overdisperion model and preparing it for use in the objective function calculations.
    def freeze(self): 
   
           
    
    
    
    
    
    
        
class EvaluatedOverdispersionModel():
"""
    A "frozen" version of the OverdispersionModel object which is used to perform
    calculation using the overdispersion parameters and which carries
"""

class OverdispersionParameter():
"""
    A container class for storing overdispersion parameter information.
    This class contains a name_label for the parameter (e.g. the length
    of circuit it is associated with if the model is a per-depth style
    one). Also associated with this object is a list of
    ckts this parameter corresponds to. When we want to add or remove
    ckts from our experiment this list will need to be updated. This
    class will keep track of the ckts in terms of their ckt objects
    and will. I also a frozen version of this which tracks the indices
    of the ckts within the ckt list for the experiment. Not sure if I'll
    do that by making a new object or by creating a method for freezing and
    overdispersion parameter object.
    
    Parameters
    ----------
    name_label: str
        The name of this overdispersion parameter. This should be something
        which uniquely determines the condition that the ckts matched to
        as well as the value of that condition if relevant. Maybe adopt a
        convention similar to the gate labels used for constructing ckts.
        i.e. condition:(condition value(s)). This could then be parsed as
        needed to identify which condition came from where.
    ckt_list: list of Circuit objects
        A list of ckt objects which this parameter corresponds to. This can
        be set when constructing an object or set later on. It can also be
        appended to.
    index_list: list of ints
        A list of indices into a ckt list object to which the ckts in the
        internally stored ckt_list object correspond. Not set by default
        and needs to be constructed by a method which takes a ckt list
        as input. Used in the freezing process for an EvaluatedOverdispersionModel
        construction.
    value: float
        The numerical value of this overdispersion parameter, not set by default.
        This will typically be set by the objective function optimization,
        but it can be set manually too at initialization or otherwise. I might make it so that if the value is
        set when running the optimizer that the optimizer takes the value as an initial
        guess.
    is_frozen: Boolean
        if True this object contains an index_list into a ckt list.
        
    condition: str or function handle (optional)
        The parameter object should also carry with it the condition used
        to assign ckts to it. Keep it optional for now, if you try and check
        compatibility with a parameter object with no condition associated with it
        then throw an error. This can be manually set later on.
        This can be useful for later checking whether
        certain ckts are compatible with the condition for this parameter
        even if the OverdispersionModel's precedence scheme didn't select
        that condition to assign that ckt's parameter. Could also be useful
        for manually adding new ckts to a parameter object or for manually
        constructing an overdispersion model.
        
    subcondition: tuple (optional)
        An optional tuple for storing what I am calling subcondition information.
        This is optional additional categorization information that a condition function
        or string specified option can return when appropriate. For example, a function
        that applies overdisperion parameters to ckts of different depths may
        assign a subcondition value for the length of the ckt. It's a tuple since it could
        be longer than 1 element. This can be set manually later on if you like. Default value
        is the empty tuple.
    
"""

    def __init__(self, name_label=None, ckt_list=None, value=None, condition=None, subcondition=()):
    
        #initialize the name, ckt_list and value of the overdispersion parameter, if specified
        if name_label is not None:
            self.name_label= name_label
        else:
            self.name_label="TBD"
            
        self.ckt_list= ckt_list
        self.value=value
        
        #set the value of is_frozen to false.
        self.is_frozen=False
        
        #set the value of the condition and subcondition
        self.condition=condition
        #default value of the subcondition is an empty tuple
        self.subcondition=subcondition
        
        
    #define some helper functions for setting and getting values,
    #iterating through/indexing into the object and
    
    def set_value(self, value):
        self.value=value
    
    def set_ckt_list(self,ckt_list):
        #this does a wholesale replacement, to add elements use the add_ckt method
        self.ckt_list= ckt_list

    #the length function returns the length of the internal ckt_list
    def length(self):
        if self.ckt_list is not None:
            return len(self.ckt_list)
        #if the ckt_list hasn't been set then raise a ValueError
        else:
            raise ValueError('The parameter\'s ckt_list attribute hasn\'t been initialized yet.')
            
    #define a function which takes as input a list of ckts and then internally
    #sets the values of the the index_list which indexes the ckts in this object
    #into that list
    def set_ckt_indices(self, circuits):
        if self.ckt_list is None:
            raise ValueError('This overdispersion parameter\'s list of circuits is empty')
        else:
            #create the index_list
            self.index_list= []
            for ckts in self.ckt_list:
                #append onto the list the first index for which the ckt appears in the list we passed in
                #this will raise a ValueErrorif any of the ckts in our interal ckt list doesn't appear in the
                #list of ckts we passed in.
                self.index_list.append(circuits.index(ckts))
            #finish by setting the is_frozen flag to True
            self.is_frozen=True
            
    #Define a function for adding a new ckt(s) to the parameter
    #can take either a Circuit, CircuitList or simply a list of ckts as input
    #TODO: This function should check the compatibility of these ckts with the condition and subconditions for this parameter.
    def add_ckt(self, ckts):
        if self.is_frozen=True
            raise ParameterFrozenError('Yo, buddy. This overdispersion parameter has been frozen. You can\'t add anymore ckts right now.')
        else:
            #if the ckt_list attribute hasn't been initialized yet then create an empty one.
            if self.ckt_list is None:
                self.ckt_list =[]
            else:            
                #check the type of the input and append it to the end of the list.
                #I realize this isn't the best way to write this, but life is too short...
                if type(ckts) is Circuit:
                    #simply use append
                    self.ckt_list.append(ckts)
                elif (type(ckts) is CircuitList) or (type(ckts) is list):
                    #use extend to add all of the circuits in the list
                    self.ckt_list.extend(ckts)
                #if the input is none of these types then raise an exception.
                else:
                    raise ValueError('Input must be a Circuit, CircuitList or list of Circuit objects.')
    
    #Should also add a function for removing a ckt from the parameter's list. Just implemented for a single ckt right
    #now. I might add the ability to remove multiple circuits later.
    def remove_ckt(self, ckt):
        if self.is_frozen=True
            raise ParameterFrozenError('Yo, buddy. This overdispersion parameter has been frozen. You can\'t remove any ckts right now.')
        else:
            #If the ckt doesn't appear in the list this will raise a ValueError
            self.ckt_list.remove(ckt)
            
    #With the above defined method for removing a ckt throwing an error if the ckt is not in the list we should
    #also include a function for checking if a certain circuit is in the list.
    
    def is_ckt_member(self, ckt):
        #this function should return a Boolean value
        #check to make sure the ckt list has been initialized, otherwise throw a warning and return False
        if self.ckt_list is None:
            warn('The internal ckt list for this parameter has not be initialized.')
            is_it=False
        else:
            is_it= ckt in self.ckt_list
        return is_it   

    #define a function which checks whether the circuit is compatible with the condition (and subcondition) for this parameter
    def is_ckt_compatible(self,ckt):
        #if the internal ckt list isn't initialized then raise a warning and return false.
        if self.ckt_list is None:
            warn('The internal ckt list for this parameter has not been initialized.')
            is_it=False
        #Next check whether the condition attribute has been initialized. If not raise a warning and return false.
        if self.condition is None:
            warn('The matching condition for this parameter has not been initialized.')
            is_it=False
        #if this isn't an issue then run the condition check.
        else:
            #If the condition is a string then look for the internally defined function that
            #corresponds to the string:
            if isinstance(self.condition, str) :
                #I think I'll create a helper class that we can instantiate with the function corresponding to a built in string option.
                condition_checker= BuiltInOverdispersionModelConditions(self.condition)
                #then call the condition checker on the ckt
                #the condition_checker returns a tuple with the first element being the compatibility Boolean, and the second being the subcondition
                #which is the subcondition value.
                condition, subcondition= condition_checker(ckt)
                if condition and subcondition==self.subcondition:
                    compatibility=True
                else:
                    compatibility=False
            #if not a string then check that the condition is callable (ideally meaning that this is a function, but it technically could be a class)
            elif callable(self.condition)
                #call the condition function on the ckt. I am assuming that the user specified condition function returns results in the same format
                #as the built in version above. I'll need to add a specification description that let's people know about that formatting requirement.
                condition, subcondition=self.condition(ckt)
                if condition and subcondition==self.subcondition:
                    compatibility=True
                else:
                    compatibility=False
        return compatibility


#Helper class for built-in overdispersion model matching conditions.
class BuiltInOverdispersionModelConditions():
    """
    Helper class for creating matching condition functions for built-in string options for defining overdispersion parameter conditions.
    Parameters
    ----------
    
    condition_label: str (mandatory)
        The string labeling the built-in matching condition function this object instantiates
    built_in_labels: set of strs
        A set container with the built in labels supported
    """
    
    #define the built in string labels
    built_in_labels= {'global', 'per-depth'}
    
    def __init__(self, condition_label):
        #check to make sure the passed in condition_label is in the set
        #of built in labels. Otherwise throw an error.
        if condition_label not in built_in_labels:
            raise ValueError('Not a built-in matching condition label.')
        #otherwise set the internal label to the passed in one.
        else:
            self.condition_label= condition_label
    
    #Define a __call__ method which will allow us to call an instantiated object as if it was a function.
    #This will take care of the branching needed for evaluating correctly given the internal condition_label we set.
    #For now I'll just specify the behavior here, but in the future I should have separate functions defined for each
    #of the option strings.
    def __call__(self, ckt):
        if self.condition_label=='global'
            compatibility=True
            #I'm making the choice to return an empty tuple for the subcondition in this case
            subcondition=()
            #also assign a string name to be passed into the overdisperion parameter object
            name= "global"
        elif self.condition_label=='per-depth'
            compatibility=True
            subcondition= (len(ckt),)
            #When there is a non empty subcondition tuple we'll adopt a notation similar to that used in pygsti
            #for gates that take arguments and separate out the string representations of the tuple elements by
            #semicolons.
            name= "per-depth;".join(subcondition)
        return (compatibility, subcondition, name)



#error class for if you try to add ckts to a frozen overdispersion parameter
class ParameterFrozenError(Exception):
        def __init__(self, message):
            self.message = message
    
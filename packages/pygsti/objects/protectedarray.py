import numpy as _np

class ProtectedArray(_np.ndarray):
    def __new__(cls, input_array, indicesToProtect=None):
        obj = _np.asarray(input_array).view(cls)

        #Get protected indices, a specified as:
        obj.indicesToProtect = []
        if indicesToProtect is not None:
            if not (isinstance(indicesToProtect, tuple) 
                    or isinstance(indicesToProtect, list)):
                indicesToProtect = (indicesToProtect,)
            
            assert(len(indicesToProtect) <= len(obj.shape))
            for ky,L in zip(indicesToProtect,obj.shape):
                if isinstance( ky, slice ):
                    pindices = xrange(*ky.indices(L))
                elif isinstance( ky, int ):
                    i = ky+L if ky<0 else ky
                    if i < 0 or i > L: 
                        raise IndexError("index (%d) is out of range." % ky)
                    pindices = (i,)
                elif isinstance( ky, list ):
                    pindices = ky
                else: raise TypeError("Invalid index type: %s" % type(ky))
                obj.indicesToProtect.append(pindices)

        if len(obj.indicesToProtect) == 0:
            obj.indicesToProtect = None
        obj.flags.writeable = True
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return # let __new__ handle flags
        self.flags.writeable = False #default, in case getitem misses anything
        self.indicesToProtect = None #default, in case getitem misses anything


    def __getslice__(self, i, j): 
        #For special cases when getslice is still called, e.g. A[:]
        return self.__getitem__(slice(i, j)) 


    def __getitem__( self, key ) :

        writeable = True

        #check if key matches/overlaps protected region
        if self.indicesToProtect is not None:
            new_indicesToProtect = []; nUnprotectedIndices = 0
            tup_key = key if isinstance( key, tuple ) else (key,)

            while len(tup_key) < len(self.shape):
                tup_key = tup_key + (slice(None,None,None),)

            for ky,pindices,L in zip(tup_key,self.indicesToProtect,self.shape):

                #Get requested indices
                if isinstance( ky, slice ):
                    indices = xrange(*ky.indices(L))

                    new_pindices = []
                    for ii,i in enumerate(indices):
                        if i in pindices:
                            new_pindices.append( ii ) #index of i within indices
                    new_pindices = sorted(list(set(new_pindices)))
                    new_indicesToProtect.append(new_pindices)

                    #tally how many indices in this dimension are unprotected
                    nUnprotectedIndices += (len(indices) - len(new_pindices))

                elif isinstance( ky, int ):
                    i = ky+L if ky<0 else ky
                    if i > L: 
                        raise IndexError("The index (%d) is out of range." % ky)
                    if i not in pindices:
                        nUnprotectedIndices += 1 # a single unprotected index
                else: raise TypeError("Invalid index type: %s" % type(ky))
                        
            if len(new_indicesToProtect) > 0:
                new_indicesToProtect = tuple(new_indicesToProtect)
            else: new_indicesToProtect = None
            
            writeable =  bool(nUnprotectedIndices > 0)

        else: # (if nothing is protected)
            writeable = True 
            new_indicesToProtect = None

        ret = _np.ndarray.__getitem__(self,key)

        if not _np.isscalar(ret):
            if writeable:
                ret.indicesToProtect = new_indicesToProtect
                ret.flags.writeable = True
            else:
                ret.indicesToProtect = None
                ret.flags.writeable = False

        #print "   writeable = ",ret.flags.writeable
        #print "   new_toProtect = ",ret.indicesToProtect
        #print "<< END getitem"
        return ret


    def __setitem__( self, key, val ) :
        #print "In setitem with key = ", key, "val = ",val

        protectionViolation = [] # per dimension
        if self.indicesToProtect is not None:
            tup_key = key if isinstance( key, tuple ) else (key,)
            for ky,pindices,L in zip(tup_key,self.indicesToProtect,self.shape):

                #Get requested indices
                if isinstance( ky, slice ):
                    indices = xrange(*ky.indices(L))
                    if any(i in pindices for i in indices):
                        protectionViolation.append(True)
                    else: protectionViolation.append(False)

                elif isinstance( ky, int ):
                    i = ky+L if ky<0 else ky
                    if i > L: 
                        raise IndexError("The index (%d) is out of range." % ky)
                    protectionViolation.append( i in pindices )

                else: raise TypeError("Invalid index type: %s" % type(ky))
            
            if all(protectionViolation): #assigns to a protected index in each dim
                raise ValueError("**assignment destination is read-only")                        
        return _np.ndarray.__setitem__(self,key,val)

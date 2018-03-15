# Contains general matrix utilities. Some, but not all, of these tools are specific to 
# matrices over the ints modulo 2.
from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np

def dotmod2(m1,m2):

    return _np.dot(m1,m2) % 2

# A utility function used by the random symplectic matrix sampler.
def matrix_directsum(m1,m2):
    """
    Returns the direct sum of two square matrices of integers.
    """
    n1=len(m1[0,:]) 
    n2=len(m2[0,:])
    output = _np.zeros(( n1+n2 , n1+n2 ),dtype='int8') 
    output[0:n1,0:n1] = m1
    output[n1:n1+n2,n1:n1+n2] = m2 
    
    return output

def diagonal_as_vec(m):
    """
    Returns a 1D array containing the diagonal of the input square 2D array m.
    
    """
    l = _np.shape(m)[0]
    vec = _np.zeros(l,int)        
    for i in range(0,l):
        vec[i] = m[i,i]        
    return vec
    
def strictly_upper_triangle(m):
    """
    Returns a matrix containing the strictly upper triangle of m and zeros elsewhere. This
    function does not alter the input m
    
    """
    l = _np.shape(m)[0]
    out = m.copy()
        
    for i in range(0,l):
        for j in range(0,i+1):
            out[i,j] = 0
     
    return out
    
def diagonal_as_matrix(m):
    """
    Returns a diagonal matrix containing the diagonal of m. This function does not alter the 
    input m
    
    """
    l = _np.shape(m)[0]
    out = _np.zeros((l,l))
        
    for i in range(0,l):
        out[i,i] = m[i,i]
     
    return out

   
###
#
# Factorize a symmetric matrix A = F F.T over GL(n,2)


# The algorithm mostly follows the proof in

# *Orthogonal Matrices Over Finite Fields* by Jessie MacWilliams in The American Mathematical Monthly, Vol. 76, No. 2 (Feb., 1969), pp. 152-164

#
# Todo : go through the code below and find / removing anything that duplicates other parts of 
# the code
#

def random_bitstring(n, p, failcount = 0):
    """
    Constructs a random bitstring of length n with parity p
    
    """
    bitstring = _np.random.randint(0,2,size=n)
    if _np.mod(sum(bitstring),2) == p:
        return bitstring
    elif failcount < 100:
        return _np.array(random_bitstring(n,p, failcount+1),dtype='int')

def create_M(n, failcount=0):
    """
    Finds a random invertable matrix M over GL(n,2)
    
    """
    M = _np.array([random_bitstring(n,_np.random.randint(0,2)) for x in range(n)])
    if _np.isclose(_np.mod(_np.round(_np.linalg.det(M)),2),0.):
        if failcount<100:
            return create_M(n,failcount+1)
    else:
        return M

def create_D(n):
    """
    Creates a random, symmetric, invertible matrix from GL(n,2)
    
    """
    M = create_M(n)
    return _np.mod(_np.round_(_np.dot(M,M.T)),2)

def multidot(x):
    """
    Takes the dot product mod 2 of a sequence of matrices
    
    """
    return _np.mod(_np.round_(_np.linalg.multi_dot(x)),2)

def binmat(stringy):
    return _np.array(list([[int(x) for x in y] for y in stringy.split(',')]))

def onesify(A, failcount=0):
    """
    Returns M such that M A M.T has ones along the main diagonal
    
    """
    # This is probably the slowest function since it just tries things
    t = len(A)
    count = 0
    test_string = _np.diag(A)

    M = []
    while len(M) < t and count < 10:
        bitstr = random_bitstring(t, _np.random.randint(0,2))
        if multidot([bitstr, test_string]) == 1:
            if not _np.any([_np.array_equal(bitstr, m) for m in M]):
                M += [bitstr]
            else:
                count += 1
    M = _np.array(M, dtype='int')
    if _np.array_equal(multidot([M,inv_mod2(M)]), _np.eye(t)):
        return _np.array(M)
    else:
        if failcount<100:
            return onesify(A,failcount+1)

def inv_mod2(A):
    """
    
    """
    t = len(A)
    C = _np.append(A,_np.eye(t),1)
    return _np.array(rref_mod2(C)[:,t:])

def Axb_mod2(A,b):
    """
    Solves Ax = b over GF(2)
    
    """
    b = _np.array([b]).T
    C = _np.append(A,b,1)
    return _np.array([rref_mod2(C)[:,-1]]).T

def rref_mod2(A):
    """
    Gaussian elimination mod2
    
    """
    
    A = _np.array(A, dtype='int')
    m,n = A.shape
    i, j = 0, 0

    while (i < m) and (j < n):
        k = A[i:m,j].argmax() + i
        A[_np.array([i, k]),:] = A[_np.array([k, i]),:]
        aijn = _np.array([A[i,j:]])
        col = _np.array([A[:,j]]).T
        col[i] = 0
        flip = _np.dot(col,aijn)
        A[:,j:] = _np.bitwise_xor( A[:,j:], flip )
        i += 1
        j += 1
    return A

def permute_top(A,i):
    """
    Permutes the first row & col with the i'th row & col
    
    """
    t = len(A)
    P = _np.eye(t)
    P[0,0] = 0
    P[i,i] = 0
    P[0,i] = 1
    P[i,0] = 1
    return multidot([P,A,P]), P


def fix_top(A):
    """
    Takes a symmetric binary matrix with ones along the diagonal
    and returns the permutation matrix P such that the [1:t,1:t]
    submatrix of P A P is invertible

    """
    if A.shape==(1,1):
        return _np.eye(1,dtype='int')

    b_rank_deficient = True
    t = len(A)

    found_B = False
    for ind in range(t):
        aa, P = permute_top(A, ind)
        z = aa[0,1:]
        B = _np.round_(aa[1:,1:])

        if _np.isclose(_np.mod(_np.round_(_np.linalg.det(B)),2),0.):
            continue
        else:
            found_B = True
            break

    if not found_B:
        print("FAILED")
        print(A)
        return A
    return P

def proper_permutation(A):
    """
    Takes a symmetric binary matrix with ones along the diagonal
    and returns the permutation matrix P such that all [n:t,n:t]
    submatrices of P A P are invertible.
    
    """

    t = len(A)
    Ps = [] # permutation matrices
    for ind in range(t):
        perm = fix_top(A[ind:,ind:])
        zer = _np.zeros([ind, t-ind])
        full_perm = _np.array(_np.bmat([[_np.eye(ind), zer],[zer.T, perm]]))
        A = multidot([full_perm,A,full_perm.T])
        Ps += [full_perm]
#     return Ps
    return _np.linalg.multi_dot(list(reversed(Ps)))

def albert_factor(D, failcount = 0):
    """
    Return matrix M such that D = M M.T
    
    """
    D = _np.array(D, dtype='int')

    proper= False
    while not proper:
        N = onesify(D)
        aa = multidot([N,D,N.T])
        P = proper_permutation(aa)
        A = multidot([P,aa,P.T])
        proper = check_proper_permutation(A)

    t = len(A)

    # Start in lower right
    L = _np.array([[1]])

    for ind in range(t-2,-1,-1):
        block = A[ind:,ind:].copy()
        z = block[0,1:]
        B = block[1:,1:]
        # n = np.array([np.dot(np.array(Matrix(B).inv_mod(2)),z)])
        n = Axb_mod2(B, z).T
        x = _np.array(_np.dot(n,L), dtype='int')
        zer = _np.zeros([t-ind-1,1])
        L = _np.array(_np.bmat([[_np.eye(1), x],[zer, L]]), dtype='int')

    # A = P N D N.T P.T = L L.T
    # D = inv(P N) L L.T inv(N.T P.T)
    Qinv = _np.array(inv_mod2(multidot([P,N])))
    L = _np.array(multidot([_np.array(Qinv), L]),dtype='int')

    return L

def check_proper_permutation(A):
    """
    Check to see if the matrix has been properly permuted
    This should be redundent to what is already built into
    'fix_top'
    
    """
    t = len(A)
    for ind in range(0,t):
        b = A[ind:,ind:]
        if _np.isclose(_np.mod(_np.round(_np.linalg.det(b)),2),0.):
            return False
    return True

# Tim has no idea what this does.
if __name__ == '__main__':
    D = create_D(20)
    L = albert_factor(D)
    _np.allclose(D,multidot([L,L.T]))
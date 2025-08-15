from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as sparla

class RealLinOp:
    
    # Function implementations below are merely defaults.
    # Don't hesitate to override them if need be.

    __array_priority__ = 100

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self):
        return self._adjoint

    def item(self):
        # If self.size == 1, return a scalar representation of this linear operator.
        # Otherwise, error.
        raise NotImplementedError()

    def __matmul__(self, other):
        return self._linop @ other
    
    def __rmatmul__(self, other):
        return other @ self._linop


class DyadicKronStructed(RealLinOp):

    def __init__(self, A, B, adjoint=None):
        assert A.ndim == 2
        assert B.ndim == 2
        self.A = A
        self.B = B
        self._A_is_trivial = A.size == 1
        self._B_is_trivial = B.size == 1
        self._shape = ( A.shape[0]*B.shape[0], A.shape[1]*B.shape[1] )
        self._size = self.shape[0] * self.shape[1]
        self._fwd_matvec_core_shape = (B.shape[1], A.shape[1])
        self._adj_matvec_core_shape = (B.shape[0], A.shape[0])
        self._dtype = A.dtype
        self._linop =  sparla.LinearOperator(dtype=self.dtype, shape=self.shape, matvec=self.matvec, rmatvec=self.rmatvec)
        self._adjoint = DyadicKronStructed(A.T, B.T, adjoint=self) if adjoint is None else adjoint

    def item(self):
        # This will raise a ValueError if self.size > 1.
        return self.A.item() * self.B.item()
    
    def matvec(self, other):
        inshape = other.shape
        assert other.size == self.shape[1]
        if self._A_is_trivial:
            return self.A.item() * (self.B @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A @ other)
        out = self.B @ np.reshape(other, self._fwd_matvec_core_shape, order='F') @ self.A.T
        out = np.reshape(out, inshape, order='F')
        return out

    def rmatvec(self, other):
        inshape = other.shape
        assert other.size == self.shape[0]
        if self._A_is_trivial:
            return self.A.item() * (self.B.T @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A.T @ other)
        out = self.B.T @ np.reshape(other, self._adj_matvec_core_shape, order='F') @ self.A
        out = np.reshape(out, inshape, order='F')
        return out
    
    @staticmethod
    def build_polyadic(kron_operands) -> DyadicKronStructed:
        if len(kron_operands) == 2:
            out = DyadicKronStructed(kron_operands[0], kron_operands[1])
            return out
        # else, recurse
        arg = DyadicKronStructed.build_polyadic(kron_operands[1:])
        out = DyadicKronStructed(kron_operands[0], arg)
        return out


class KronStructured(RealLinOp):

    def __init__(self, kron_operands):
        self.kron_operands = kron_operands
        # assert all([op.ndim == 2 for op in kron_operands])
        self.shapes = np.array([op.shape for op in kron_operands])
        self._shape = tuple(int(i) for i in np.prod(self.shapes, axis=0))
        self.dyadic_struct = DyadicKronStructed.build_polyadic(self.kron_operands)
        self._linop   = self.dyadic_struct._linop
        self._adjoint = self.dyadic_struct.T
        self._dtype = self.kron_operands[0].dtype

    def to_full_array(self) -> np.ndarray:
        """
        Return the full dense matrix. Do not use this method in a performance sensitive routine
        as you will not be utilizing the structure of the matrix to its full
        potential. This is mainly used as a debugging tool.
        """
        output = 1
        for i in range(len(self.kron_operands)):
            output = np.kron(output, self.kron_operands[i])
        return output
    
    def update_operand(self, lane: int, matrix: np.ndarray) -> None:
        """
        Replace a specific matrix with a new matrix of the same size and layout.
        """

        assert lane >= 0 and lane < len(self.kron_operands)

        # Iterate through the structure and replace the A matrix of the appropriate index with the new matrix.

        def walk_forward(curr_loc: int, dydadic_struct: DyadicKronStructed, mat: np.ndarray):
            if curr_loc == lane:
                dydadic_struct.A = mat
                breakpoint()
                return
            elif lane == len(self.kron_operands) - 1 and (curr_loc == lane -1):
                # We have hit the leaf node.
                dydadic_struct.B = mat
                return
            return walk_forward(curr_loc + 1, dydadic_struct.B, mat) 
        
        walk_forward(0, self.dyadic_struct, matrix)
        walk_forward(0, self._adjoint, matrix.T)



class KronStructuredPath:
    """
    This class is for making a path for computing a matvec and adjoint of a kron structured matrix.

    When those functions are called you need to specify the actual dense operands being used.
    This way we can just swap out one the matrices each time we update the representation of a gate.
    """


    def __init__(self, shapes: list[tuple[int, int]]):


        pass
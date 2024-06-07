from pygsti.extras.errorgenpropagation.propagatableerrorgen import  propagatableerrorgen
from numpy import complex128

class errordict(dict):

    def __setitem__(self, __key: any, __value: any) -> None:
        if __key in self :
            super().__setitem__(__key,self[__key]+__value)
        else:
            super().__setitem__(__key,__value)
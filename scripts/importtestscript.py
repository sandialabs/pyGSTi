import pygsti
print(pygsti.__path__)
print(pygsti.__version__)

try:
    _importlib.import_module(pygsti.evotypes.densitymx.statereps)
except ModuleNotFoundError:
    print('importing pygsti.evotypes.densitymx.statereps failed using import lib. ModuleNotFoundError raised.') 

try:
    import pygsti.evotypes.densitymx.statereps
except ModuleNotFoundError:
    print('importing pygsti.evotypes.densitymx.statereps failed. ModuleNotFoundError raised.') 
    

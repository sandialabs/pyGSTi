import os

def tool(function):
    def wrapper(*args, **kwargs):
        owd = os.getcwd() # Handle moving between directories
        os.chdir('..')
        function(*args, **kwargs)
        os.chdir(owd)
    return wrapper

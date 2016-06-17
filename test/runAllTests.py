from test_tools.runModule  import run_module
from test_tools.message    import show_message
import os

if __name__ == "__main__":

    show_message('BEGINNING TESTS')
    _, directories, _ = os.walk(os.getcwd()).next()

    for directory in directories:
        run_module(directory)

    show_message('ENDING TESTS')

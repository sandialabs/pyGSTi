#!/usr/bin/env python
from helpers.pylint           import get_score
from helpers.automation_tools import read_yaml, write_yaml, get_args
import sys

def check_score():
    yamlFile = 'pylint_config.yml'

    config       = read_yaml(yamlFile)
    desiredScore = config['desired-score']
    print('Score should be: %s' % desiredScore)
    score        = get_score()
    print('Score was: %s' % score)
    
    if float(score) >= float(desiredScore):
        config['desired-score'] = score # Update the score if it is higher than the last one
        write_yaml(config, yamlFile)
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv) 
    # No arguments specified
    if len(args[0]) == 0 and len(kwargs) == 0:
        check_score() 
    else:
        print(args, kwargs)
    '''
    if 'kwarg' in kwargs:
        action()
    '''

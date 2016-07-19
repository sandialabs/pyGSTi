#!/usr/bin/env python
import sys
sys.path.append('..')
from lintAll            import lint_all
from automation_tools import read_yaml, write_yaml


def get_score():
    lintResult = lint_all()
    scoreLine  = lintResult[-2]
    # Score is located between leftmost slash and then rightmost space:
    # -> Your code has been rated at 9.57/10 (previous run: 9.57/10, +0.00) 
    #                                    ^
    # -> Your code has been rated at 9.57
    #                               ^
    # -> 9.57
    score      = scoreLine.split('/', 1)[0].rsplit(' ', 1)[1]
    return score
    

if __name__ == "__main__":
    yamlFile = 'config.yml'

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
    
    

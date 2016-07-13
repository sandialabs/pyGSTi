#!/usr/bin/env python
import sys
from lintAll import lint_all
from readyaml import read_yaml

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

    config       = read_yaml('config.yml')
    desiredScore = config['desired-score']
    print('Score should be: %s' % desiredScore)
    score        = get_score()
    print('Score was: %s' % score)
    
    if float(score) >= float(desiredScore):
        sys.exit(0)
    else:
        sys.exit(1)
    
    

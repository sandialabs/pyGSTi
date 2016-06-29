import subprocess

if __name__ == "__main__":
    with open('refactors.txt', 'r') as refactors:
        for line in refactors:
            commands = ['pylint', '--disable=C,W,R,E,F', 
                                  '--enable=' + line.replace('\n', ''), 
                                  '--reports=n', 
                                  '../../packages/pygsti']
            subprocess.call(commands)

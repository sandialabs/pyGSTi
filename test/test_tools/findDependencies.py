from __future__ import print_function
from tool       import tool
import os
import sys

def increase_count(dictionary, item):
    if item in dictionary:
        dictionary[item] = dictionary[item] + 1
    else:
        dictionary[item] = 1

def build_qualified_imports(line, qualifiedImports):
    tokenized = [t for t in line.split(' ') if (t != '' and t != ',')]
    if tokenized[0] == 'import' and \
       len(tokenized) > 2       and \
       tokenized[2] == 'as':
	    qualifiedImports.append((tokenized[3], tokenized[1]))
    elif tokenized[0] == 'from':
	package = tokenized[1]
	if len(tokenized) < 5:
	    qualifiedImports.append((tokenized[3], package + '.' + tokenized[3]))
	elif len(tokenized) > 4:
	    if tokenized[4] == 'as':
                qualifiedImports.append((tokenized[5], package + '.' + tokenized[3]))
            else:
                for token in tokenized[2:]:
                    token = token.replace(',', '')
                    qualifiedImports.append((token, package + '.' + token))

@tool
def find_dependencies(filename, packageName='pygsti'):
    dependencies = dict()
    qualifiedImports = []
    with open(filename, 'r') as source:
        for line in source.read().splitlines():
            # Replace qualified imports in the line
            for qImport in qualifiedImports:
                if qImport[0] in line:
                    line = line.replace(qImport[0], qImport[1])

            # Add qualified imports to the list, in the form (qualifier, actual)
            if 'import' in line:
                build_qualified_imports(line, qualifiedImports)

            elif packageName in line:
                 occurances = line.split(packageName)
                 for occurance in occurances[1:]:
                     module = ''
                     for char in occurance[1:]:
                         if char not in '. ()':
                             module += char
                         else:
                             break
                     increase_count(dependencies, module)
    return dependencies


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print('Dependencies for %s:' % arg)
        deps = find_dependencies(arg)
        print(deps)
        if len(deps) > 0:
            print('Most Imported: %s' % max(deps, key=deps.get))


'''
if __name__ == "__main__":
    dependencyList = []
    for subdir, dirs, files in os.walk(os.getcwd()):
	for filename in files:
	    filepath = subdir + os.sep + filename
	    if filepath.endswith('.py'):
                dependencyList.append((filepath, find_dependencies(filepath)))

    for item in dependencyList:
        print('Dependencies for %s:' % item[0])
        print(item[1])
        if len(item[1]) > 0:
            print('Most Imported: ', max(item[1], key=item[1].get))
        print('')

   '''

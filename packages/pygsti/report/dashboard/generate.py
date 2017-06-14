#!/usr/bin/env python3
import webbrowser

def readfile(filename):
    with open(filename, 'r') as infile:
        return infile.read()

def main():
    templatefile = 'template.html'
    outputfile   = 'output.html'
    showfile     = 'show.js'
    bodyfile     = 'body.html'

    template = readfile(templatefile)
    showscript = readfile(showfile)

    body =readfile(bodyfile)

    dashboard = template.format(top='TOP', abody=body, bbody='B', aleft='a-options', bleft='b-options', showscript=showscript)

    with open(outputfile, 'w') as outfile:
        outfile.write(dashboard)
    webbrowser.open(outputfile)

if __name__ == '__main__':
    main()

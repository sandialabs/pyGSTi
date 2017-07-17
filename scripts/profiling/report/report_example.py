import pickle
import pygsti

def main():
    with open('2qbit_results.pkl', 'rb') as infile:
        results = pickle.load(infile)
    pygsti.report.create_general_report(results, "tutorial_files/exampleGenReport.html",
                                        verbosity=0, auto_open=False)

main()

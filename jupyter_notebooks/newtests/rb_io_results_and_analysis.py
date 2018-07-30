from pygsti.extras import rb

def test_rb_io_results_and_analysis():

    # Just checks that we can succesfully import the standard data type.
    data = rb.io.import_rb_summary_data(['testfiles/rb_io_test.txt',])
    # This is a basic test that the imported dataset makes sense : we can
    # successfully run the analysis on it.
    out = rb.analysis.std_practice_analysis(data,bootstrap_samples=100)
    # Checks plotting works. This requires matplotlib, so should do a try/except
    out.plot()
    
    return
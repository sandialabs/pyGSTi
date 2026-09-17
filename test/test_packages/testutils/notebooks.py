
def run_notebook(notebook_path):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python')
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        return e

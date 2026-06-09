"""Tiny helper to assemble and execute tutorial notebooks with nbformat."""
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def md(text): return nbf.v4.new_markdown_cell(text)
def code(src): return nbf.v4.new_code_cell(src)

def build(path, cells, *, execute=True, timeout=1800, exec_kernel="sbtut"):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    # Stored kernelspec is generic so the notebook runs on any user's machine.
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    if execute:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name=exec_kernel)
        ep.preprocess(nb, {"metadata": {"path": str(Path(path).parent)}})
    nbf.write(nb, path)
    return path

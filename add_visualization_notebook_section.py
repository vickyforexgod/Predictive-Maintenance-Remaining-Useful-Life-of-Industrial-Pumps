import ast
from pathlib import Path

import nbformat


notebook_path = Path("pump_datascience_process.ipynb")
script_path = Path("pump_datascience_process.py")

notebook = nbformat.read(notebook_path, as_version=4)

if notebook.cells and notebook.cells[0].cell_type == "markdown":
    notebook.cells[0].source = (
        "# Pump Predictive Maintenance Data Science Process\n\n"
        "Charter-aligned notebook with data inspection, cleaning, feature engineering, "
        "visualization, modelling, alert evaluation, and running-fleet predictions."
    )

setup = notebook.cells[1].source
for import_line in [
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
]:
    if import_line not in setup:
        setup = setup.replace("import pandas as pd\n", "import pandas as pd\n" + import_line + "\n")
notebook.cells[1].source = setup

script = script_path.read_text(encoding="utf-8")
tree = ast.parse(script)
lines = script.splitlines(keepends=True)
visualization_function = next(
    "".join(lines[node.lineno - 1:node.end_lineno])
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "create_visualizations"
)

visualization_heading = nbformat.v4.new_markdown_cell(
    "## Visualization\n\n"
    "The charts below are generated from cleaned sensor data, maintenance records, "
    "RUL predictions, failure-mode predictions, and model comparison results."
)
visualization_code = nbformat.v4.new_code_cell(visualization_function)
run_visualizations = nbformat.v4.new_code_cell(
    'PLOTS_DIR = OUTPUT_DIR / "plots"\n'
    'PLOTS_DIR.mkdir(parents=True, exist_ok=True)\n'
    'print("Visualization output folder:", PLOTS_DIR)'
)

insert_at = next(
    (
        index + 1
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown"
        and "Modelling and Alert Evaluation" in cell.source
    ),
    len(notebook.cells),
)
notebook.cells[insert_at:insert_at] = [
    visualization_heading,
    visualization_code,
    run_visualizations,
]

nbformat.write(notebook, notebook_path)
nbformat.validate(notebook)
print(f"Added visualization section to {notebook_path}")

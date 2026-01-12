
import sys
import os

sys.path.append(os.getcwd())

from dash import html, dcc, dash_table
from deriv_quant_py.dashboard.app import app, backtest_layout

print("Verifying Backtest Layout...")
layout = backtest_layout()

def traverse(component, level=0):
    indent = "  " * level
    comp_type = type(component).__name__
    comp_id = getattr(component, 'id', 'No ID')
    print(f"{indent}{comp_type} (id={comp_id})")

    if hasattr(component, 'children'):
        children = component.children
        if children is None:
            return

        if not isinstance(children, list):
            children = [children]

        for child in children:
            traverse(child, level + 1)

print("Layout Structure:")
traverse(layout)

# Helper to find components by ID
def find_component(component, id_to_find):
    if hasattr(component, 'id') and component.id == id_to_find:
        return component
    if hasattr(component, 'children'):
        children = component.children
        if children is None:
            return None
        if not isinstance(children, list):
            children = [children]

        for child in children:
            found = find_component(child, id_to_find)
            if found:
                return found
    return None

chart = find_component(layout, "bt-chart")
table_container = find_component(layout, "bt-table-container")

print("\nSearch Results:")
if chart:
    print(f"SUCCESS: Found bt-chart")
else:
    print("FAILURE: bt-chart not found")

if table_container:
    print(f"SUCCESS: Found bt-table-container")
else:
    print("FAILURE: bt-table-container not found")

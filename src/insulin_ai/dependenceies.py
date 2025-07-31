from modulegraph2 import ModuleGraph
import sys
import os

script_path = "app.py"
script_dir = os.path.dirname(os.path.abspath(script_path))

# Add script's directory to sys.path
sys.path.insert(0, script_dir)

mg = ModuleGraph()
mg.add_script(script_path)
mg.run()

for node in sorted(mg.nodes(), key=lambda x: x.identifier):
    print(node.identifier)

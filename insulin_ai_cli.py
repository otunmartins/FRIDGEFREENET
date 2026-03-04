#!/usr/bin/env python3
"""
Insulin AI CLI launcher.

Run from project root. Sets up paths and invokes the CLI.
Usage: python insulin_ai_cli.py discover --iterations 2
       python insulin_ai_cli.py evaluate "[*]OCC[*]"
"""

import sys
import os

# Ensure src/python and project root are on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

# Invoke CLI
from insulin_ai.cli import main
sys.exit(main())

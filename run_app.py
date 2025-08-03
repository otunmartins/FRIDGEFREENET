#!/usr/bin/env python3
"""
Development script for running the Insulin AI Streamlit app.

This script properly installs the package and runs the Streamlit app.
Use this instead of `streamlit run src/insulin_ai/app.py` to avoid import issues.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Install the package and run the Streamlit app."""
    
    print("🚀 Starting Insulin AI App...")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Step 1: Install the package in development mode
    print("📦 Installing package in development mode...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True, capture_output=True, text=True)
        print("✅ Package installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    
    # Step 2: Run the Streamlit app
    print("🌐 Starting Streamlit app...")
    try:
        # Use the entry point we defined
        app_file = project_root / "src" / "insulin_ai" / "main.py"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file)
        ] + sys.argv[1:])  # Pass through any additional arguments
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
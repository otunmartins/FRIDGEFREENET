#!/usr/bin/env python3
"""
Cleanup Script for Insulin-AI Project

This script safely removes unnecessary files and directories that clutter
the project, including:
- Test directories and files
- Temporary simulation outputs
- Generated molecular files
- Preprocessed directories
- Backup files

Usage:
    python cleanup_clutter.py [--dry-run] [--aggressive]
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
from typing import List, Set
import sys

# Files and directories to remove (safe to delete)
CLUTTER_PATTERNS = [
    # Test directories
    "test_*",
    "*_test",
    "*_temp", 
    "temp_*",
    
    # Simulation outputs
    "automated_simulations/",
    "simple_md_simulations/",
    "enhanced_md_simulations/",
    "postprocessing_results/",
    
    # Preprocessed directories
    "preprocessed_insulin_*",
    "enhanced_polymer_*",
    
    # Amorphous test directories
    "*_amorphous*",
    "fast_amorphous/",
    
    # Specific test directories
    "test_error_handling/",
    "test_fixed_psp_*/",
    "test_psp_*/",
    "test_asterisk*/",
    "test_simple*/",
    "test_robust_fallback/",
    "test_fallback_pipeline/",
    "test_packmol_*/",
    "test_efficient_pipeline/",
    "test_final_*/",
    "test_stop_*/",
    "test_complete_*/",
    "test_interface*/",
    "test_candidates_*/",
    "test_sim_info_*/",
    "test_file_discovery/",
    "work_dir/",
    
    # Molecular files
    "*.pdb",
    "*.dcd", 
    "*.log",
    "trajectory*.pdb",
    "simulation_log*.txt",
    "polymer_*.pdb",
    "insulin_*.pdb",
    "*.inp",
    
    # Output files
    "output_*.csv",
    "*.xyz",
    
    # Backup files
    "*.bak",
    "*.backup",
    "*~",
    
    # Python cache
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    
    # IDE files
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
]

# Test files to remove (Python files that are clearly test files)
TEST_PYTHON_FILES = [
    "test_*.py",
    "*_test.py",
    "verify_*.py",
    "mmgbsa_test.py",
    "openmm_test*.py", 
    "test.py",
    "dependenceies.py",  # Typo in original filename
]

# Files to keep (important files that shouldn't be deleted)
KEEP_FILES = {
    "README.md",
    "README_*.md", 
    "LICENSE",
    "pyproject.toml",
    "MANIFEST.in",
    "INSTALLATION.md",
    ".gitignore",
    ".gitmodules",
    "requirements.txt",
    "setup.py",
    "app.py",  # Original main app
    "cleanup_clutter.py",  # This script
}

# Directories to keep (core project directories)
KEEP_DIRS = {
    "src/",
    "core/",
    "integration/", 
    "utils/",
    "app/",
    "docs/",
    "requirements/",
    ".git/",
    "PSP/",  # Keep PSP submodule
}

def should_keep_file(file_path: Path) -> bool:
    """Check if a file should be kept (not deleted)"""
    filename = file_path.name
    
    # Check if it matches any keep patterns
    for pattern in KEEP_FILES:
        if pattern.endswith("*"):
            if filename.startswith(pattern[:-1]):
                return True
        elif filename == pattern:
            return True
    
    return False

def should_keep_dir(dir_path: Path) -> bool:
    """Check if a directory should be kept (not deleted)"""
    dir_name = dir_path.name + "/"
    
    # Check if it matches any keep patterns
    for keep_dir in KEEP_DIRS:
        if dir_name == keep_dir or str(dir_path).startswith(keep_dir):
            return True
    
    return False

def find_clutter(root_dir: Path, aggressive: bool = False) -> tuple[List[Path], List[Path]]:
    """Find files and directories that should be cleaned up"""
    files_to_remove = []
    dirs_to_remove = []
    
    print(f"🔍 Scanning for clutter in: {root_dir}")
    
    # Find files matching clutter patterns
    for pattern in CLUTTER_PATTERNS:
        matches = list(root_dir.glob(pattern))
        for match in matches:
            if match.is_file() and not should_keep_file(match):
                files_to_remove.append(match)
            elif match.is_dir() and not should_keep_dir(match):
                dirs_to_remove.append(match)
    
    # Find test Python files
    for pattern in TEST_PYTHON_FILES:
        matches = list(root_dir.glob(pattern))
        for match in matches:
            if match.is_file() and not should_keep_file(match):
                files_to_remove.append(match)
    
    # In aggressive mode, also look for additional patterns
    if aggressive:
        print("🔥 Aggressive mode: Looking for additional clutter...")
        
        # Look for any directory with "test" in the name
        for item in root_dir.iterdir():
            if item.is_dir() and "test" in item.name.lower() and not should_keep_dir(item):
                dirs_to_remove.append(item)
        
        # Look for large files that might be simulation outputs
        for item in root_dir.rglob("*"):
            if item.is_file() and item.stat().st_size > 10_000_000:  # > 10MB
                if any(ext in item.suffix for ext in ['.pdb', '.dcd', '.xtc', '.trr']):
                    files_to_remove.append(item)
    
    # Remove duplicates and sort
    files_to_remove = sorted(list(set(files_to_remove)))
    dirs_to_remove = sorted(list(set(dirs_to_remove)))
    
    return files_to_remove, dirs_to_remove

def calculate_size(paths: List[Path]) -> int:
    """Calculate total size of files/directories"""
    total_size = 0
    for path in paths:
        if path.exists():
            if path.is_file():
                total_size += path.stat().st_size
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        try:
                            total_size += file_path.stat().st_size
                        except (OSError, PermissionError):
                            pass
    return total_size

def format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def preview_cleanup(files_to_remove: List[Path], dirs_to_remove: List[Path]):
    """Show what would be cleaned up"""
    print("\n📋 CLEANUP PREVIEW")
    print("=" * 50)
    
    if files_to_remove:
        print(f"\n🗃️  Files to remove ({len(files_to_remove)}):")
        for file_path in files_to_remove[:20]:  # Show first 20
            size = format_size(file_path.stat().st_size) if file_path.exists() else "0 B"
            print(f"   📄 {file_path} ({size})")
        if len(files_to_remove) > 20:
            print(f"   ... and {len(files_to_remove) - 20} more files")
    
    if dirs_to_remove:
        print(f"\n📁 Directories to remove ({len(dirs_to_remove)}):")
        for dir_path in dirs_to_remove[:20]:  # Show first 20
            print(f"   📂 {dir_path}/")
        if len(dirs_to_remove) > 20:
            print(f"   ... and {len(dirs_to_remove) - 20} more directories")
    
    # Calculate total size
    all_paths = files_to_remove + dirs_to_remove
    total_size = calculate_size(all_paths)
    print(f"\n💾 Total size to be freed: {format_size(total_size)}")
    
    if not files_to_remove and not dirs_to_remove:
        print("✨ No clutter found! Directory is already clean.")

def perform_cleanup(files_to_remove: List[Path], dirs_to_remove: List[Path], dry_run: bool = False):
    """Actually perform the cleanup"""
    if dry_run:
        print("\n🔍 DRY RUN - No files will be deleted")
        preview_cleanup(files_to_remove, dirs_to_remove)
        return
    
    print("\n🧹 PERFORMING CLEANUP")
    print("=" * 50)
    
    errors = []
    removed_count = 0
    
    # Remove files
    for file_path in files_to_remove:
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"✅ Removed file: {file_path}")
                removed_count += 1
        except Exception as e:
            error_msg = f"❌ Failed to remove file {file_path}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    # Remove directories
    for dir_path in dirs_to_remove:
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"✅ Removed directory: {dir_path}/")
                removed_count += 1
        except Exception as e:
            error_msg = f"❌ Failed to remove directory {dir_path}: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    print(f"\n🎉 Cleanup complete! Removed {removed_count} items.")
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors occurred:")
        for error in errors:
            print(f"   {error}")

def main():
    parser = argparse.ArgumentParser(description="Clean up clutter from Insulin-AI project")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be cleaned without actually deleting")
    parser.add_argument("--aggressive", action="store_true",
                       help="More aggressive cleanup (removes additional files)")
    parser.add_argument("--root", type=str, default=".",
                       help="Root directory to clean (default: current directory)")
    
    args = parser.parse_args()
    
    root_dir = Path(args.root).resolve()
    
    if not root_dir.exists():
        print(f"❌ Directory does not exist: {root_dir}")
        sys.exit(1)
    
    print("🧬 Insulin-AI Project Cleanup Tool")
    print("=" * 50)
    print(f"📁 Root directory: {root_dir}")
    print(f"🔍 Dry run: {'Yes' if args.dry_run else 'No'}")
    print(f"🔥 Aggressive mode: {'Yes' if args.aggressive else 'No'}")
    
    # Find clutter
    files_to_remove, dirs_to_remove = find_clutter(root_dir, args.aggressive)
    
    if not files_to_remove and not dirs_to_remove:
        print("\n✨ No clutter found! Project directory is clean.")
        return
    
    # Show preview
    preview_cleanup(files_to_remove, dirs_to_remove)
    
    if args.dry_run:
        print("\n💡 Run without --dry-run to actually perform the cleanup.")
        return
    
    # Confirm before cleanup
    print(f"\n⚠️  WARNING: This will permanently delete {len(files_to_remove)} files and {len(dirs_to_remove)} directories!")
    response = input("Do you want to continue? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        perform_cleanup(files_to_remove, dirs_to_remove, dry_run=False)
    else:
        print("❌ Cleanup cancelled.")

if __name__ == "__main__":
    main() 
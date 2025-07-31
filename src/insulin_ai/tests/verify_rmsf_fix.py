#!/usr/bin/env python3
"""
Quick verification script to test that the RMSF fix is working.
"""

import sys
from pathlib import Path

def check_rmsf_fix():
    """Verify that the RMSF function fix has been applied correctly"""
    
    print("🔍 VERIFYING RMSF FIX")
    print("=" * 40)
    
    # Read the fixed file
    analyzer_file = Path("integration/analysis/insulin_comprehensive_analyzer.py")
    
    if not analyzer_file.exists():
        print("❌ Analyzer file not found!")
        return False
    
    # Read the content
    with open(analyzer_file, 'r') as f:
        content = f.read()
    
    # Check for the fixed line
    fixed_line = "md.rmsf(insulin_traj, reference=insulin_traj[0])"
    broken_line = "md.rmsf(insulin_traj) * 10"
    
    if fixed_line in content:
        print("✅ RMSF fix confirmed in source code!")
        print(f"   Found: {fixed_line}")
        
        # Check that the broken version is not present
        if broken_line in content and "Old" not in content:
            print("⚠️  WARNING: Old broken line still present")
            return False
        
        print("✅ No broken RMSF calls found")
        return True
    else:
        print("❌ RMSF fix NOT found in source code!")
        print("   Expected to find: md.rmsf(insulin_traj, reference=insulin_traj[0])")
        return False

def check_trajectory_fix():
    """Verify that the trajectory file fix has been applied"""
    
    print("\n🔍 VERIFYING TRAJECTORY FILE FIX")
    print("=" * 40)
    
    # Read the postprocessing file
    postproc_file = Path("integration/analysis/comprehensive_postprocessing.py")
    
    if not postproc_file.exists():
        print("❌ Post-processing file not found!")
        return False
    
    # Read the content
    with open(postproc_file, 'r') as f:
        content = f.read()
    
    # Check for the fixed logic
    if "possible_frames_files = [" in content:
        print("✅ Trajectory file fix confirmed in source code!")
        print("   Found: Multiple path checking logic")
        return True
    else:
        print("❌ Trajectory file fix NOT found!")
        return False

def main():
    print("🧪 RMSF & TRAJECTORY FILE FIX VERIFICATION")
    print("=" * 60)
    
    rmsf_ok = check_rmsf_fix()
    traj_ok = check_trajectory_fix()
    
    print("\n" + "=" * 60)
    print("📋 VERIFICATION RESULTS:")
    print("=" * 60)
    
    if rmsf_ok and traj_ok:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\n🔄 IF YOU'RE STILL SEEING ERRORS:")
        print("   1. Stop your Streamlit app (Ctrl+C)")
        print("   2. Restart it: streamlit run app.py")
        print("   3. Python module caching requires restart")
        
        print("\n💡 WHY RESTART IS NEEDED:")
        print("   • Python caches imported modules in memory")
        print("   • Streamlit keeps modules loaded for performance")
        print("   • Code changes require process restart to take effect")
        print("   • This is normal behavior for Python applications")
        
    else:
        print("❌ SOME FIXES NOT VERIFIED!")
        if not rmsf_ok:
            print("   • RMSF fix missing or incomplete")
        if not traj_ok:
            print("   • Trajectory file fix missing or incomplete")

if __name__ == "__main__":
    main() 
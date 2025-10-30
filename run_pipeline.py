#!/usr/bin/env python3
"""
Political Discourse Analysis - Full Pipeline
Author: 3DL - Data Driven Decision Lab
Website: https://datadrivendecisionlab.com

Run the complete analysis pipeline:
1. Process speeches into paragraphs
2. Assess hate/violence scores

Usage:
    python run_pipeline.py [--skip-processing]
    
Options:
    --skip-processing    Skip speech processing, only run assessment
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Verify required files and dependencies exist."""
    print("Checking dependencies...")
    
    # Check for .env file
    if not Path(".env").exists():
        print("❌ ERROR: .env file not found")
        print("Please create a .env file with your OPENAI_API_KEY")
        return False
    
    # Check for source files
    source_dir = Path("sources/state_union")
    if not source_dir.exists() or not list(source_dir.glob("*.txt")):
        print("❌ ERROR: No source files found in sources/state_union/")
        return False
    
    print("✅ Dependencies verified")
    return True

def run_processing():
    """Run the speech processing step."""
    print("\n" + "="*60)
    print("STEP 1: Processing speeches into paragraphs")
    print("="*60 + "\n")
    
    result = subprocess.run(
        [sys.executable, "process_speeches.py"],
        cwd=os.getcwd()
    )
    
    if result.returncode != 0:
        print("\n❌ Processing failed!")
        return False
    
    print("\n✅ Processing completed successfully")
    return True

def run_assessment():
    """Run the hate/violence assessment step."""
    print("\n" + "="*60)
    print("STEP 2: Assessing hate/violence scores")
    print("="*60 + "\n")
    
    # Check if input file exists
    input_file = Path("outputs/state_union_paragraphs.csv")
    if not input_file.exists():
        print("❌ ERROR: Input file not found")
        print("Please run speech processing first (without --skip-processing)")
        return False
    
    result = subprocess.run(
        [sys.executable, "assess_speeches.py"],
        cwd=os.getcwd()
    )
    
    if result.returncode != 0:
        print("\n❌ Assessment failed!")
        return False
    
    print("\n✅ Assessment completed successfully")
    return True

def main():
    """Run the complete pipeline."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        Political Discourse Analysis Pipeline                ║
║        3DL - Data Driven Decision Lab                       ║
║        https://datadrivendecisionlab.com                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    skip_processing = "--skip-processing" in sys.argv
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 1: Process speeches (unless skipped)
    if not skip_processing:
        if not run_processing():
            sys.exit(1)
    else:
        print("\n⏭️  Skipping speech processing (--skip-processing flag)")
    
    # Step 2: Run assessment
    if not run_assessment():
        sys.exit(1)
    
    # Success!
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput files:")
    print("  📄 outputs/state_union_paragraphs.csv")
    print("  📊 outputs/state_union_with_assessment.csv")
    print("\nNext steps:")
    print("  - Analyze results in outputs/state_union_with_assessment.csv")
    print("  - Visualize score distributions by president/party/year")
    print("  - Conduct statistical analysis of temporal trends")
    print()

if __name__ == "__main__":
    main()

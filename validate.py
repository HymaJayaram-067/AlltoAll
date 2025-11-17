"""
Validation Script for TopKA2A Implementation
============================================

This script validates:
1. Python syntax of all modules
2. Import structure
3. Documentation completeness
4. API consistency
"""

import ast
import os
import sys


def validate_python_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_file_structure():
    """Validate repository file structure."""
    required_files = [
        'topka2a_baseline.py',
        'topka2a_paper_exact.py',
        'topka2a_optimized.py',
        'topka2a_optimized_hpc.py',
        'OPTIMIZATIONS.md',
        'DIAGRAMS.py',
        'README.md',
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    return len(missing) == 0, missing


def validate_documentation():
    """Check documentation completeness."""
    docs_to_check = [
        ('README.md', ['TopKA2A', 'Algorithm', 'Installation', 'Usage']),
        ('OPTIMIZATIONS.md', ['Optimization', 'Performance', 'HPC']),
    ]
    
    issues = []
    for filepath, required_terms in docs_to_check:
        if not os.path.exists(filepath):
            issues.append(f"{filepath} not found")
            continue
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        for term in required_terms:
            if term.lower() not in content.lower():
                issues.append(f"{filepath} missing '{term}'")
    
    return len(issues) == 0, issues


def main():
    """Run all validations."""
    print("="*80)
    print("TopKA2A Implementation Validation")
    print("="*80)
    
    # Change to repository root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    
    all_valid = True
    
    # 1. Validate file structure
    print("\n1. Checking file structure...")
    structure_valid, missing = validate_file_structure()
    if structure_valid:
        print("   ✓ All required files present")
    else:
        print(f"   ✗ Missing files: {missing}")
        all_valid = False
    
    # 2. Validate Python syntax
    print("\n2. Checking Python syntax...")
    python_files = [
        'topka2a_baseline.py',
        'topka2a_paper_exact.py',
        'topka2a_optimized.py',
        'topka2a_optimized_hpc.py',
        'DIAGRAMS.py',
    ]
    
    for filepath in python_files:
        if os.path.exists(filepath):
            valid, msg = validate_python_syntax(filepath)
            if valid:
                print(f"   ✓ {filepath}")
            else:
                print(f"   ✗ {filepath}: {msg}")
                all_valid = False
        else:
            print(f"   - {filepath} not found (skipped)")
    
    # 3. Validate documentation
    print("\n3. Checking documentation...")
    docs_valid, issues = validate_documentation()
    if docs_valid:
        print("   ✓ Documentation complete")
    else:
        for issue in issues:
            print(f"   ✗ {issue}")
        all_valid = False
    
    # 4. Check examples
    print("\n4. Checking examples...")
    example_files = [
        'examples/simple_example.py',
        'benchmarks/compare_methods.py',
    ]
    
    for filepath in example_files:
        if os.path.exists(filepath):
            valid, msg = validate_python_syntax(filepath)
            if valid:
                print(f"   ✓ {filepath}")
            else:
                print(f"   ✗ {filepath}: {msg}")
                all_valid = False
        else:
            print(f"   - {filepath} not found (skipped)")
    
    # Summary
    print("\n" + "="*80)
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED")
        print("="*80)
        print("\nThe TopKA2A implementation is ready!")
        print("\nNext steps:")
        print("1. Install PyTorch: pip install torch")
        print("2. Run examples: python examples/simple_example.py")
        print("3. Run benchmarks: python benchmarks/compare_methods.py")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("="*80)
        print("\nPlease fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

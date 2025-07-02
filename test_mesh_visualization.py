#!/usr/bin/env python3
"""
Test script to demonstrate mesh visualization functionality.
"""

import os
import sys
import yaml

def test_mesh_visualization_config():
    """Test that the mesh visualization configuration is properly set up."""
    print("Testing mesh visualization configuration...")
    
    try:
        with open('cfgs/geballe_no_diamond.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check that performance section exists
        performance = cfg.get('performance', {})
        if 'visualize_mesh' in performance:
            visualize_setting = performance['visualize_mesh']
            print(f"✓ Mesh visualization setting found: {visualize_setting}")
            
            if isinstance(visualize_setting, bool):
                print("✓ Mesh visualization setting is a boolean value")
            else:
                print(f"✗ Mesh visualization setting should be boolean, got {type(visualize_setting)}")
                return False
        else:
            print("✗ Mesh visualization setting not found in performance section")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing mesh visualization config: {e}")
        return False

def demonstrate_visualization_options():
    """Demonstrate the different ways to enable mesh visualization."""
    print("\nDemonstrating mesh visualization options...")
    
    print("1. Via configuration file:")
    print("   Set 'performance.visualize_mesh: true' in your YAML config")
    print("   Example:")
    print("   performance:")
    print("     visualize_mesh: true")
    
    print("\n2. Via command line (overrides config):")
    print("   python run_optimized.py --config cfgs/geballe_no_diamond.yaml \\")
    print("     --mesh-folder meshes/geballe_no_diamond \\")
    print("     --output-folder outputs/geballe_no_diamond \\")
    print("     --visualize-mesh")
    
    print("\n3. Via code:")
    print("   engine = OptimizedSimulationEngine(cfg, mesh_folder, output_folder)")
    print("   engine.run(visualize_mesh=True)")
    
    print("\nNote: Mesh visualization requires gmsh to be installed and available.")
    print("The visualization will open a gmsh window showing the mesh structure.")
    
    return True

def check_gmsh_availability():
    """Check if gmsh is available for mesh visualization."""
    print("\nChecking gmsh availability...")
    
    try:
        import gmsh
        print("✓ gmsh is available for mesh visualization")
        return True
    except ImportError:
        print("✗ gmsh is not available for mesh visualization")
        print("  Install gmsh to enable mesh visualization:")
        print("  pip install gmsh")
        return False
    except Exception as e:
        print(f"✗ Error checking gmsh: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing mesh visualization functionality...\n")
    
    tests = [
        test_mesh_visualization_config,
        demonstrate_visualization_options,
        check_gmsh_availability
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Mesh visualization is properly configured.")
        print("\nTo test mesh visualization:")
        print("1. Set 'performance.visualize_mesh: true' in your config file, or")
        print("2. Use the --visualize-mesh command line flag")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 
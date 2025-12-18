"""
Windows-specific setup script for SUMO configuration
"""
import os
import sys
import subprocess
import winreg


def find_sumo_installation():
    """Find SUMO installation on Windows"""
    print("Searching for SUMO installation...")
    
    # Common installation paths
    search_paths = [
        "D:\\sumo\\bin",
        "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin", 
        "C:\\Program Files\\Eclipse\\Sumo\\bin",
        "C:\\sumo\\bin",
        "D:\\Program Files\\sumo\\bin",
        "E:\\sumo\\bin"
    ]
    
    for path in search_paths:
        sumo_exe = os.path.join(path, "sumo.exe")
        sumo_gui_exe = os.path.join(path, "sumo-gui.exe")
        
        if os.path.exists(sumo_exe) and os.path.exists(sumo_gui_exe):
            print(f"✓ Found SUMO at: {path}")
            return path
    
    # Ask user for custom path
    print("SUMO not found in common locations.")
    custom_path = input("Enter the path to your SUMO bin folder (e.g., D:\\sumo\\bin): ").strip()
    
    if custom_path and os.path.exists(os.path.join(custom_path, "sumo.exe")):
        print(f"✓ SUMO found at custom path: {custom_path}")
        return custom_path
    
    return None


def add_to_system_path(sumo_path):
    """Add SUMO to Windows system PATH (requires admin rights)"""
    try:
        # Try to modify system PATH (requires admin)
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment",
                           0, winreg.KEY_ALL_ACCESS) as key:
            
            current_path, _ = winreg.QueryValueEx(key, "PATH")
            
            if sumo_path not in current_path:
                new_path = current_path + ";" + sumo_path
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                print("✓ SUMO added to system PATH (requires restart)")
                return True
            else:
                print("✓ SUMO already in system PATH")
                return True
                
    except PermissionError:
        print("⚠️  Admin rights required to modify system PATH")
        return False
    except Exception as e:
        print(f"✗ Error modifying system PATH: {e}")
        return False


def add_to_user_path(sumo_path):
    """Add SUMO to user PATH (no admin required)"""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_ALL_ACCESS) as key:
            try:
                current_path, _ = winreg.QueryValueEx(key, "PATH")
            except FileNotFoundError:
                current_path = ""
            
            if sumo_path not in current_path:
                new_path = current_path + ";" + sumo_path if current_path else sumo_path
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                print("✓ SUMO added to user PATH")
                return True
            else:
                print("✓ SUMO already in user PATH")
                return True
                
    except Exception as e:
        print(f"✗ Error modifying user PATH: {e}")
        return False


def create_batch_files(sumo_path):
    """Create convenient batch files for running the system"""
    
    # Main batch file
    main_batch = f"""@echo off
title AI Traffic Light System
echo ================================================
echo AI-POWERED ADAPTIVE TRAFFIC LIGHT SYSTEM
echo ================================================
echo.

REM Add SUMO to PATH for this session
set PATH=%PATH%;{sumo_path}

echo SUMO Path: {sumo_path}
echo Python: {sys.executable}
echo.

:menu
echo Available Options:
echo 1. Quick Demo
echo 2. Train Models
echo 3. Evaluate Models  
echo 4. Interactive Demo
echo 5. System Test
echo 6. Exit
echo.
set /p choice="Select option (1-6): "

if "%choice%"=="1" goto demo
if "%choice%"=="2" goto train
if "%choice%"=="3" goto evaluate
if "%choice%"=="4" goto interactive
if "%choice%"=="5" goto test
if "%choice%"=="6" goto exit
goto menu

:demo
echo Running quick demo...
python run_example.py demo
pause
goto menu

:train
echo Starting training...
python src/train.py
pause
goto menu

:evaluate
echo Running evaluation...
python src/evaluate.py
pause
goto menu

:interactive
echo Starting interactive demo...
python src/demo.py
pause
goto menu

:test
echo Running system test...
python test_system.py
pause
goto menu

:exit
echo Goodbye!
pause
"""

    with open("run_traffic_system.bat", "w") as f:
        f.write(main_batch)
    
    # Quick test batch file
    test_batch = f"""@echo off
set PATH=%PATH%;{sumo_path}
echo Testing AI Traffic Light System...
python test_system.py
pause
"""
    
    with open("test_system.bat", "w") as f:
        f.write(test_batch)
    
    print("✓ Created batch files:")
    print("  - run_traffic_system.bat (main interface)")
    print("  - test_system.bat (quick test)")


def setup_sumo_windows():
    """Complete SUMO setup for Windows"""
    print("="*60)
    print("SUMO SETUP FOR WINDOWS")
    print("="*60)
    
    # Find SUMO
    sumo_path = find_sumo_installation()
    if not sumo_path:
        print("✗ SUMO installation not found!")
        print("\nPlease install SUMO:")
        print("1. Download from: https://eclipse.dev/sumo/")
        print("2. Install to D:\\sumo or C:\\Program Files\\Eclipse\\Sumo")
        print("3. Run this script again")
        return False
    
    # Set PATH for current session
    os.environ["PATH"] = sumo_path + ";" + os.environ.get("PATH", "")
    print("✓ SUMO added to current session PATH")
    
    # Try to add to permanent PATH
    print("\nAdding SUMO to permanent PATH...")
    if not add_to_system_path(sumo_path):
        if add_to_user_path(sumo_path):
            print("✓ Added to user PATH (recommended)")
        else:
            print("⚠️  Could not add to PATH automatically")
            print(f"Please manually add {sumo_path} to your PATH")
    
    # Create batch files
    print("\nCreating convenience scripts...")
    create_batch_files(sumo_path)
    
    # Test SUMO
    print("\nTesting SUMO...")
    try:
        result = subprocess.run([os.path.join(sumo_path, "sumo.exe"), "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ SUMO is working correctly")
            print(f"Version: {result.stdout.strip()}")
        else:
            print("⚠️  SUMO test failed")
    except Exception as e:
        print(f"⚠️  SUMO test error: {e}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("You can now run:")
    print("1. run_traffic_system.bat - Main interface")
    print("2. test_system.bat - Quick system test")
    print("3. python test_system.py - Python test")
    print("\nOr restart your command prompt and run:")
    print("python run_example.py demo")
    
    return True


if __name__ == "__main__":
    try:
        import winreg
        success = setup_sumo_windows()
        input("\nPress Enter to exit...")
        sys.exit(0 if success else 1)
    except ImportError:
        print("This script is for Windows only.")
        print("For other systems, use: python setup.py")
        sys.exit(1)
import os
import matplotlib 

print("--- Starting matplotlibrc file creation script ---")


try:
    config_dir = matplotlib.get_configdir()
    print(f"Matplotlib config directory found: {config_dir}")
except Exception as e:
    print(f"Error getting Matplotlib config directory: {e}")
    print("Please ensure matplotlib is installed (`pip install matplotlib`).")
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".matplotlib")
    print(f"Falling back to assumed config directory: {config_dir}")



os.makedirs(config_dir, exist_ok=True)


matplotlibrc_path = os.path.join(config_dir, 'matplotlibrc')


content = "backend: Agg\n"

try:
    
    with open(matplotlibrc_path, 'w') as f:
        f.write(content)
    print(f"Successfully created/updated '{matplotlibrc_path}' with 'backend: Agg'.")
    print("\n--- Next Steps ---")
    print("1. ***RESTART YOUR COMPUTER*** (This is crucial for config changes to apply system-wide).")
    print("2. After restarting, open a FRESH terminal/PowerShell window.")
    print("3. Navigate to your backend directory: cd D:\\SocialPulseAnalyzer\\backend\\")
    print("4. Run your Flask app: python app.py")
    print("5. Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R).")
    print("6. Test your app: Run multiple analysis scenarios.")

except Exception as e:
    print(f"Error writing to file '{matplotlibrc_path}': {e}")
    print("Please check file permissions for the directory.")
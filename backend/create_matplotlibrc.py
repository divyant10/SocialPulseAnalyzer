# D:\SocialPulseAnalyzer\backend\create_matplotlibrc.py

import os
import matplotlib # Import matplotlib here

print("--- Starting matplotlibrc file creation script ---")

# Get Matplotlib's configuration directory
try:
    config_dir = matplotlib.get_configdir()
    print(f"Matplotlib config directory found: {config_dir}")
except Exception as e:
    print(f"Error getting Matplotlib config directory: {e}")
    print("Please ensure matplotlib is installed (`pip install matplotlib`).")
    # Fallback to common user home directory for .matplotlib folder if get_configdir fails
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, ".matplotlib")
    print(f"Falling back to assumed config directory: {config_dir}")


# Ensure the config directory exists
os.makedirs(config_dir, exist_ok=True)

# Define the full path for the matplotlibrc file
matplotlibrc_path = os.path.join(config_dir, 'matplotlibrc')

# Content for the matplotlibrc file
content = "backend: Agg\n"

try:
    # Write the content to the file
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
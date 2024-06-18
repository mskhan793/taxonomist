# version_info.py

import sys

def print_python_version():
    version = sys.version
    version_info = sys.version_info

    print(f"Full Python Version: {version}")
    print(f"Major Version: {version_info.major}")
    print(f"Minor Version: {version_info.minor}")
    print(f"Micro Version: {version_info.micro}")
    print(f"Release Level: {version_info.releaselevel}")
    print(f"Serial: {version_info.serial}")

if __name__ == "__main__":
    print_python_version()

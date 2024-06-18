import zipfile
import os

def unzip_file(zip_file, extract_to):
    """
    Unzips a file to the specified location.
    
    Parameters:
    - zip_file (str): Path to the zip file.
    - extract_to (str): Directory to extract the zip file contents to.
    """
    # Check if the zip file exists
    if not os.path.exists(zip_file):
        print(f"The zip file '{zip_file}' does not exist.")
        return
    
    # Check if the extract directory exists, if not, create it
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"The zip file '{zip_file}' has been successfully extracted to '{extract_to}'.")

# Path to the zip file
zip_file = "data/raw/cifar10.zip"

# Directory to extract the zip file contents to
extract_to = "data/raw/cifar10"

# Call the function to unzip the file
unzip_file(zip_file, extract_to)
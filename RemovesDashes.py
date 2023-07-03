import shutil

# Define the file paths
file_path = "YOUR DATA HERE"
backup_file_path = "YOUR DATA HERE"

# Create a backup file
shutil.copy(file_path, backup_file_path)

# Open the file for reading
with open(file_path, "r") as file:
    # Read the contents of the file
    file_content = file.read()

# Remove any "-" characters from the text
modified_content = file_content.replace("-", "")

# Open the file for writing
with open(file_path, "w") as file:
    # Write the modified content to the file
    file.write(modified_content)

# Close the file
file.close()

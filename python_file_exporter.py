import os

def export_python_files():
    output_filename = "all_python_code.txt"
    with open(output_filename, "w") as outfile:
        for subdir, dirs, files in os.walk("."):
            if "justin" in subdir.split(os.path.sep):
                continue

            for file in files:
                if file.endswith(".py") and file != "python_file_exporter.py" and file != "ML_models.py":
                    filepath = os.path.join(subdir, file)
                    with open(filepath, "r") as infile:
                        outfile.write(f"{filepath.replace('./', '')}\n" + "-"*len(filepath.replace('./', '')) + "\n\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n==============================================\n\n")
                    print(filepath.replace('./', ''))

if __name__ == "__main__":
    export_python_files() 
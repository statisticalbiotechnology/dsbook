import os
import jupytext

# Define the directory containing your Markdown files
md_directory = "./dsbook/"  # Adjust this path to point to your MyST Markdown files

# Walk through all subdirectories and files in the directory
for root, _, files in os.walk(md_directory):
    # Skip the "_build" directory
    if "_build" in root:
        continue
    for filename in files:
        if filename.endswith(".md"):
            md_path = os.path.join(root, filename)
            print(f"Converting {md_path}")
            # Read the Markdown file and convert it to a notebook
            nb = jupytext.read(md_path)
            # Save the notebook with the same name but with an .ipynb extension
            jupytext.write(nb, md_path.replace(".md", ".ipynb"))

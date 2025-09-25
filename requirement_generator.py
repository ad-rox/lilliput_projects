import os
import ast

def extract_libraries_from_file(file_path):
    libraries = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read(), filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        libraries.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    libraries.add(node.module)
        except SyntaxError as e:
            print(f"Error parsing {file_path}: {e}")
    
    return libraries

def scan_and_generate_requirements():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    python_files = [f for f in os.listdir(current_directory) if f.endswith('.py')]

    all_libraries = set()

    for python_file in python_files:
        file_path = os.path.join(current_directory, python_file)
        libraries = extract_libraries_from_file(file_path)
        all_libraries.update(libraries)

    with open('requirements.txt', 'w', encoding='utf-8') as req_file:
        for library in sorted(all_libraries):
            req_file.write(f"{library}\n")

    print("Requirements file generated successfully.")

if __name__ == "__main__":
    scan_and_generate_requirements()

"""Goes over files in the directory read the file content and converts the text to lowercase and outputs the result to the original file."""

import argparse
from pathlib import Path

def convert_to_lowercase(file_path: Path):
    with open(file_path, 'r') as file:
        content = file.read().lower()
    with open(file_path, 'w') as file:
        file.write(content)

def convert_files_to_lowercase(directory_path: Path):
    for file in directory_path.iterdir():
        if file.is_file():
            convert_to_lowercase(file)


def main():
    parser = argparse.ArgumentParser(description="Converts all files in the directory to lowercase.")
    parser.add_argument("--directory_path", required=True, help="The path to the directory with files to convert.")
    args = parser.parse_args()
    directory_path = Path(args.directory_path)
    convert_files_to_lowercase(directory_path)

if __name__ == "__main__":
    main()
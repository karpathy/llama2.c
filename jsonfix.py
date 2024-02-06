import os
import json
import argparse

def convert_json_to_utf8(filename):
    # Read the JSON file
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Write the JSON file back with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Convert JSON files to UTF-8 encoding')
    parser.add_argument('directory', type=str, help='Directory containing JSON files')
    args = parser.parse_args()

    # Iterate over JSON files in the directory
    for filename in os.listdir(args.directory):
        if filename.endswith('.json'):
            filepath = os.path.join(args.directory, filename)
            convert_json_to_utf8(filepath)

if __name__ == "__main__":
    main()
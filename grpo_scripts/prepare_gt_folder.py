
import json
import os
import shutil
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract images from JSON dataset to a GT folder.")
    parser.add_argument("--json_file", type=str, default="/disk/CJN/Chart2SVG_dataset/test_data/grpo_test_beagle.json", help="Path to the dataset JSON file")
    parser.add_argument("--output_folder", type=str, default="grpo_test_beagle_gt", help="Path to the output GT folder")
    args = parser.parse_args()

    json_file = args.json_file
    output_folder = args.output_folder

    if not os.path.exists(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    print(f"Loading dataset from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    print(f"Found {len(data)} entries. Starting copy process...")
    
    success_count = 0
    fail_count = 0

    for entry in tqdm(data, desc="Copying images"):
        images = entry.get("images", [])
        if not images:
            continue
        
        # Assuming only one image per entry as per user example, or taking the first one
        src_path = images[0]
        
        if not src_path:
            continue

        # Handle relative paths if necessary (though example shows absolute)
        # If src_path is relative, it might be relative to the json file location or cwd.
        # Here we assume absolute path or correct relative path from CWD.
        
        if not os.path.exists(src_path):
            # Try checking relative to json file directory just in case
            json_dir = os.path.dirname(os.path.abspath(json_file))
            alt_path = os.path.join(json_dir, src_path)
            if os.path.exists(alt_path):
                src_path = alt_path
            else:
                # print(f"Warning: Source image not found: {src_path}")
                fail_count += 1
                continue

        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, filename)

        try:
            shutil.copy2(src_path, dst_path)
            success_count += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            fail_count += 1

    print("-" * 50)
    print(f"Processing complete.")
    print(f"Successfully copied: {success_count}")
    print(f"Failed/Missing: {fail_count}")
    print(f"Images saved to: {output_folder}")

if __name__ == "__main__":
    main()

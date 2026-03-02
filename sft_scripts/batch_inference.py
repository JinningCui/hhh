import subprocess
from pathlib import Path
import os
import sys

# Define the list of datasets to process
datasets = [
    # "/disk/CJN/ChartLlama/ours/box_chart.json",
    # "/disk/CJN/ChartLlama/ours/candlestick_chart.json",
    # "/disk/CJN/ChartLlama/ours/funnel_chart.json",
    # "/disk/CJN/ChartLlama/ours/gantt_chart.json",
    # "/disk/CJN/ChartLlama/ours/heatmap_chart.json",
    # "/disk/CJN/ChartLlama/ours/polar_chart.json",
    # "/disk/CJN/ChartLlama/ours/scatter_chart.json",
    # "/disk/CJN/VisAnatomy/charts_png.json"
    # '/disk/CJN/Chart2SVG_dataset/test_data/grpo_test_beagle_ckp130.json',
    '/disk/CJN/Chart2SVG_dataset/draft/draft_sft.json',
    # "/disk/CJN/vis_test_05.json"
    # '/home/u20249114/ms-swift/Dataset/test.jsonl'
]

# Path to the inference script
script_path = "/home/u20249114/ms-swift/examples/train/run_inference_and_render.py"

if not os.path.exists(script_path):
    print(f"Error: Inference script not found at {script_path}")
    sys.exit(1)

for dataset_str in datasets:
    dataset_path = Path(dataset_str)
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Warning: Dataset {dataset_path} not found. Skipping.")
        continue
        
    parent_dir = dataset_path.parent
    stem = dataset_path.stem
    
    # Define output directories in the parent folder of the json file
    # We use specific directories for each dataset to avoid overwriting results
    # if multiple datasets are in the same folder.
    output_dir = parent_dir / f"{stem}_inference_results"
    resized_dir = parent_dir / f"{stem}_resized_inputs"
    
    print(f"\n{'='*50}")
    print(f"Processing dataset: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Resized Inputs Directory: {resized_dir}")
    print(f"{'='*50}\n")
    
    cmd = [
        "python3",
        script_path,
        "--dataset_path", str(dataset_path),
        "--output_dir", str(output_dir),
        "--resized_dir", str(resized_dir)
    ]
    
    try:
        # Run the inference script
        subprocess.run(cmd, check=True)
        print(f"\nSuccessfully processed {dataset_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError processing {dataset_path}: {e}")
        # Option: continue to next dataset or stop?
        # We continue to try processing others.
    except Exception as e:
        print(f"\nUnexpected error for {dataset_path}: {e}")

print("\nBatch inference completed.")

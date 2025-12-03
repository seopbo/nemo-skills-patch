import subprocess
import os
import sys
from pathlib import Path

def run_augment_mcq_batch():
    """
    Batch process multiple files through augment_mcq.py
    Similar structure to tmp.py but for MCQ augmentation
    """
    
    # Define your input files and their corresponding output directories
    dataset_dict = [
        {"input_file": "/seed_data/mcq-4/aops_mcq.jsonl", "output_base": "/workspace/DATA/augmented"},
        {"input_file": "/seed_data/mcq-4/cdquestions_mcq.jsonl", "output_base": "/workspace/DATA/augmented"},
        {"input_file": "/seed_data/mcq-4/rqa_mcq.jsonl", "output_base": "/workspace/DATA/augmented"},

    ]
    
    cluster = "dfw"  # Change to "eos", "oci", etc. as needed
    
    for dct in dataset_dict:
        input_file = dct["input_file"]
        output_base = dct["output_base"]
        
        # Extract dataset name from input file
        dataset_name = Path(input_file).stem
        output_dir = f"{output_base}/{dataset_name}"
        
        # Construct the command
        cmd = [
            "python", 
            "augment_mcq.py",
            f"input_file={input_file}",
            f"output_dir={output_dir}",
            f"cluster={cluster}",
        ]
        
        print(f"Running MCQ augmentation for {dataset_name}...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 80)
        
        try:
            # Use Popen with direct stdout/stderr to preserve colors
            process = subprocess.Popen(
                cmd, 
                stdout=None,  # Let subprocess write directly to terminal
                stderr=None,  # Let subprocess write directly to terminal
                text=True
            )
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✓ Success: {dataset_name}")
            else:
                print(f"✗ Failed: {dataset_name} (exit code: {return_code})")
                
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {e}")
        
        print("=" * 80)
        print()  # Add blank line between datasets


if __name__ == "__main__":
    run_augment_mcq_batch()
import subprocess
import os
import sys
from pathlib import Path

# def run_pipeline_on_chunks():
    
#     base_path = "/ebakhturina/data/wolfram/wa_chunks/fixed"
    
#     chunk_files = [
#         "1MPhysicsSamples.wolfram.train.decontaminated_chunk_1.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_2.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_3.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_4.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_5.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_6.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_7.jsonl",
#         # "1MPhysicsSamples.wolfram.train.decontaminated_chunk_8.jsonl",
#     ]

#     chunk_files = [
#         "/workspace/DATA/_data/seed_data/1MPhysicsSamples_wolfram.jsonl"
#     ]
    
#     for chunk_file in chunk_files:
#         # Extract dataset name (remove .jsonl extension)
#         dataset_name = "1MPhysicsSamples_wolfram"
        
#         # Construct the command
#         cmd = [
#             "python", 
#             "pipeline/sdg_pipeline.py",
#             "--settings", "seed_data",
#             "--override",
#             "cluster=eos",
#             f"dataset_name={dataset_name}",
#             f"input_file={chunk_file}",
#             f"expname={dataset_name}_prep",
#             f"base_output_dir=/workspace/DATA/_data/seed_data/{dataset_name}",
#         ]
        
#         print(f"Running pipeline for {dataset_name}...")
        
#         try:
#             result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#             print(f"Success: {dataset_name}")
#             print(f"Output: {result.stdout}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error processing {dataset_name}: {e}")
#             print(f"Error output: {e.stderr}")



# def run_pipeline_on_chunks():
    

#     dataset_dict = [
#         # {"path":"/seed_data/openq/vedantu_physics_openq.jsonl" , "num_seeds": 1, "chunks": 20},
#         # {"path":"/seed_data/openq/vedantu_biology_chemistry_openq.jsonl" , "num_seeds": 5, "chunks": 3},

#         # {"path":"/seed_data/openq/cdquestions_openq.jsonl" , "num_seeds": 5, "chunks": 3},
#         # {"path":"/seed_data/openq/aops_openq.jsonl" , "num_seeds": 5, "chunks": 3},
#         # {"path":"/seed_data/openq/limo_train.jsonl" , "num_seeds": 5, "chunks": 3},
#         # {"path":"/seed_data/openq/askfilo.jsonl" , "num_seeds": 5, "chunks": 20},
#         # {"path":"/seed_data/openq/ipho.jsonl" , "num_seeds": 5, "chunks": 3},
#         # {"path":"/seed_data/openq/doubtnut.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/openq/scale.jsonl" , "num_seeds": 30, "chunks": 20},
#         # {"path":"/seed_data/openq/rqa_oqa.jsonl" , "num_seeds": 5, "chunks": 5},
#         # {"path":"/seed_data/openq/auxiliary_train.jsonl" , "num_seeds": 5, "chunks": 20},
#         # {"path":"/seed_data/openq/icho.jsonl" , "num_seeds": 30, "chunks": 20},
#         # {"path":"/seed_data/openq/olimpicos-aapt.jsonl" , "num_seeds": 5, "chunks": 20},
#     ]
    
#     for dct in dataset_dict:
#         dataset_name = Path(dct["path"]).stem
#         dataset_path = dct["path"]
#         num_seeds = dct["num_seeds"]
#         chunks = dct["chunks"]
        
#         cmd = [
#             "python", 
#             "pipeline/sdg_pipeline.py",
#             "--settings", "python_enabled",
#             "--override",
#             "cluster=oci",
#             f"dataset_name={dataset_name}",
#             f"input_file={dataset_path}",
#             f"expname={dataset_name}_prep",
#             f"num_random_seeds={num_seeds}",
#             f"num_chunks={chunks}",
#             f"base_output_dir=/workspace/DATA/_data/solution_gen_new_sandbox/{dataset_name}",
#         ]
        
#         print(f"Running pipeline for {dataset_name}...")
#         print(f"Command: {' '.join(cmd)}")
#         print("-" * 80)
        
#         try:
#             # Use Popen with direct stdout/stderr to preserve colors
#             process = subprocess.Popen(
#                 cmd, 
#                 stdout=None,  # Let subprocess write directly to terminal
#                 stderr=None,  # Let subprocess write directly to terminal
#                 text=True
#             )
            
#             # Wait for process to complete
#             return_code = process.wait()
            
#             if return_code == 0:
#                 print(f"✓ Success: {dataset_name}")
#             else:
#                 print(f"✗ Failed: {dataset_name} (exit code: {return_code})")
                
#         except Exception as e:
#             print(f"✗ Error processing {dataset_name}: {e}")
        
#         print("=" * 80)
#         print()  # Add blank line between datasets


# def run_pipeline_on_chunks():
    

#     dataset_dict = [

#         # {"path":"/seed_data/mcq-4-augmented/so.jsonl" , "num_seeds": 5, "chunks": 20},
#         # {"path":"/seed_data/mcq-4-augmented/syn_gpqa_v1.2_4mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         # {"path":"/seed_data/mcq-4-augmented/vedantu_biology_chemistry_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4-augmented/vedantu_physics_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#     ]
    
#     for dct in dataset_dict:
#         dataset_name = Path(dct["path"]).stem
#         dataset_path = dct["path"]
#         num_seeds = dct["num_seeds"]
#         chunks = dct["chunks"]
        
#         cmd = [
#             "python", 
#             "pipeline/sdg_pipeline.py",
#             "--settings", "seed_data_postprocess",
#             "--override",
#             "cluster=eos",
#             f"dataset_name={dataset_name}",
#             f"input_file={dataset_path}",
#             f"expname={dataset_name}_prep",
#             f"num_random_seeds={num_seeds}",
#             f"num_chunks={chunks}",
#             f"base_output_dir=/workspace/DATA/_data/solution_gen_new_sandbox/{dataset_name}",
#         ]
        
#         print(f"Running pipeline for {dataset_name}...")
#         print(f"Command: {' '.join(cmd)}")
#         print("-" * 80)
        
#         try:
#             # Use Popen to stream output in real-time
#             process = subprocess.Popen(
#                 cmd, 
#                 stdout=subprocess.PIPE, 
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 bufsize=1,
#                 universal_newlines=True
#             )
            
#             # Stream output line by line
#             while True:
#                 output = process.stdout.readline()
#                 if output == '' and process.poll() is not None:
#                     break
#                 if output:
#                     print(f"[{dataset_name}] {output.strip()}")
#                     sys.stdout.flush()  # Ensure immediate output
            
#             # Wait for process to complete and get return code
#             return_code = process.poll()
            
#             if return_code == 0:
#                 print(f"✓ Success: {dataset_name}")
#             else:
#                 print(f"✗ Failed: {dataset_name} (exit code: {return_code})")
                
#         except Exception as e:
#             print(f"✗ Error processing {dataset_name}: {e}")
        
#         print("=" * 80)
#         print()  # Add blank line between datasets



def run_pipeline_on_chunks():
    

    dataset_dict = [
        # {"path":"/seed_data/openq/vedantu_physics_openq.jsonl" , "num_seeds": 1, "chunks": 20},
        # {"path":"/seed_data/openq/vedantu_biology_chemistry_openq.jsonl" , "num_seeds": 5, "chunks": 3},

        # {"path":"/seed_data/openq/cdquestions_openq.jsonl" , "num_seeds": 5, "chunks": 3},
        # {"path":"/seed_data/openq/aops_openq.jsonl" , "num_seeds": 5, "chunks": 3},
        # {"path":"/seed_data/openq/limo_train.jsonl" , "num_seeds": 5, "chunks": 3},
        # {"path":"/seed_data/openq/askfilo.jsonl" , "num_seeds": 5, "chunks": 20},
        # {"path":"/seed_data/openq/ipho.jsonl" , "num_seeds": 5, "chunks": 3},
        # {"path":"/seed_data/openq/doubtnut.jsonl" , "num_seeds": 5, "chunks": 20},
        # {"path":"/seed_data/openq/scale.jsonl" , "num_seeds": 30, "chunks": 20},
        # {"path":"/seed_data/openq/rqa_oqa.jsonl" , "num_seeds": 5, "chunks": 5},
        # {"path":"/seed_data/openq/auxiliary_train.jsonl" , "num_seeds": 5, "chunks": 20},
        {"path":"/seed_data/openq/icho.jsonl" , "num_seeds": 5, "chunks": 20},
        # {"path":"/seed_data/openq/olimpicos-aapt.jsonl" , "num_seeds": 5, "chunks": 20},
    ]
    
    for dct in dataset_dict:
        dataset_name = Path(dct["path"]).stem
        dataset_path = dct["path"]
        num_seeds = dct["num_seeds"]
        chunks = dct["chunks"]
        
        cmd = [
            "python", 
            "pipeline/sdg_pipeline.py",
            "--settings", "kimi-no-tool-hle",
            "--override",
            "cluster=lax",
            f"dataset_name={dataset_name}",
            f"input_file={dataset_path}",
            f"expname={dataset_name}_prep",
            f"num_random_seeds={num_seeds}",
            f"num_chunks={chunks}",
            f"base_output_dir=/workspace/DATA/solution_gen_new_sandbox/{dataset_name}",
        ]
        
        print(f"Running pipeline for {dataset_name}...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 80)
        
        try:
            # Use Popen with direct stdout/stderr to preserve colors
            process = subprocess.Popen(
                cmd, 
                stdout=None,  
                stderr=None,  
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


# def run_pipeline_on_chunks():   

#     dataset_dict = [

#         {"path":"/seed_data/mcq-4-augmented/so.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4-augmented/syn_gpqa_v1.2_4mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4-augmented/vedantu_biology_chemistry_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4-augmented/vedantu_physics_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-10/aops_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#     ]
    
#     for dct in dataset_dict:
#         dataset_name = Path(dct["path"]).stem
#         mid_dir = dct["path"].split("/")[-2]
#         dataset_path = dct["path"]
#         num_seeds = dct["num_seeds"]
#         chunks = dct["chunks"]
        
#         cmd = [
#             "python", 
#             "pipeline/sdg_pipeline.py",
#             "--settings", "kimi-10-mcq",
#             "--override",
#             "cluster=lax",
#             f"dataset_name={dataset_name}",
#             f"input_file={dataset_path}",
#             f"expname={dataset_name}_prep",
#             f"num_random_seeds={num_seeds}",
#             f"num_chunks={chunks}",
#             f"base_output_dir=/workspace/DATA/solution_gen_new_sandbox/{mid_dir}/{dataset_name}",
#         ]
        
#         print(f"Running pipeline for {dataset_name}...")
#         print(f"Command: {' '.join(cmd)}")
#         print("-" * 80)
        
#         try:
#             # Use Popen to stream output in real-time
#             process = subprocess.Popen(
#                 cmd, 
#                 stdout=subprocess.PIPE, 
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 bufsize=1,
#                 universal_newlines=True
#             )
            
#             # Stream output line by line
#             while True:
#                 output = process.stdout.readline()
#                 if output == '' and process.poll() is not None:
#                     break
#                 if output:
#                     print(f"[{dataset_name}] {output.strip()}")
#                     sys.stdout.flush()  # Ensure immediate output
            
#             # Wait for process to complete and get return code
#             return_code = process.poll()
            
#             if return_code == 0:
#                 print(f"✓ Success: {dataset_name}")
#             else:
#                 print(f"✗ Failed: {dataset_name} (exit code: {return_code})")
                
#         except Exception as e:
#             print(f"✗ Error processing {dataset_name}: {e}")
        
#         print("=" * 80)
#         print()  # Add blank line between datasets



# def run_pipeline_on_chunks():   

#     dataset_dict = [

#         {"path":"/seed_data/mcq-4/so.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4/syn_gpqa_v1.2_4mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4/vedantu_biology_chemistry_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4/vedantu_physics_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4/aops_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4/cdquestions_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#         {"path":"/seed_data/mcq-4/rqa_mcq.jsonl" , "num_seeds": 5, "chunks": 20},
#     ]
    
    
#     for dct in dataset_dict:
#         dataset_name = Path(dct["path"]).stem
#         mid_dir = dct["path"].split("/")[-2]
#         dataset_path = dct["path"]
#         num_seeds = dct["num_seeds"]
#         chunks = dct["chunks"]
        
#         cmd = [
#             "python", 
#             "pipeline/sdg_pipeline.py",
#             "--settings", "kimi-4-mcq",
#             "--override",
#             "cluster=lax",
#             f"dataset_name={dataset_name}",
#             f"input_file={dataset_path}",
#             f"expname={dataset_name}_prep",
#             f"num_random_seeds={num_seeds}",
#             f"num_chunks={chunks}",
#             f"base_output_dir=/workspace/DATA/solution_gen_new_sandbox/{mid_dir}/{dataset_name}",
#         ]
        
#         print(f"Running pipeline for {dataset_name}...")
#         print(f"Command: {' '.join(cmd)}")
#         print("-" * 80)
        
#         try:
#             # Use Popen to stream output in real-time
#             process = subprocess.Popen(
#                 cmd, 
#                 stdout=subprocess.PIPE, 
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 bufsize=1,
#                 universal_newlines=True
#             )
            
#             # Stream output line by line
#             while True:
#                 output = process.stdout.readline()
#                 if output == '' and process.poll() is not None:
#                     break
#                 if output:
#                     print(f"[{dataset_name}] {output.strip()}")
#                     sys.stdout.flush()  # Ensure immediate output
            
#             # Wait for process to complete and get return code
#             return_code = process.poll()
            
#             if return_code == 0:
#                 print(f"✓ Success: {dataset_name}")
#             else:
#                 print(f"✗ Failed: {dataset_name} (exit code: {return_code})")
                
#         except Exception as e:
#             print(f"✗ Error processing {dataset_name}: {e}")
        
#         print("=" * 80)
#         print()  # Add blank line between datasets



if __name__ == "__main__":
    run_pipeline_on_chunks()
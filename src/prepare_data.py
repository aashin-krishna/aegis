import os
import gc
import random
import pandas as pd
import numpy as np

# Define input paths (relative to workspace root d:/PROJECT)
CICIDS_INPUT = 'Payload_data_CICIDS2017.csv'
UNSW_INPUT = 'Payload_data_UNSW.csv'
SYNTH_INPUT = 'synthetic_threats_large.csv'

# Define output directories
TRAIN_DIR = 'aegis_scratch/data/train'
VAL_DIR = 'aegis_scratch/data/val'

# Define individual output paths
CICIDS_TRAIN = os.path.join(TRAIN_DIR, 'cicids_train.csv')
CICIDS_VAL = os.path.join(VAL_DIR, 'cicids_val.csv')

UNSW_TRAIN = os.path.join(TRAIN_DIR, 'unsw_train.csv')
UNSW_VAL = os.path.join(VAL_DIR, 'unsw_val.csv')

SYNTH_TRAIN = os.path.join(TRAIN_DIR, 'synthetic_train.csv')
SYNTH_VAL = os.path.join(VAL_DIR, 'synthetic_val.csv')

# Combined output paths
COMBINED_TRAIN = os.path.join(TRAIN_DIR, 'combined_train.csv')
COMBINED_VAL = os.path.join(VAL_DIR, 'combined_val.csv')

# 6 unified behavior classes: BENIGN, DoS, PortScan, Infiltration, Bot, Brute Force
UNSW_MAPPING = {
    'normal': 'BENIGN',
    'dos': 'DoS',
    'reconnaissance': 'PortScan',
    'exploits': 'Infiltration',
    'backdoor': 'Infiltration',
    'shellcode': 'Infiltration',
    'fuzzers': 'Infiltration',
    'generic': 'Infiltration',
    'worms': 'Bot',
    'analysis': 'Brute Force'
}

CICIDS_MAPPING = {
    'benign': 'BENIGN',
    'portscan': 'PortScan',
    'dos slowloris': 'DoS',
    'dos hulk': 'DoS',
    'dos goldeneye': 'DoS',
    'dos slowhttptest': 'DoS',
    'dos': 'DoS',
    'ddos': 'DoS',
    'infiltration': 'Infiltration',
    'bot': 'Bot',
    'ssh-patator': 'Brute Force',
    'ftp-patator': 'Brute Force',
    'web attack – brute force': 'Brute Force',
    'web attack - brute force': 'Brute Force',
    'web attack – sql injection': 'Infiltration',
    'web attack - sql injection': 'Infiltration',
    'web attack – xss': 'Infiltration',
    'web attack - xss': 'Infiltration',
    'heartbleed': 'Infiltration'
}

# Synthetic dataset uses the same label set as CICIDS
SYNTH_MAPPING = CICIDS_MAPPING

def split_and_map_dataset(input_path, output_train_path, output_val_path, mapping, name):
    print(f"\n=== Processing {name} dataset from {input_path} ===")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Skipping...")
        return False
        
    # Pass 1: Indexing
    print("Pass 1: Ingesting labels and indexing...")
    class_indices = {}
    global_idx = 0
    chunksize = 100000
    
    for chunk in pd.read_csv(input_path, usecols=['label'], chunksize=chunksize):
        for raw_label in chunk['label']:
            lbl_str = str(raw_label).strip().lower()
            mapped_label = mapping.get(lbl_str, None)
            if mapped_label is not None:
                if mapped_label not in class_indices:
                    class_indices[mapped_label] = []
                class_indices[mapped_label].append(global_idx)
            global_idx += 1
            
    print("Mapped class distribution:")
    total_valid = 0
    for lbl, idxs in class_indices.items():
        print(f"  {lbl}: {len(idxs)} rows")
        total_valid += len(idxs)
        
    # Split indices 60-40 stratified by class
    train_indices = set()
    val_indices = set()
    
    random.seed(42)
    for lbl, idxs in class_indices.items():
        random.shuffle(idxs)
        split_point = int(len(idxs) * 0.6)
        train_idxs = idxs[:split_point]
        val_idxs = idxs[split_point:]
        
        train_indices.update(train_idxs)
        val_indices.update(val_idxs)
        
    print(f"Index split completed. Train rows: {len(train_indices)}, Val rows: {len(val_indices)}")
    
    # Free memory
    del class_indices
    gc.collect()
    
    # Pass 2: Writing files
    print("Pass 2: Reading dataset in chunks and writing train/val files...")
    
    dtypes = {f'payload_byte_{i}': np.uint8 for i in range(1, 1501)}
    dtypes['ttl'] = np.uint16
    dtypes['total_len'] = np.uint32
    dtypes['protocol'] = str
    dtypes['t_delta'] = np.float32
    dtypes['label'] = str
    
    train_header_written = False
    val_header_written = False
    
    global_idx = 0
    
    if os.path.exists(output_train_path):
        os.remove(output_train_path)
    if os.path.exists(output_val_path):
        os.remove(output_val_path)
        
    for chunk in pd.read_csv(input_path, dtype=dtypes, chunksize=chunksize):
        chunk_indices = range(global_idx, global_idx + len(chunk))
        
        # Apply label mapping to chunk
        chunk['label'] = chunk['label'].astype(str).str.strip().str.lower().map(mapping)
        # Drop rows that don't match our mapping (e.g. unknown classes)
        chunk = chunk.dropna(subset=['label'])
        
        # Filter for train
        train_mask = [idx in train_indices for idx in chunk_indices]
        train_chunk = chunk[train_mask]
        if not train_chunk.empty:
            train_chunk.to_csv(output_train_path, mode='a', index=False, header=not train_header_written)
            train_header_written = True
            
        # Filter for val
        val_mask = [idx in val_indices for idx in chunk_indices]
        val_chunk = chunk[val_mask]
        if not val_chunk.empty:
            val_chunk.to_csv(output_val_path, mode='a', index=False, header=not val_header_written)
            val_header_written = True
            
        global_idx += len(chunk)
        
    print(f"Dataset {name} split and saved successfully.")
    gc.collect()
    return True

def combine_datasets(train_paths, val_paths, combined_train_out, combined_val_out):
    print("\n=== Combining split datasets ===")
    
    # Combine training sets
    if os.path.exists(combined_train_out):
        os.remove(combined_train_out)
    train_header_written = False
    for path in train_paths:
        if not os.path.exists(path):
            continue
        print(f"Appending train dataset: {path}")
        for chunk in pd.read_csv(path, chunksize=100000):
            chunk.to_csv(combined_train_out, mode='a', index=False, header=not train_header_written)
            train_header_written = True
            
    # Combine validation sets
    if os.path.exists(combined_val_out):
        os.remove(combined_val_out)
    val_header_written = False
    for path in val_paths:
        if not os.path.exists(path):
            continue
        print(f"Appending val dataset: {path}")
        for chunk in pd.read_csv(path, chunksize=100000):
            chunk.to_csv(combined_val_out, mode='a', index=False, header=not val_header_written)
            val_header_written = True
            
    print(f"\nCombined training dataset saved to: {combined_train_out}")
    print(f"Combined validation dataset saved to: {combined_val_out}")

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # 1. Process CICIDS
    cicids_success = split_and_map_dataset(
        CICIDS_INPUT, 
        CICIDS_TRAIN, 
        CICIDS_VAL, 
        CICIDS_MAPPING, 
        "CICIDS2017"
    )
    
    # 2. Process UNSW
    unsw_success = split_and_map_dataset(
        UNSW_INPUT, 
        UNSW_TRAIN, 
        UNSW_VAL, 
        UNSW_MAPPING, 
        "UNSW-NB15"
    )
    
    # 3. Process Synthetic
    synth_success = split_and_map_dataset(
        SYNTH_INPUT, 
        SYNTH_TRAIN, 
        SYNTH_VAL, 
        SYNTH_MAPPING, 
        "Synthetic Threats"
    )
    
    # 4. Combine them
    train_paths = []
    val_paths = []
    
    if cicids_success:
        train_paths.append(CICIDS_TRAIN)
        val_paths.append(CICIDS_VAL)
    if unsw_success:
        train_paths.append(UNSW_TRAIN)
        val_paths.append(UNSW_VAL)
    if synth_success:
        train_paths.append(SYNTH_TRAIN)
        val_paths.append(SYNTH_VAL)
        
    combine_datasets(train_paths, val_paths, COMBINED_TRAIN, COMBINED_VAL)
    print("\nDataset preparation pipeline finished successfully!")

if __name__ == '__main__':
    main()

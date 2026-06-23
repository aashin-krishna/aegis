import os
import gc
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import custom model
from model import PayloadCNNBiLSTMBERT

# Constants
MODEL_DIR = 'aegis_scratch/models'
VAL_DIR = 'aegis_scratch/data/val'

def load_dataset_numpy(csv_path):
    print(f"\nLoading validation set: {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist. Skipping evaluation for this subset.")
        return None, None, None
        
    payload_cols = [f'payload_byte_{i}' for i in range(1, 1501)]
    meta_cols = ['ttl', 'total_len', 't_delta', 'protocol']
    
    dtypes = {f'payload_byte_{i}': np.uint8 for i in range(1, 1501)}
    dtypes['ttl'] = np.uint16
    dtypes['total_len'] = np.uint32
    dtypes['protocol'] = str
    dtypes['t_delta'] = np.float32
    dtypes['label'] = str
    
    chunksize = 100000
    payload_list = []
    meta_list = []
    labels_list = []
    
    for chunk in pd.read_csv(csv_path, dtype=dtypes, chunksize=chunksize):
        payload_list.append(chunk[payload_cols].values)
        meta_list.append(chunk[meta_cols])
        labels_list.append(chunk['label'].values)
        
    payloads = np.concatenate(payload_list, axis=0)
    meta_df = pd.concat(meta_list, ignore_index=True)
    labels = np.concatenate(labels_list, axis=0)
    
    print(f"Loaded {len(payloads)} rows.")
    return payloads, meta_df, labels

def evaluate_on_dataset(model, csv_path, label_encoder, protocol_encoder, scaler, batch_size, device, name):
    payloads, meta_df, labels_str = load_dataset_numpy(csv_path)
    if payloads is None:
        return
        
    # Pre-process
    y_true = label_encoder.transform(labels_str)
    
    val_protos = meta_df['protocol'].astype(str).str.strip().str.lower().fillna('unknown')
    # Handle unseen protocols
    val_protos = val_protos.map(lambda s: s if s in protocol_encoder.classes_ else 'unknown')
    if 'unknown' not in protocol_encoder.classes_:
        protocol_encoder.classes_ = np.append(protocol_encoder.classes_, 'unknown')
    meta_df['protocol_encoded'] = protocol_encoder.transform(val_protos)
    
    meta_cols = ['ttl', 'total_len', 't_delta', 'protocol_encoded']
    X_meta = scaler.transform(meta_df[meta_cols].values.astype(np.float32))
    
    # Dataset and DataLoader
    dataset = TensorDataset(
        torch.tensor(payloads, dtype=torch.uint8),
        torch.tensor(X_meta, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluation loop
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_payloads, batch_metas, batch_labels in loader:
            batch_payloads = batch_payloads.to(device)
            batch_metas = batch_metas.to(device)
            
            outputs = model(batch_payloads, batch_metas)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.numpy())
            
    print(f"\n==========================================")
    print(f"Classification Report for: {name}")
    print(f"==========================================")
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=label_encoder.classes_, 
        labels=range(len(label_encoder.classes_)),
        zero_division=0
    ))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print(f"==========================================\n")
    
    # Free memory
    del payloads, meta_df, dataset, loader
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Aegis Threat Classifier")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")
    
    # Load model and preprocessing helpers
    label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
    protocol_encoder_path = os.path.join(MODEL_DIR, 'protocol_encoder.joblib')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    model_weights_path = os.path.join(MODEL_DIR, 'model.pth')
    
    if not (os.path.exists(label_encoder_path) and os.path.exists(model_weights_path)):
        print("Error: Trained model weights or encoders not found in aegis_scratch/models/. Please run train.py first.")
        return
        
    label_encoder = joblib.load(label_encoder_path)
    protocol_encoder = joblib.load(protocol_encoder_path)
    scaler = joblib.load(scaler_path)
    
    num_classes = len(label_encoder.classes_)
    
    # Instantiate and load model
    model = PayloadCNNBiLSTMBERT(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    print("Model weights loaded successfully.")
    
    # Evaluate on individual val datasets
    val_datasets = {
        "CICIDS2017 Validation Set": os.path.join(VAL_DIR, 'cicids_val.csv'),
        "UNSW-NB15 Validation Set": os.path.join(VAL_DIR, 'unsw_val.csv'),
        "Synthetic Threats Validation Set": os.path.join(VAL_DIR, 'synthetic_val.csv'),
        "Combined Validation Set (CICIDS + UNSW + Synthetic)": os.path.join(VAL_DIR, 'combined_val.csv')
    }
    
    for name, path in val_datasets.items():
        evaluate_on_dataset(
            model=model,
            csv_path=path,
            label_encoder=label_encoder,
            protocol_encoder=protocol_encoder,
            scaler=scaler,
            batch_size=args.batch_size,
            device=device,
            name=name
        )

if __name__ == '__main__':
    main()

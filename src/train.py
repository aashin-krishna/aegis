import os
import gc
import random
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import our custom model and Focal Loss
from model import PayloadCNNBiLSTMBERT, PayloadMambaAttentionClassifier, FocalLoss

# Define constants
MODEL_DIR = 'aegis_scratch/models'
TRAIN_DATA_PATH = 'aegis_scratch/data/train/combined_train.csv'
VAL_DATA_PATH = 'aegis_scratch/data/val/combined_val.csv'

def load_dataset_numpy(csv_path):
    print(f"Loading {csv_path} to numpy arrays...")
    
    # 1500 payload byte column names
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
    
    print(f"Loaded {len(payloads)} rows successfully.")
    return payloads, meta_df, labels

def sample_balanced(payloads, meta_df, labels, sample_size):
    print(f"Sampling balanced training subset of size {sample_size}...")
    unique_classes, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)
    target_per_class = sample_size // num_classes
    
    indices_to_keep = []
    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        # If class has fewer samples than target, take all of them
        take_n = min(len(c_indices), target_per_class)
        selected = np.random.choice(c_indices, take_n, replace=False)
        indices_to_keep.extend(selected)
        
    random.shuffle(indices_to_keep)
    
    sampled_payloads = payloads[indices_to_keep]
    sampled_meta_df = meta_df.iloc[indices_to_keep].reset_index(drop=True)
    sampled_labels = labels[indices_to_keep]
    
    print(f"Balanced sampling complete. Rows kept: {len(sampled_labels)}")
    print(pd.Series(sampled_labels).value_counts())
    return sampled_payloads, sampled_meta_df, sampled_labels

def main():
    parser = argparse.ArgumentParser(description="Train AegisNet Threat Classifiers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Batch size for validation")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=100000, help="Downsample training set to this size (balanced) for speed, set <=0 for full data")
    parser.add_argument("--val_sample_size", type=int, default=20000, help="Downsample validation set to this size for speed, set <=0 for full data")
    parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "mamba_attention"], help="Which architecture to train")
    parser.add_argument("--robust_train", action="store_true", help="Enable PGD adversarial training")
    parser.add_argument("--cv", action="store_true", help="Perform 5-fold cross-validation")
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load Training Data
    X_train_payload, meta_train_df, y_train_labels = load_dataset_numpy(TRAIN_DATA_PATH)
    
    # Balance / Sample Training Data if specified
    if args.sample_size > 0 and args.sample_size < len(y_train_labels):
        X_train_payload, meta_train_df, y_train_labels = sample_balanced(
            X_train_payload, meta_train_df, y_train_labels, args.sample_size
        )
        
    # Load Validation Data
    X_val_payload, meta_val_df, y_val_labels = load_dataset_numpy(VAL_DATA_PATH)
    
    # Sample Validation Data if specified (balanced split if possible, or simple sample)
    if args.val_sample_size > 0 and args.val_sample_size < len(y_val_labels):
        X_val_payload, meta_val_df, y_val_labels = sample_balanced(
            X_val_payload, meta_val_df, y_val_labels, args.val_sample_size
        )

    # 1. Label Encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    
    num_classes = len(label_encoder.classes_)
    print(f"Model will classify {num_classes} classes: {list(label_encoder.classes_)}")
    
    # 2. Protocol Encoding
    protocol_encoder = LabelEncoder()
    # Normalize protocol strings
    train_protos = meta_train_df['protocol'].astype(str).str.strip().str.lower().fillna('unknown')
    val_protos = meta_val_df['protocol'].astype(str).str.strip().str.lower().fillna('unknown')
    
    # Fit encoder
    all_protos = pd.concat([train_protos, val_protos]).unique()
    protocol_encoder.fit(all_protos)
    
    meta_train_df['protocol_encoded'] = protocol_encoder.transform(train_protos)
    meta_val_df['protocol_encoded'] = protocol_encoder.transform(val_protos)
    joblib.dump(protocol_encoder, os.path.join(MODEL_DIR, 'protocol_encoder.joblib'))
    
    # 3. Scale Metadata Features
    meta_cols = ['ttl', 'total_len', 't_delta', 'protocol_encoded']
    scaler = StandardScaler()
    X_train_meta = scaler.fit_transform(meta_train_df[meta_cols].values.astype(np.float32))
    X_val_meta = scaler.transform(meta_val_df[meta_cols].values.astype(np.float32))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    print("Pre-processing complete. Saving encoders to models/ directory.")
    
    # Free dataframe memory
    del meta_train_df, meta_val_df, train_protos, val_protos
    gc.collect()
    
    # Create PyTorch datasets and loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_payload, dtype=torch.uint8),
        torch.tensor(X_train_meta, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val_payload, dtype=torch.uint8),
        torch.tensor(X_val_meta, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    
    print(f"Data loaders initialized. Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # 4. Focal Loss & Class Weights Setup
    # Calculate inverse class frequencies as alpha weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    # Avoid division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Calculated Focal Loss class weights (alpha): {class_weights}")
    
    # Initialize Model, Loss, Optimizer
    if args.model_type == "mamba_attention":
        model = PayloadMambaAttentionClassifier(num_classes=num_classes).to(device)
    else:
        model = PayloadCNNBiLSTMBERT(num_classes=num_classes).to(device)
        
    criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    
    if args.cv:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        print("\nStarting 5-Fold Cross-Validation...")
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_payload)):
            print(f"\n--- Fold {fold+1}/5 ---")
            fold_train_dataset = TensorDataset(
                torch.tensor(X_train_payload[train_idx], dtype=torch.uint8),
                torch.tensor(X_train_meta[train_idx], dtype=torch.float32),
                torch.tensor(y_train[train_idx], dtype=torch.long)
            )
            fold_val_dataset = TensorDataset(
                torch.tensor(X_train_payload[val_idx], dtype=torch.uint8),
                torch.tensor(X_train_meta[val_idx], dtype=torch.float32),
                torch.tensor(y_train[val_idx], dtype=torch.long)
            )
            
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=args.batch_size, shuffle=True)
            fold_val_loader = DataLoader(fold_val_dataset, batch_size=args.val_batch_size, shuffle=False)
            
            if args.model_type == "mamba_attention":
                fold_model = PayloadMambaAttentionClassifier(num_classes=num_classes).to(device)
            else:
                fold_model = PayloadCNNBiLSTMBERT(num_classes=num_classes).to(device)
                
            fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=args.lr)
            
            for epoch in range(min(args.epochs, 2)):
                fold_model.train()
                for payloads, metas, labels in fold_train_loader:
                    payloads = payloads.to(device)
                    metas = metas.to(device)
                    labels = labels.to(device)
                    
                    fold_optimizer.zero_grad()
                    if args.robust_train:
                        embeddings = fold_model.embedding(payloads.long()).clone().detach().requires_grad_(True)
                        for _ in range(5):
                            outputs = fold_model.forward_from_embeddings(embeddings, metas)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            with torch.no_grad():
                                grad = embeddings.grad
                                if grad is not None:
                                    embeddings = embeddings + 0.01 * grad.sign()
                                embeddings = embeddings.clone().detach().requires_grad_(True)
                            fold_model.zero_grad()
                        outputs = fold_model.forward_from_embeddings(embeddings, metas)
                    else:
                        outputs = fold_model(payloads, metas)
                        
                    loss = criterion(outputs, labels)
                    loss.backward()
                    fold_optimizer.step()
            
            fold_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for payloads, metas, labels in fold_val_loader:
                    payloads = payloads.to(device)
                    metas = metas.to(device)
                    labels = labels.to(device)
                    outputs = fold_model(payloads, metas)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            fold_acc = correct / total
            print(f"Fold {fold+1} Accuracy: {fold_acc:.2%}")
            cv_scores.append(fold_acc)
            
        print(f"\n5-Fold CV Completed. Mean Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
        return
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nStarting training loop...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for payloads, metas, labels in train_loader:
            payloads = payloads.to(device)
            metas = metas.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            if args.robust_train:
                embeddings = model.embedding(payloads.long()).clone().detach().requires_grad_(True)
                for _ in range(5):
                    outputs = model.forward_from_embeddings(embeddings, metas)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    with torch.no_grad():
                        grad = embeddings.grad
                        if grad is not None:
                            embeddings = embeddings + 0.01 * grad.sign()
                        embeddings = embeddings.clone().detach().requires_grad_(True)
                    model.zero_grad()
                outputs = model.forward_from_embeddings(embeddings, metas)
            else:
                outputs = model(payloads, metas)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * payloads.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for payloads, metas, labels in val_loader:
                payloads = payloads.to(device)
                metas = metas.to(device)
                labels = labels.to(device)
                
                outputs = model(payloads, metas)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * payloads.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2%} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2%}")
              
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_name = 'gnn_model.pth' if args.model_type == 'mamba_attention' else 'model.pth'
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, save_name))
            print(f"  --> Best model weights saved as {save_name}!")
            
    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2%}")


if __name__ == '__main__':
    main()

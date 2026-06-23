import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation for multi-class classification.
        alpha: Tensor of shape (num_classes,) containing weights for each class.
        gamma: Focusing parameter to balance easy and hard examples.
        reduction: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute unweighted cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of correct class prediction
        
        # Calculate focal loss scale factor
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # Map class weights to targets
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PayloadCNNBiLSTMBERT(nn.Module):
    def __init__(self, num_classes=6):
        """
        CNN-BiLSTM-Transformer (BERT-style) classifier for packet payload data.
        num_classes: Number of threat classes (default: 6 for our common labels).
        """
        super(PayloadCNNBiLSTMBERT, self).__init__()
        # 1. Byte Embedding Layer (256 possible byte values to 32 dimensions)
        self.embedding = nn.Embedding(256, 32)
        
        # 2. CNN Block (extracts local packet header/signature sequences)
        self.conv1d = nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 2)
        
        # 3. BiLSTM Block (captures sequential and temporal patterns)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 4. Transformer Encoder (BERT Component)
        # Sequence input dimension is 128 (hidden_size 64 * 2 directions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 5. Classifier Fully Connected Layers
        # Input size: 128 (Transformer representation) + 4 (Metadata: ttl, total_len, t_delta, protocol)
        self.fc = nn.Sequential(
            nn.Linear(128 + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, payload, metadata):
        """
        Forward pass.
        payload: (batch_size, 1500) tensor of byte values
        metadata: (batch_size, 4) tensor of scaled network metadata features
        """
        # Embed and transpose for Conv1d
        x = self.embedding(payload.long())       # (batch_size, 1500, 32)
        x = x.transpose(1, 2)                      # (batch_size, 32, 1500)
        
        # Convolution, activation, and pooling
        x = self.conv1d(x)                        # (batch_size, 64, 375)
        x = self.relu(x)
        x = self.pool(x)                          # (batch_size, 64, 187)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)                      # (batch_size, 187, 64)
        
        # LSTM sequence processing
        lstm_out, _ = self.lstm(x)                 # (batch_size, 187, 128)
        
        # Transformer (BERT) self-attention
        trans_out = self.transformer(lstm_out)    # (batch_size, 187, 128)
        
        # Extract features of the final sequence element
        features = trans_out[:, -1, :]             # (batch_size, 128)
        
        # Concatenate payload features with metadata features
        combined = torch.cat((features, metadata), dim=1) # (batch_size, 132)
        
        # Classification head
        out = self.fc(combined)                    # (batch_size, num_classes)
        return out

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

    def forward_from_embeddings(self, x, metadata):
        """
        Forward pass starting from continuous byte embedding vectors.
        x: (batch_size, 1500, 32) continuous embedding tensor
        metadata: (batch_size, 4) metadata tensor
        """
        x = x.transpose(1, 2)                      # (batch_size, 32, 1500)
        x = self.conv1d(x)                        # (batch_size, 64, 375)
        x = self.relu(x)
        x = self.pool(x)                          # (batch_size, 64, 187)
        x = x.transpose(1, 2)                      # (batch_size, 187, 64)
        lstm_out, _ = self.lstm(x)                 # (batch_size, 187, 128)
        trans_out = self.transformer(lstm_out)    # (batch_size, 187, 128)
        features = trans_out[:, -1, :]             # (batch_size, 128)
        combined = torch.cat((features, metadata), dim=1) # (batch_size, 132)
        out = self.fc(combined)                    # (batch_size, num_classes)
        return out


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Activation
        self.act = nn.SiLU()
        
        # Selective projections (input-dependent B, C, Delta)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, self.d_inner, bias=True)
        
        # SSM parameters (A)
        A_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # 1. Project input
        projected = self.in_proj(x)
        x_proj, res = projected.chunk(2, dim=-1)
        
        # 2. 1D Depthwise Convolution over time
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj)[:, :, :seq_len]
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.act(x_proj)
        
        # 3. Selective SSM Scan
        A = -torch.exp(self.A_log)
        
        # Project x_proj to B, C, Delta
        x_proj_proj = self.x_proj(x_proj)
        B, C, dt = torch.split(x_proj_proj, [self.d_state, self.d_state, self.d_model], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        
        # Recurrent scan loop in PyTorch
        y = torch.zeros_like(x_proj)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            x_t = x_proj[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]
            
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y[:, t, :] = torch.sum(h * C_t.unsqueeze(1), dim=-1)
            
        # 4. Multiply by residual path and project out
        out = y * self.act(res)
        out = self.out_proj(out)
        return out


class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super(MultiScaleAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Key, Query, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Scaling projection for multi-scale context
        self.scale_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Multi-scale feature enrichment
        x_scaled = x.transpose(1, 2)
        x_scaled = self.scale_conv(x_scaled).transpose(1, 2)
        x_combined = x + x_scaled
        
        q = self.q_proj(x_combined).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_combined).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_combined).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        return self.out_proj(context)


class PayloadMambaAttentionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        """
        AegisNet Hybrid Mamba-Attention Classifier.
        Combines Depthwise Separable Convolutions, Selective Scan Mamba SSM blocks,
        and Multi-Scale Self-Attention for optimal threat parsing.
        """
        super(PayloadMambaAttentionClassifier, self).__init__()
        self.embedding = nn.Embedding(256, 32)
        
        # Depthwise Separable Conv1D
        self.conv1d = DepthwiseSeparableConv1d(32, 64, kernel_size=7, stride=4, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 2)
        
        # Mamba SSM Block
        self.mamba = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2)
        
        # Multi-Scale Attention Block
        self.attention = MultiScaleAttention(d_model=64, n_heads=4)
        
        # Classifier Head (Fused with scaled metadata features)
        self.fc = nn.Sequential(
            nn.Linear(64 + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, payload, metadata):
        x = self.embedding(payload.long())         # (batch, 1500, 32)
        x = x.transpose(1, 2)                      # (batch, 32, 1500)
        
        x = self.conv1d(x)                        # (batch, 64, 375)
        x = self.relu(x)
        x = self.pool(x)                          # (batch, 64, 187)
        x = x.transpose(1, 2)                      # (batch, 187, 64)
        
        x_mamba = self.mamba(x)                   # (batch, 187, 64)
        x_attn = self.attention(x_mamba)          # (batch, 187, 64)
        
        # Global Pooling (extract last time step)
        features = x_attn[:, -1, :]               # (batch, 64)
        
        # Fuse with scaled metadata features
        combined = torch.cat((features, metadata), dim=1) # (batch, 68)
        
        out = self.fc(combined)
        return out

    def forward_from_embeddings(self, x, metadata):
        """
        Forward pass starting from continuous byte embedding vectors.
        x: (batch_size, 1500, 32) continuous embedding tensor
        metadata: (batch_size, 4) metadata tensor
        """
        x = x.transpose(1, 2)                      # (batch, 32, 1500)
        x = self.conv1d(x)                        # (batch, 64, 375)
        x = self.relu(x)
        x = self.pool(x)                          # (batch, 64, 187)
        x = x.transpose(1, 2)                      # (batch, 187, 64)
        x_mamba = self.mamba(x)                   # (batch, 187, 64)
        x_attn = self.attention(x_mamba)          # (batch, 187, 64)
        features = x_attn[:, -1, :]               # (batch, 64)
        combined = torch.cat((features, metadata), dim=1) # (batch, 68)
        out = self.fc(combined)
        return out


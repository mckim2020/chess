import torch
import torch.nn as nn
import torch.nn.functional as F
import math


STATE_DIM = 64
ACTION_SPACE = 4096

HIDDEN_PLANES = 256
NUM_RES_BLOCKS = 16
SUPPORT_SIZE = 601


# Transformer MuZero Network
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=8, width=8):
        super().__init__()
        self.d_model = d_model
        # Create learned embeddings for ranks and files
        self.row_embed = nn.Parameter(torch.randn(height, d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(width, d_model // 2))

    def forward(self, x):
        # x shape: (Batch, 64, d_model)
        batch_size = x.size(0)
        # Expand row/col embeds to match 8x8 grid
        r = self.row_embed.unsqueeze(1).repeat(1, 8, 1) # (8, 8, d/2)
        c = self.col_embed.unsqueeze(0).repeat(8, 1, 1) # (8, 8, d/2)
        pos = torch.cat([r, c], dim=-1).view(64, self.d_model) # (64, d_model)
        return x + pos.unsqueeze(0)


class TransformerMuZero(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        
        # 1. h: Representation (12 piece-planes -> 64 tokens)
        self.obs_projector = nn.Linear(12, d_model)
        self.pos_encoder = PositionalEncoding2D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.repr_transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 2. g: Dynamics (State Tokens + Action -> Next Tokens)
        self.action_embedding = nn.Embedding(4096, d_model)
        self.dyn_transformer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.reward_head = nn.Linear(d_model, 1)

        # 3. f: Prediction (State Tokens -> Policy/Value)
        self.policy_head = nn.Linear(d_model, 4096)
        self.value_head = nn.Linear(d_model, 1)

    def h(self, obs):
        # obs: (B, 12, 8, 8) -> reshape to (B, 64, 12)
        b, c, h, w = obs.shape
        x = obs.view(b, c, h*w).permute(0, 2, 1) 
        x = self.obs_projector(x)
        x = self.pos_encoder(x)
        return self.repr_transformer(x) # Returns (B, 64, d_model)

    def g(self, state, action):
        # action: (B, 1) -> embed and prepend to state sequence
        a_emb = self.action_embedding(action) # (B, 1, d_model)
        combined = torch.cat([a_emb, state], dim=1) # (B, 65, d_model)
        
        # Process through dynamics transformer
        out = self.dyn_transformer(combined)
        next_state = out[:, 1:, :] # Take the 64 squares back
        reward = self.reward_head(out[:, 0, :]) # Use action token for reward prediction
        return next_state, reward

    def f(self, state):
        # Aggregate the 64 square tokens (Mean Pooling)
        latent_vector = state.mean(dim=1) 
        policy_logits = self.policy_head(latent_vector)
        value = torch.tanh(self.value_head(latent_vector))
        return policy_logits, value


# Simple MuZero Network
class MuZeroNet(nn.Module):
    """
    Implements the three core functions of MuZero:
    h: Representation (Obs -> Hidden State)
    g: Dynamics (State + Action -> Next State, Reward)
    f: Prediction (State -> Policy, Value)
    """
    def __init__(self, state_dim=STATE_DIM, action_space=ACTION_SPACE):
        super().__init__()
        # h: Representation Function
        self.repr_fn = nn.Sequential(
            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, state_dim)
        )
        
        # g: Dynamics Function
        self.dyn_state = nn.Linear(state_dim + 1, state_dim)
        self.dyn_reward = nn.Linear(state_dim + 1, 1)
        
        # f: Prediction Function
        self.pred_policy = nn.Linear(state_dim, action_space)
        self.pred_value = nn.Linear(state_dim, 1)

    def h(self, obs):
        return torch.relu(self.repr_fn(obs))

    def g(self, state, action):
        # Normalize action index to [0, 1]
        action_input = action.view(-1, 1).float() / ACTION_SPACE
        combined = torch.cat([state, action_input], dim=1)
        next_state = torch.relu(self.dyn_state(combined))
        reward = self.dyn_reward(combined)
        return next_state, reward

    def f(self, state):
        policy_logits = self.pred_policy(state)
        value = torch.tanh(self.pred_value(state))
        return policy_logits, value


# Residual MuZero Network
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ResMuZeroNet(nn.Module):
    def __init__(self, num_blocks=8, num_channels=64, action_space=4672):
        super().__init__()
        # h: Representation Function
        self.repr_conv = nn.Conv2d(12, num_channels, kernel_size=3, padding=1)
        self.repr_bn = nn.BatchNorm2d(num_channels)
        self.repr_resnet = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        
        # g: Dynamics Function
        # The paper suggests encoding actions as planes. Here we simplify by 
        # concatenating a scalar plane to the state.
        self.dyn_conv = nn.Conv2d(num_channels + 1, num_channels, kernel_size=3, padding=1)
        self.dyn_bn = nn.BatchNorm2d(num_channels)
        self.dyn_resnet = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.dyn_reward = nn.Linear(num_channels * 8 * 8, 1)
        
        # f: Prediction Function
        self.pred_policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1) # Plane reduction
        self.pred_policy_fc = nn.Linear(2 * 8 * 8, action_space)
        self.pred_value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.pred_value_fc = nn.Linear(1 * 8 * 8, 1)

    def h(self, obs):
        """Initial representation: Map observation to hidden state s0."""
        s = F.relu(self.repr_bn(self.repr_conv(obs)))
        s = self.repr_resnet(s)
        return self._normalize_state(s)

    def g(self, state, action):
        """Dynamics: Predict next hidden state and reward."""
        # Action encoding: broadcast action index to an 8x8 plane
        action_plane = (action.view(-1, 1, 1, 1).float() / 4672.0).expand(-1, 1, 8, 8)
        combined = torch.cat([state, action_plane], dim=1)
        
        s_next = F.relu(self.dyn_bn(self.dyn_conv(combined)))
        s_next = self.dyn_resnet(s_next)
        
        reward = self.dyn_reward(s_next.view(s_next.size(0), -1))
        return self._normalize_state(s_next), reward

    def f(self, state):
        """Prediction: Predict policy (logits) and value (v)."""
        # Policy Head
        p = F.relu(self.pred_policy_conv(state))
        policy_logits = self.pred_policy_fc(p.view(p.size(0), -1))
        
        # Value Head
        v = F.relu(self.pred_value_conv(state))
        value = torch.tanh(self.pred_value_fc(v.view(v.size(0), -1)))
        return policy_logits, value

    def _normalize_state(self, state):
        """Scale hidden state to [0, 1] range to stabilize training."""
        batch_size = state.size(0)
        s_flat = state.view(batch_size, -1)
        s_min = s_flat.min(1, keepdim=True)[0].view(-1, 1, 1, 1)
        s_max = s_flat.max(1, keepdim=True)[0].view(-1, 1, 1, 1)
        return (state - s_min) / (s_max - s_min + 1e-6)


# Transformer MuZero Network
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        # x shape: (batch, 64, embed_dim)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class MuZeroAttentionNet(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, action_space=4672):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional Encoding (Crucial: attention doesn't know where squares are!)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, embed_dim))
        self.piece_embedding = nn.Linear(12, embed_dim)

        # h: Representation (Attention-based)
        self.repr_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        
        # g: Dynamics (Predicts next latent sequence)
        self.action_embedding = nn.Embedding(action_space, embed_dim)
        self.dyn_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.reward_head = nn.Linear(embed_dim * 64, 1)

        # f: Prediction
        self.pred_blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.policy_head = nn.Linear(embed_dim * 64, action_space)
        self.value_head = nn.Linear(embed_dim * 64, 1)

    def h(self, obs):
        # obs: (batch, 12, 8, 8)
        batch_size = obs.shape[0]
        
        # 1. Flatten spatial to tokens: (B, 12, 64) -> (B, 64, 12)
        x = obs.view(batch_size, 12, 64).transpose(1, 2)
        
        # 2. Embed to transformer dim: (B, 64, embed_dim)
        x = self.piece_embedding(x) + self.pos_embedding
        
        # 3. Pass through transformer
        s = self.repr_blocks(x) 
        
        # Ensure it's 3D: (Batch, 64, embed_dim)
        return s

    def g(self, state, action):
        # state: (batch, 64, embed_dim)
        # action: (batch, 1)
        
        # 1. Embed action and match dimensions
        a_emb = self.action_embedding(action) # (batch, 1, embed_dim)
        
        # 2. Add action info to every token in the sequence
        # 'state' is 3D, 'a_emb' is 3D. Result is 3D.
        x = state + a_emb 
        
        # 3. Predict next latent sequence
        next_s = self.dyn_blocks(x)
        
        # 4. Predict reward from the flattened sequence
        reward = self.reward_head(next_s.view(next_s.size(0), -1))
        
        return next_s, reward

    def f(self, state):
        # state: (batch, 64, embed_dim)
        x = self.pred_blocks(state)
        
        # Flatten: (batch, 64 * embed_dim)
        x_flat = x.reshape(x.size(0), -1) 
        
        logits = self.policy_head(x_flat)
        value = torch.tanh(self.value_head(x_flat))
        return logits, value
import torch
import torch.nn as nn


STATE_DIM = 64
ACTION_SPACE = 4096


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
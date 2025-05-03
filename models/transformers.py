import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Injects positional information into input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor: Output with positional encoding added
        """
        return x + self.pe[:x.size(1)]


class TransformerDecisionModel(nn.Module):
    """
    A lightweight Transformer-based agent for sequential decision-making tasks.
    Trained using policy gradient methods.
    """
    def __init__(self, state_dim=4, act_dim=2, hidden_size=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(state_dim, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, act_dim)

    def forward(self, x):
        """
        Forward pass through the Transformer RL agent
        
        Args:
            x (Tensor): Input tensor of shape [B, T, N] (batch, time steps, features)
        Returns:
            Tensor: Action logits over time [T, B, A]
        """
        # Embedding and positional encoding
        x = self.embed(x)                  # [B, T, H]
        x = self.pos_encoder(x)           # Add positional encoding
        x = x.transpose(0, 1)             # [T, B, H] for Transformer
        x = self.transformer(x)           # [T, B, H]
        x = x.transpose(0, 1)             # [B, T, H]
        return x                           # Return full sequence

    def get_action(self, x):
        """
        Sample an action from current policy
        
        Args:
            x (Tensor): Current state vector [N] or batched [1, N]
        Returns:
            int: Action sampled from policy distribution
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            x = self.forward(x)              # [1, T, H]
            logits = x[:, -1]               # Take last step → [1, H]
            action_logits = self.head(logits)  # [1, A]
            action_probs = F.softmax(action_logits, dim=-1)  # [A]
            action = torch.multinomial(action_probs, 1).item()

        return action

# Example Usage
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    
    model = TransformerDecisionModel()
    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32)

    action = model.get_action(state_tensor)
    print(f"Sampled action: {action}")
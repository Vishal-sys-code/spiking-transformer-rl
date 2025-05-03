import torch
import torch.nn as nn
import torch.nn.functional as F
from models.snn import OptimizedSNNModel
from norse.torch import LIFCell
from typing import Optional


class FastVectorLIFCell(nn.Module):
    """
    A vectorized LIF cell that avoids Python loops over time steps.
    Simulates spiking behavior across multiple time steps efficiently.
    """
    def __init__(self, input_size: int, hidden_size: int, alpha: float = 0.9, threshold: float = 1.0):
        super(FastVectorLIFCell, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=False)
        self.lif_cell = LIFCell()
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.alpha = alpha

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq (Tensor): Input tensor of shape [B, T, D] or [T, D]
        Returns:
            Tensor: Output spikes of shape [B, T, H] or [T, H]
        """
        if len(x_seq.shape) == 2:
            x_seq = x_seq.unsqueeze(0)  # Add batch dim if not present

        B, T, D = x_seq.shape
        device = x_seq.device

        # Initialize membrane potentials
        m = torch.zeros(B, self.hidden_size, device=device)

        out_seq = []
        for t in range(T):
            xt = self.fc(x_seq[:, t, :])  # [B, H]
            z, v, i = self.lif_cell(xt, (xt, m, None))  # Simplified LIF step
            out_seq.append(z)

            # Update membrane potential
            m = v * (1 - z) + self.alpha * v * z

        return torch.stack(out_seq, dim=1)  # [B, T, H]


class OptimizedSNNModel(nn.Module):
    """
    Optimized Spiking Neural Network Model for Sequential Decision-Making
    
    Designed for low-latency inference and biological plausibility,
    this model integrates policy gradient learning with spiking neurons.
    """
    def __init__(self,
                 input_size: int = 4,
                 hidden_size: int = 64,
                 output_size: int = 2,
                 timesteps: int = 8):
        super(OptimizedSNNModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.lif_block = FastVectorLIFCell(hidden_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)
        self.timesteps = timesteps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SNN model
        
        Args:
            x (Tensor): Input tensor of shape [B, D] or [B, T, D]
        Returns:
            Tensor: Action logits [B, A]
        """
        B, T, D = x.shape
        x = self.encoder(x)  # [B, T, H]

        # Pass through LIF block
        lif_output = self.lif_block(x)  # [B, T, H]

        # Temporal pooling
        pooled = lif_output.mean(dim=1)  # [B, H]

        # Final action prediction
        logits = self.readout(pooled)  # [B, A]
        return logits

    def get_action(self, x: torch.Tensor) -> int:
        """
        Sample an action from the current state using the SNN policy
        
        Args:
            x (Tensor): Current state [D]
        Returns:
            int: Action sampled from policy
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # [1, D]

        # Expand across time steps
        x = x.repeat(1, self.timesteps, 1)  # [1, T, D]

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            action = torch.multinomial(probs, 1).item()

        return action
    
# Example usage
# Create model
model = OptimizedSNNModel(timesteps=8)

# Example input
state = torch.tensor([0.1, -0.2, 0.3, 0.0], dtype=torch.float32)

# Get action
action = model.get_action(state)
print(f"Sampled action: {action}")
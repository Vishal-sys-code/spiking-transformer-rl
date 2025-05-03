import torch

def poisson_encode(x, time_steps=16):
    """
    Encodes a single input vector into a Poisson-distributed binary spike train.
    
    Parameters:
        x (torch.Tensor): Input tensor of shape [input_size] or [B, input_size]
        time_steps (int): Number of time steps to simulate
        
    Returns:
        torch.Tensor: Binary spike train of shape [time_steps, B, input_size]
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it batched if not already
    
    batch_size, input_size = x.shape
    device = x.device

    # Normalize to [0, 1] per batch item
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)

    # Generate random numbers for each time step
    rate_tensor = x_norm.unsqueeze(0)  # Shape: [1, B, input_size]
    rate_tensor = rate_tensor.expand(time_steps, -1, -1)  # [T, B, input_size]

    # Sample from Bernoulli distribution using the rate tensor
    spikes = (torch.rand_like(rate_tensor) < rate_tensor).float()
    return spikes  # Shape: [T, B, input_size]
import torch
def latency_encode(x, max_time=8):
    norm_x = (x - x.min()) / (x.max() - x.min())
    spike_times = (norm_x * max_time).int()
    spikes = torch.zeros(max_time, *x.shape, dtype=torch.float32)
    for t in range(max_time):
        spikes[t] = (spike_times == t).float()
    return spikes
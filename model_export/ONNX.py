# Save TorchScript model
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.snn import OptimizedSNNModel
from norse.torch import LIFCell
script_model = torch.jit.script(snn_model)
torch.jit.save(script_model, "saved_models/optimized_snn_rl_agent.pth")

# ONNX export example
dummy_input = torch.randn(1, 8, 4)
torch.onnx.export(
    snn_model,
    dummy_input,
    "saved_models/snn_model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
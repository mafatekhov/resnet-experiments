import copy
import os

import torch

from quantized_resnet import QuantizedResNet18
from resnet18 import resnet18
from training_utils import calibrate_model
from utils import load_model, get_data, save_torchscript_model, save_model

random_seed = 0
num_classes = 10
batch_size = 64

train_loader, valid_loader = get_data(batch=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = "saved_models"
model_filename = "resnet18_cifar10.pt"
quantized_model_filename = "resnet18_ptq_cifar10.pt"

model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

model = resnet18(num_classes=10).to(device)
model = load_model(model=model, model_filepath=model_filepath, device='cpu')
quantized_model_jit_filename = "resnet18_ptq_jit_cifar10.pt"
quantized_model_jit_filepath = os.path.join(model_dir, quantized_model_jit_filename)
# Make a copy of the model for layer fusion
fused_model_static = copy.deepcopy(model)
fused_model_static.eval()

# Fuse the model in place rather manually.
fused_model_static = torch.quantization.fuse_modules(fused_model_static, [["conv1", "bn1", "relu"]], inplace=True)
for module_name, module in fused_model_static.named_children():
    if "layer" in module_name:
        for basic_block_name, basic_block in module.named_children():
            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            for sub_block_name, sub_block in basic_block.named_children():
                if sub_block_name == "downsample":
                    torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

static_quantized_model = QuantizedResNet18(model_fp32=fused_model_static)
static_quantization_config = torch.quantization.get_default_qconfig("fbgemm")
static_quantized_model.qconfig = static_quantization_config

torch.quantization.prepare(static_quantized_model, inplace=True)
static_quantized_model.to('cpu')
static_quantized_model.eval()

calibrate_model(model=static_quantized_model, loader=train_loader)
static_quantized_model = torch.quantization.convert(static_quantized_model, inplace=True)

# Save quantized model.
save_torchscript_model(model=static_quantized_model, model_dir=model_dir, model_filename=quantized_model_jit_filename)
save_model(model=static_quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)

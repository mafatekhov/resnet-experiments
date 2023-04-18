"""
Script to perform post-training quantization of renset18 model.
Resnet34 can be quantized in the same manner, for Resnet50 use
"""
import copy
import os

import torch

from quantized_resnet import prepare_resnet50_for_quantization, QuantizedResNet, prepare_model_for_quantization
from resnet18 import resnet50, resnet18
from training_utils import calibrate_model
from utils import load_model, get_data, save_torchscript_model, save_model

random_seed = 0
num_classes = 10
batch_size = 64

train_loader, valid_loader = get_data(batch=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = "saved_models"
resnet50_filename = "resnet50_cifar10.pt"
quantized_resnet50_filename = "resnet50_ptq_cifar10.pt"

resnet50_filepath = os.path.join(model_dir, resnet50_filename)
quantized_resnet50_filepath = os.path.join(model_dir, quantized_resnet50_filename)

resnet50_model = resnet50(num_classes=10).to(device)
resnet50_model = load_model(model=resnet50_model, model_filepath=resnet50_filepath, device='cpu')
quantized_resnet50_jit_filename = "resnet50_ptq_jit_cifar10.pt"
quantized_resnet50_jit_filepath = os.path.join(model_dir, quantized_resnet50_jit_filename)

resnet50_ptq = prepare_resnet50_for_quantization(resnet50_model)
static_quantization_config = torch.quantization.get_default_qconfig("fbgemm")
resnet50_ptq.qconfig = static_quantization_config

torch.quantization.prepare(resnet50_ptq, inplace=True)
calibrate_model(model=resnet50_ptq, loader=train_loader)
resnet50_ptq = torch.quantization.convert(resnet50_ptq, inplace=True)

# Save quantized resnet50.
save_torchscript_model(model=resnet50_ptq, model_dir=model_dir,
                       model_filename=quantized_resnet50_jit_filename)
save_model(model=resnet50_ptq, model_dir=model_dir, model_filename=quantized_resnet50_filename)

model_dir = "saved_models"
model_filename = "resnet18_cifar10.pt"
quantized_model_filename = "resnet18_ptq_cifar10.pt"

model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

model = resnet18(num_classes=10).to(device)
model = load_model(model=model, model_filepath=model_filepath, device='cpu')
quantized_model_jit_filename = "resnet18_ptq_jit_cifar10.pt"
quantized_model_jit_filepath = os.path.join(model_dir, quantized_model_jit_filename)
static_quantized_model = prepare_model_for_quantization(model)
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

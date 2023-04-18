import copy
import os

import torch.nn as nn
import torch.quantization

from train import train_model


class QuantizedResNet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


def prepare_model_for_quantization(fp32_model, device='cpu'):
    fp32_model.to(device)
    fused_model = copy.deepcopy(fp32_model)
    fused_model.eval()
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.ao.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                                                   inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    quantized_model = QuantizedResNet(model_fp32=fused_model)
    return quantized_model


def prepare_resnet50_for_quantization(fp32_model, device='cpu'):
    fp32_model.to(device)
    fused_model = copy.deepcopy(fp32_model)
    fused_model.eval()
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block,
                                                [["conv1", "bn1", "relu1"], ["conv2", "bn2"], ["conv3", "bn3"]],
                                                inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    quantized_model = QuantizedResNet(model_fp32=fused_model)
    return quantized_model


def perform_qat(quantized_model, train_loader, valid_loader, device='cpu', learning_rate=1e-3, num_epochs=10,
                plot_name='qat'):
    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    quantized_model = copy.deepcopy(quantized_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    torch.quantization.prepare_qat(quantized_model, inplace=True)
    print("Training QAT Model...")
    quantized_model.train()
    train_model(quantized_model, train_loader, valid_loader, device,
                learning_rate=learning_rate, num_epochs=num_epochs, plot_name=plot_name)
    quantized_model.to('cpu')
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    return quantized_model

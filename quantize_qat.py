import copy
import os

from train import train_model
from resnet18 import resnet18

import torch

from quantized_resnet import QuantizedResNet
from utils import load_model, get_data, save_model, save_torchscript_model

if __name__ == '__main__':
    random_seed = 42
    num_classes = 10
    batch_size = 64

    train_loader, valid_loader = get_data(batch=batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dir = "saved_models"
    model_filename = "resnet18_cifar10.pt"
    qat_model_filename = "resnet18_qat_cifar10.pt"
    qat_jit_model_filename = "resnet18_qat_jit_cifar10.pt"

    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, qat_model_filename)
    # Load model
    model = resnet18(num_classes=10).to(device)
    model_filepath = os.path.join(model_dir, model_filename)
    model = load_model(model=model, model_filepath=model_filepath, device='cpu')
    quantized_model_filepath = os.path.join(model_dir, qat_model_filename)
    model.to(device)
    fused_model = copy.deepcopy(model)

    model.eval()
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

    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    torch.quantization.prepare_qat(quantized_model, inplace=True)
    print("Training QAT Model...")
    quantized_model.train()
    train_model(quantized_model, qat_model_filename, train_loader, valid_loader, device,
                learning_rate=1e-3, num_epochs=10, plot_name='quantized')
    quantized_model.to('cpu')
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    save_model(quantized_model, model_dir, qat_model_filename)
    save_torchscript_model(quantized_model, model_dir, qat_jit_model_filename)
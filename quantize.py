import copy
import os

import torch

from quantized_resnet import QuantizedResNet18
from resnet18_scratch import ResNet, BasicBlock
from train import train_model
from utils import load_model, get_data, save_torchscript_model

random_seed = 0
num_classes = 10
batch_size = 64

train_loader, valid_loader = get_data(batch=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = "saved_models"
model_filename = "resnet18_cifar10.pt"
quantized_model_filename = "resnet18_quantized_cifar10.pt"

model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)

if __name__ == '__main__':
    # Load model
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10)
    model = load_model(model=model, model_filepath=model_filepath, device=device)

    model.to(device)
    fused_model = copy.deepcopy(model)

    # Switch to train mode
    model.eval()
    fused_model.eval()

    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                                                inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # Print FP32 model.
    print('FP32 Model')
    print(model)
    # Print fused model.
    print('Fused Model')
    print(fused_model)

    # Prepare the model for quantization aware training. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedResNet18(model_fp32=fused_model)

    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")

    # Use training data for calibration.
    print("Training QAT Model...")
    quantized_model.train()
    train_model(model=quantized_model, train_loader=train_loader, test_loader=valid_loader, device=device,
                learning_rate=1e-3, num_epochs=10, plot_name='quantized')
    quantized_model.to(device)

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    # Print quantized model.
    print(quantized_model)
    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=quantized_model_filename)
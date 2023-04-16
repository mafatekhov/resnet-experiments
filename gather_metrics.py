import os
import copy

from quantized_resnet import QuantizedResNet18
from training_utils import measure_inference_latency, validate

import torch
from resnet18 import resnet18
from utils import get_data, load_model, load_torchscript_model


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = "saved_models"
model_filename = "resnet18_cifar10.pt"
model_jit_filename = "resnet18_jit_cifar10.pt"
model_filepath = os.path.join(model_dir, model_filename)
model_filepath_jit = os.path.join(model_dir, model_jit_filename)
jit_model = load_torchscript_model(model_filepath=model_filepath_jit, device=device)

ptq_model_filename = "resnet18_ptq_cifar10.pt"
ptq_model_jit_filename = "resnet18_ptq_jit_cifar10.pt"
ptq_model_filepath = os.path.join(model_dir, ptq_model_filename)
ptq_model_jit_filepath = os.path.join(model_dir, ptq_model_jit_filename)

qat_model_filename = "resnet18_qat_cifar10.pt"
qat_model_jit_filename = "resnet18_qat_jit_cifar10.pt"
qat_model_filepath = os.path.join(model_dir, qat_model_filename)
qat_model_jit_filepath = os.path.join(model_dir, qat_model_jit_filename)

batch_size = 64

model = resnet18(num_classes=10).to(device)
model = load_model(model=model, model_filepath=model_filepath, device=device)


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


quantized_model = QuantizedResNet18(model_fp32=fused_model)
quantization_config = torch.quantization.get_default_qconfig("fbgemm")
quantized_model.qconfig = quantization_config

torch.quantization.prepare(quantized_model, inplace=True)
quantized_model.to('cpu')
quantized_model.eval()

quantized_model = torch.quantization.convert(quantized_model, inplace=True)
quantized_model_copy = copy.deepcopy(quantized_model)


ptq_model = load_model(quantized_model, model_filepath=ptq_model_filepath, device='cpu')
ptq_jit_model = load_torchscript_model(model_filepath=ptq_model_jit_filepath, device='cpu')


qat_model = load_model(quantized_model_copy, model_filepath=qat_model_filepath, device='cpu')
qat_jit_model = load_torchscript_model(model_filepath=qat_model_jit_filepath, device='cpu')

_, valid_loader = get_data(batch=batch_size)


fp32_inference_latency = measure_inference_latency(model=model, device='cpu', input_size=(1,3,32,32), num_samples=100)
fp32_jit_inference_latency = measure_inference_latency(model=jit_model, device='cpu', input_size=(1,3,32,32), num_samples=100)

_, fp32_eval_accuracy = validate(model, valid_loader, None, 'cpu')
_, fp32_jit_eval_accuracy = validate(jit_model, valid_loader, None, 'cpu')



static_int8_inference_latency = measure_inference_latency(model=ptq_model, device='cpu', input_size=(1,3,32,32), num_samples=100)
static_int8_jit_inference_latency = measure_inference_latency(model=ptq_jit_model, device='cpu', input_size=(1,3,32,32), num_samples=100)


_, static_int8_eval_accuracy = validate(ptq_model, valid_loader, None, 'cpu')
_, static_int8_jit_eval_accuracy = validate(ptq_jit_model, valid_loader, None, 'cpu')



int8_inference_latency = measure_inference_latency(model=qat_model, device='cpu', input_size=(1,3,32,32), num_samples=100)
int8_jit_inference_latency = measure_inference_latency(model=qat_jit_model, device='cpu', input_size=(1,3,32,32), num_samples=100)


_, int8_eval_accuracy = validate(qat_model, valid_loader, None, 'cpu')
_, int8_jit_eval_accuracy = validate(qat_jit_model, valid_loader, None, 'cpu')




print("FP32 Inference Latency: {:.2f} ms / sample".format(fp32_inference_latency * 1000))
print("FP32 JIT Inference Latency: {:.2f} ms / sample".format(fp32_jit_inference_latency * 1000))

print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
print("FP32 JIT evaluation accuracy: {:.3f}".format(fp32_jit_eval_accuracy))

print("Static INT8 Inference Latency: {:.2f} ms / sample".format(static_int8_inference_latency * 1000))
print("Static INT8 JIT Inference Latency: {:.2f} ms / sample".format(static_int8_jit_inference_latency * 1000))

print("Static INT8 evaluation accuracy: {:.3f}".format(static_int8_eval_accuracy))
print("Static INT8 JIT evaluation accuracy: {:.3f}".format(static_int8_jit_eval_accuracy))

print("INT8 Inference Latency: {:.2f} ms / sample".format(int8_inference_latency * 1000))
print("INT8 JIT Inference Latency: {:.2f} ms / sample".format(int8_jit_inference_latency * 1000))

print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))
print("INT8 JIT evaluation accuracy: {:.3f}".format(int8_jit_eval_accuracy))



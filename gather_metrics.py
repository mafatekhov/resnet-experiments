import os
import copy

from quantized_resnet import prepare_model_for_quantization, prepare_resnet50_for_quantization
from training_utils import measure_inference_latency, validate, dump_to_onnx

import torch
from resnet18 import resnet18, resnet34, resnet50
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

resnet18_qat_model = prepare_model_for_quantization(fused_model)
quantization_config = torch.quantization.get_default_qconfig("fbgemm")
resnet18_qat_model.qconfig = quantization_config

torch.quantization.prepare(resnet18_qat_model, inplace=True)
resnet18_qat_model.to('cpu')
resnet18_qat_model.eval()

resnet18_qat_model = torch.quantization.convert(resnet18_qat_model, inplace=True)
quantized_model_copy = copy.deepcopy(resnet18_qat_model)

resnet18_ptq_model = load_model(resnet18_qat_model, model_filepath=ptq_model_filepath, device='cpu')
resnet18_ptq_jit_model = load_torchscript_model(model_filepath=ptq_model_jit_filepath, device='cpu')

resnet18_qat_model = load_model(quantized_model_copy, model_filepath=qat_model_filepath, device='cpu')
resnet18_qat_jit_model = load_torchscript_model(model_filepath=qat_model_jit_filepath, device='cpu')

_, valid_loader = get_data(batch=batch_size)

resnet18_fp32_inference_latency = measure_inference_latency(model=model, device='cpu', input_size=(1, 3, 32, 32),
                                                            num_samples=100)
resnet18_fp32_jit_inference_latency = measure_inference_latency(model=jit_model, device='cpu',
                                                                input_size=(1, 3, 32, 32), num_samples=100)

_, resnet18_fp32_eval_accuracy = validate(model, valid_loader, None, 'cpu')
_, resnet18_fp32_jit_eval_accuracy = validate(jit_model, valid_loader, None, 'cpu')

resnet18_static_int8_inference_latency = measure_inference_latency(model=resnet18_ptq_model, device='cpu',
                                                                   input_size=(1, 3, 32, 32), num_samples=100)
resnet18_static_int8_jit_inference_latency = measure_inference_latency(model=resnet18_ptq_jit_model, device='cpu',
                                                                       input_size=(1, 3, 32, 32), num_samples=100)

_, resnet18_static_int8_eval_accuracy = validate(resnet18_ptq_model, valid_loader, None, 'cpu')
_, resnet18_static_int8_jit_eval_accuracy = validate(resnet18_ptq_jit_model, valid_loader, None, 'cpu')

resnet18_int8_inference_latency = measure_inference_latency(model=resnet18_qat_model, device='cpu',
                                                            input_size=(1, 3, 32, 32), num_samples=100)
resnet18_int8_jit_inference_latency = measure_inference_latency(model=resnet18_qat_jit_model, device='cpu',
                                                                input_size=(1, 3, 32, 32), num_samples=100)

_, resnet18_int8_eval_accuracy = validate(resnet18_qat_model, valid_loader, None, 'cpu')
_, resnet18_int8_jit_eval_accuracy = validate(resnet18_qat_jit_model, valid_loader, None, 'cpu')

print("FP32 Inference Latency: {:.2f} ms / sample".format(resnet18_fp32_inference_latency * 1000))
print("FP32 JIT Inference Latency: {:.2f} ms / sample".format(resnet18_fp32_jit_inference_latency * 1000))

print("FP32 evaluation accuracy: {:.3f}".format(resnet18_fp32_eval_accuracy))
print("FP32 JIT evaluation accuracy: {:.3f}".format(resnet18_fp32_jit_eval_accuracy))

print("Static INT8 Inference Latency: {:.2f} ms / sample".format(resnet18_static_int8_inference_latency * 1000))
print("Static INT8 JIT Inference Latency: {:.2f} ms / sample".format(resnet18_static_int8_jit_inference_latency * 1000))

print("Static INT8 evaluation accuracy: {:.3f}".format(resnet18_static_int8_eval_accuracy))
print("Static INT8 JIT evaluation accuracy: {:.3f}".format(resnet18_static_int8_jit_eval_accuracy))

print("INT8 Inference Latency: {:.2f} ms / sample".format(resnet18_int8_inference_latency * 1000))
print("INT8 JIT Inference Latency: {:.2f} ms / sample".format(resnet18_int8_jit_inference_latency * 1000))

print("INT8 evaluation accuracy: {:.3f}".format(resnet18_int8_eval_accuracy))
print("INT8 JIT evaluation accuracy: {:.3f}".format(resnet18_int8_jit_eval_accuracy))

# dump to onnx section

fp32_onnx_model = f'{model_filename.split(".")[0]}.onnx'
fp32_jit_onnx_model = f'{model_jit_filename.split(".")[0]}.onnx'
qat_onnx_model = f'{qat_model_filename.split(".")[0]}.onnx'
qat_jit_onnx_model = f'{qat_model_jit_filename.split(".")[0]}.onnx'
ptq_onnx_model = f'{ptq_model_filename.split(".")[0]}.onnx'
ptq_jit_onnx_model = f'{ptq_model_jit_filename.split(".")[0]}.onnx'

dump_to_onnx(model, os.path.join(model_dir, fp32_onnx_model))
dump_to_onnx(jit_model, os.path.join(model_dir, fp32_jit_onnx_model))
dump_to_onnx(resnet18_qat_model, os.path.join(model_dir, qat_onnx_model))
dump_to_onnx(resnet18_qat_jit_model, os.path.join(model_dir, qat_jit_onnx_model))
dump_to_onnx(resnet18_ptq_model, os.path.join(model_dir, ptq_onnx_model))
dump_to_onnx(resnet18_ptq_jit_model, os.path.join(model_dir, ptq_jit_onnx_model))



# resnet 34
resnet34_model_filename = "resnet34_cifar10.pt"
resnet34_model_jit_filename = "resnet34_jit_cifar10.pt"
resnet34_model_filepath = os.path.join(model_dir, resnet34_model_filename)
resnet34_model_filepath_jit = os.path.join(model_dir, resnet34_model_jit_filename)
resnet34_jit_model = load_torchscript_model(model_filepath=resnet34_model_filepath_jit, device=device)

resnet34_ptq_model_filename = "resnet34_ptq_cifar10.pt"
resnet34_ptq_model_jit_filename = "resnet34_ptq_jit_cifar10.pt"
resnet34_ptq_model_filepath = os.path.join(model_dir, resnet34_ptq_model_filename)
resnet34_ptq_model_jit_filepath = os.path.join(model_dir, resnet34_ptq_model_jit_filename)

resnet34_qat_model_filename = "resnet34_qat_cifar10.pt"
resnet34_qat_model_jit_filename = "resnet34_qat_jit_cifar10.pt"
resnet34_qat_model_filepath = os.path.join(model_dir, resnet34_qat_model_filename)
resnet34_qat_model_jit_filepath = os.path.join(model_dir, resnet34_qat_model_jit_filename)

batch_size = 64

resnet34_model = resnet34(num_classes=10).to(device)
resnet34_model = load_model(model=resnet34_model, model_filepath=resnet34_model_filepath, device=device)

resnet34_model.to(device)
resnet34_fused_model = copy.deepcopy(resnet34_model)

resnet34_model.eval()
resnet34_fused_model.eval()

resnet34_qat_model = prepare_model_for_quantization(resnet34_fused_model)
quantization_config = torch.quantization.get_default_qconfig("fbgemm")
resnet34_qat_model.qconfig = quantization_config

torch.quantization.prepare(resnet34_qat_model, inplace=True)
resnet34_qat_model.to('cpu')
resnet34_qat_model.eval()

resnet34_qat_model = torch.quantization.convert(resnet34_qat_model, inplace=True)
quantized_model_copy = copy.deepcopy(resnet34_qat_model)

resnet34_ptq_model = load_model(resnet34_qat_model, model_filepath=resnet34_ptq_model_filepath, device='cpu')
resnet34_ptq_jit_model = load_torchscript_model(model_filepath=resnet34_ptq_model_jit_filepath, device='cpu')

resnet34_qat_model = load_model(quantized_model_copy, model_filepath=resnet34_qat_model_filepath, device='cpu')
resnet34_qat_jit_model = load_torchscript_model(model_filepath=resnet34_qat_model_jit_filepath, device='cpu')

_, valid_loader = get_data(batch=batch_size)

resnet34_fp32_inference_latency = measure_inference_latency(model=resnet34_model, device='cpu', input_size=(1, 3, 32, 32),
                                                            num_samples=100)
resnet34_fp32_jit_inference_latency = measure_inference_latency(model=resnet34_jit_model, device='cpu',
                                                                input_size=(1, 3, 32, 32), num_samples=100)

_, resnet34_fp32_eval_accuracy = validate(resnet34_model, valid_loader, None, 'cpu')
_, resnet34_fp32_jit_eval_accuracy = validate(resnet34_jit_model, valid_loader, None, 'cpu')

resnet34_static_int8_inference_latency = measure_inference_latency(model=resnet34_ptq_model, device='cpu',
                                                                   input_size=(1, 3, 32, 32), num_samples=100)
resnet34_static_int8_jit_inference_latency = measure_inference_latency(model=resnet34_ptq_jit_model, device='cpu',
                                                                       input_size=(1, 3, 32, 32), num_samples=100)

_, resnet34_static_int8_eval_accuracy = validate(resnet34_ptq_model, valid_loader, None, 'cpu')
_, resnet34_static_int8_jit_eval_accuracy = validate(resnet34_ptq_jit_model, valid_loader, None, 'cpu')

resnet34_int8_inference_latency = measure_inference_latency(model=resnet34_qat_model, device='cpu',
                                                            input_size=(1, 3, 32, 32), num_samples=100)
resnet34_int8_jit_inference_latency = measure_inference_latency(model=resnet34_qat_jit_model, device='cpu',
                                                                input_size=(1, 3, 32, 32), num_samples=100)

_, resnet34_int8_eval_accuracy = validate(resnet34_qat_model, valid_loader, None, 'cpu')
_, resnet34_int8_jit_eval_accuracy = validate(resnet34_qat_jit_model, valid_loader, None, 'cpu')

print("RESNET34 FP32 Inference Latency: {:.2f} ms / sample".format(resnet34_fp32_inference_latency * 1000))
print("RESNET34 FP32 JIT Inference Latency: {:.2f} ms / sample".format(resnet34_fp32_jit_inference_latency * 1000))

print("RESNET34 FP32 evaluation accuracy: {:.3f}".format(resnet34_fp32_eval_accuracy))
print("RESNET34 FP32 JIT evaluation accuracy: {:.3f}".format(resnet34_fp32_jit_eval_accuracy))

print("RESNET34 Static INT8 Inference Latency: {:.2f} ms / sample".format(resnet34_static_int8_inference_latency * 1000))
print("RESNET34 Static INT8 JIT Inference Latency: {:.2f} ms / sample".format(resnet34_static_int8_jit_inference_latency * 1000))

print("RESNET34 Static INT8 evaluation accuracy: {:.3f}".format(resnet34_static_int8_eval_accuracy))
print("RESNET34 Static INT8 JIT evaluation accuracy: {:.3f}".format(resnet34_static_int8_jit_eval_accuracy))

print("RESNET34 INT8 Inference Latency: {:.2f} ms / sample".format(resnet34_int8_inference_latency * 1000))
print("RESNET34 INT8 JIT Inference Latency: {:.2f} ms / sample".format(resnet34_int8_jit_inference_latency * 1000))

print("RESNET34 INT8 evaluation accuracy: {:.3f}".format(resnet34_int8_eval_accuracy))
print("RESNET34 INT8 JIT evaluation accuracy: {:.3f}".format(resnet34_int8_jit_eval_accuracy))

# dump to onnx section

resnet34_fp32_onnx_model = f'{resnet34_model_filename.split(".")[0]}.onnx'
resnet34_fp32_jit_onnx_model = f'{resnet34_model_jit_filename.split(".")[0]}.onnx'
resnet34_qat_onnx_model = f'{resnet34_qat_model_filename.split(".")[0]}.onnx'
resnet34_qat_jit_onnx_model = f'{resnet34_qat_model_jit_filename.split(".")[0]}.onnx'
resnet34_ptq_onnx_model = f'{resnet34_ptq_model_filename.split(".")[0]}.onnx'
resnet34_ptq_jit_onnx_model = f'{resnet34_ptq_model_jit_filename.split(".")[0]}.onnx'

dump_to_onnx(resnet34_model, os.path.join(model_dir, resnet34_fp32_onnx_model))
dump_to_onnx(resnet34_jit_model, os.path.join(model_dir, resnet34_fp32_jit_onnx_model))
dump_to_onnx(resnet34_qat_model, os.path.join(model_dir, resnet34_qat_onnx_model))
dump_to_onnx(resnet34_qat_jit_model, os.path.join(model_dir, resnet34_qat_jit_onnx_model))
dump_to_onnx(resnet34_ptq_model, os.path.join(model_dir, resnet34_ptq_onnx_model))
dump_to_onnx(resnet34_ptq_jit_model, os.path.join(model_dir, resnet34_ptq_jit_onnx_model))


# resnet50

resnet50_model_filename = "resnet50_cifar10.pt"
resnet50_model_jit_filename = "resnet50_jit_cifar10.pt"
resnet50_model_filepath = os.path.join(model_dir, resnet50_model_filename)
resnet50_model_filepath_jit = os.path.join(model_dir, resnet50_model_jit_filename)
resnet50_jit_model = load_torchscript_model(model_filepath=resnet50_model_filepath_jit, device=device)

resnet50_ptq_model_filename = "resnet50_ptq_cifar10.pt"
resnet50_ptq_model_jit_filename = "resnet50_ptq_jit_cifar10.pt"
resnet50_ptq_model_filepath = os.path.join(model_dir, resnet50_ptq_model_filename)
resnet50_ptq_model_jit_filepath = os.path.join(model_dir, resnet50_ptq_model_jit_filename)

resnet50_qat_model_filename = "resnet50_qat_cifar10.pt"
resnet50_qat_model_jit_filename = "resnet50_qat_jit_cifar10.pt"
resnet50_qat_model_filepath = os.path.join(model_dir, resnet50_qat_model_filename)
resnet50_qat_model_jit_filepath = os.path.join(model_dir, resnet50_qat_model_jit_filename)

batch_size = 64

resnet50_model = resnet50(num_classes=10).to(device)
resnet50_model = load_model(model=resnet50_model, model_filepath=resnet50_model_filepath, device=device)

resnet50_model.to(device)

resnet50_model.eval()

resnet50_quantized_model = prepare_resnet50_for_quantization(resnet50_model)
quantization_config = torch.quantization.get_default_qconfig("fbgemm")
resnet50_quantized_model.qconfig = quantization_config

torch.quantization.prepare(resnet50_quantized_model, inplace=True)
resnet50_quantized_model.to('cpu')
resnet50_quantized_model.eval()

resnet50_quantized_model = torch.quantization.convert(resnet50_quantized_model, inplace=True)
quantized_model_copy = copy.deepcopy(resnet50_quantized_model)

resnet50_ptq_model = load_model(resnet50_quantized_model, model_filepath=resnet50_ptq_model_filepath, device='cpu')
resnet50_ptq_jit_model = load_torchscript_model(model_filepath=resnet50_ptq_model_jit_filepath, device='cpu')

resnet50_qat_model = load_model(quantized_model_copy, model_filepath=resnet50_qat_model_filepath, device='cpu')
resnet50_qat_jit_model = load_torchscript_model(model_filepath=resnet50_qat_model_jit_filepath, device='cpu')

_, valid_loader = get_data(batch=batch_size)

resnet50_fp32_inference_latency = measure_inference_latency(model=resnet50_model, device='cpu', input_size=(1, 3, 32, 32),
                                                            num_samples=100)
resnet50_fp32_jit_inference_latency = measure_inference_latency(model=resnet50_jit_model, device='cpu',
                                                                input_size=(1, 3, 32, 32), num_samples=100)

_, resnet50_fp32_eval_accuracy = validate(resnet50_model, valid_loader, None, 'cpu')
_, resnet50_fp32_jit_eval_accuracy = validate(resnet50_jit_model, valid_loader, None, 'cpu')

resnet50_static_int8_inference_latency = measure_inference_latency(model=resnet50_ptq_model, device='cpu',
                                                                   input_size=(1, 3, 32, 32), num_samples=100)
resnet50_static_int8_jit_inference_latency = measure_inference_latency(model=resnet50_ptq_jit_model, device='cpu',
                                                                       input_size=(1, 3, 32, 32), num_samples=100)

_, resnet50_static_int8_eval_accuracy = validate(resnet50_ptq_model, valid_loader, None, 'cpu')
_, resnet50_static_int8_jit_eval_accuracy = validate(resnet50_ptq_jit_model, valid_loader, None, 'cpu')

resnet50_int8_inference_latency = measure_inference_latency(model=resnet50_qat_model, device='cpu',
                                                            input_size=(1, 3, 32, 32), num_samples=100)
resnet50_int8_jit_inference_latency = measure_inference_latency(model=resnet50_qat_jit_model, device='cpu',
                                                                input_size=(1, 3, 32, 32), num_samples=100)

_, resnet50_int8_eval_accuracy = validate(resnet50_qat_model, valid_loader, None, 'cpu')
_, resnet50_int8_jit_eval_accuracy = validate(resnet50_qat_jit_model, valid_loader, None, 'cpu')

print("resnet50 FP32 Inference Latency: {:.2f} ms / sample".format(resnet50_fp32_inference_latency * 1000))
print("resnet50 FP32 JIT Inference Latency: {:.2f} ms / sample".format(resnet50_fp32_jit_inference_latency * 1000))

print("resnet50 FP32 evaluation accuracy: {:.3f}".format(resnet50_fp32_eval_accuracy))
print("resnet50 FP32 JIT evaluation accuracy: {:.3f}".format(resnet50_fp32_jit_eval_accuracy))

print("resnet50 Static INT8 Inference Latency: {:.2f} ms / sample".format(resnet50_static_int8_inference_latency * 1000))
print("resnet50 Static INT8 JIT Inference Latency: {:.2f} ms / sample".format(resnet50_static_int8_jit_inference_latency * 1000))

print("resnet50 Static INT8 evaluation accuracy: {:.3f}".format(resnet50_static_int8_eval_accuracy))
print("resnet50 Static INT8 JIT evaluation accuracy: {:.3f}".format(resnet50_static_int8_jit_eval_accuracy))

print("resnet50 INT8 Inference Latency: {:.2f} ms / sample".format(resnet50_int8_inference_latency * 1000))
print("resnet50 INT8 JIT Inference Latency: {:.2f} ms / sample".format(resnet50_int8_jit_inference_latency * 1000))

print("resnet50 INT8 evaluation accuracy: {:.3f}".format(resnet50_int8_eval_accuracy))
print("resnet50 INT8 JIT evaluation accuracy: {:.3f}".format(resnet50_int8_jit_eval_accuracy))

# dump to onnx section

resnet50_fp32_onnx_model = f'{resnet50_model_filename.split(".")[0]}.onnx'
resnet50_fp32_jit_onnx_model = f'{resnet50_model_jit_filename.split(".")[0]}.onnx'
resnet50_qat_onnx_model = f'{resnet50_qat_model_filename.split(".")[0]}.onnx'
resnet50_qat_jit_onnx_model = f'{resnet50_qat_model_jit_filename.split(".")[0]}.onnx'
resnet50_ptq_onnx_model = f'{resnet50_ptq_model_filename.split(".")[0]}.onnx'
resnet50_ptq_jit_onnx_model = f'{resnet50_ptq_model_jit_filename.split(".")[0]}.onnx'

dump_to_onnx(resnet50_model, os.path.join(model_dir, resnet50_fp32_onnx_model))
dump_to_onnx(resnet50_jit_model, os.path.join(model_dir, resnet50_fp32_jit_onnx_model))
dump_to_onnx(resnet50_quantized_model, os.path.join(model_dir, resnet50_qat_onnx_model))
dump_to_onnx(resnet50_qat_jit_model, os.path.join(model_dir, resnet50_qat_jit_onnx_model))
dump_to_onnx(resnet50_ptq_model, os.path.join(model_dir, resnet50_ptq_onnx_model))
dump_to_onnx(resnet50_ptq_jit_model, os.path.join(model_dir, resnet50_ptq_jit_onnx_model))
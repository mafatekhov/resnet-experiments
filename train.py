import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

from resnet18_scratch import ResNet, BasicBlock
from resnet18 import resnet18
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data, save_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision', 'cifar10']
)
args = vars(parser.parse_args())
# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
# TODO parametrize
model_dir = "saved_models"
model_filename = "resnet18_cifar10.pt"

# Learning and training parameters.
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = get_data(batch=batch_size)
# Define model based on the argument parser string.
model = None
plot_name = None
if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
    plot_name = 'resnet_scratch'
if args['model'] == 'torchvision':
    print('[INFO]: Training the Torchvision ResNet18 model...')
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device)
    plot_name = 'resnet_torchvision'
if args['model'] == 'cifar10':
    print('[INFO]: Training the cifar10 ResNet18 model...')
    model = resnet18(num_classes=10).to(device)
    plot_name = 'resnet_cifar10'
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


def train_model(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=20, plot_name='scratch'):
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    # Optimizer.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {num_epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model,
            test_loader,
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)

    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Save the loss and accuracy plots.
    save_plots(
        train_acc,
        valid_acc,
        train_loss,
        valid_loss,
        name=plot_name
    )
    print('TRAINING COMPLETE')


if __name__ == '__main__':
    train_model(model, train_loader, valid_loader, device, plot_name=plot_name)
    print('TRAINING COMPLETE')

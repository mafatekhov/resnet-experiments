import matplotlib as mpl
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

plt.style.use('ggplot')
# mpl.use('TkAgg')


def get_data(batch=64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=train_transform
    )

    dataset_valid = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_valid)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch,
        sampler=train_sampler
    )

    test_loader = DataLoader(
        dataset=dataset_valid,
        batch_size=batch,
        sampler=test_sampler
    )

    return train_loader, test_loader


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(f'outputs_{name}_accuracy.png'))

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(f'outputs_{name}_loss.png'))


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    return model


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

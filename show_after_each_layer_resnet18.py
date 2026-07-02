# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch

if torch.cuda.is_available():
    print(f"Nombre de GPU disponibles : {torch.cuda.device_count()}\n")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} : {torch.cuda.get_device_name(i)}")
        print(f"  Device disponible : {torch.cuda.get_device_properties(i).name}")
        print(f"  Capacité de calcul : {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        print(f"  Mémoire totale : {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Mémoire utilisée : {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Mémoire libre : {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB\n")
else:
    print("Aucun GPU disponible")

# %%
import torch

device = torch.device("cuda:0")

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

c = a @ b

torch.cuda.synchronize()

print(c.sum())
print("OK GPU computation")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import math

cudnn.benchmark = True
plt.ion()

# %%
data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = "barcelona"
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transforms[x])
                   for x in ['train', 'valid', 'test']}
dataloaders = { x : torch.utils.data.DataLoader(image_datasets[x], batch_size = 4,
                                               shuffle = True, num_workers = 4)
                   for x in ['train', 'valid', 'test']}

dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# %%
#loading the model:
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 8)
model.load_state_dict(torch.load("barcelona_resnet18.pth", map_location="cpu"))


# %%
sample_number = 100
path, label = image_datasets['valid'].samples[sample_number]
original_image = Image.open(path)

# %%
transform_image = image_datasets['valid'][sample_number][0]


# %%
def run_block(block, x):
    identity = x
    out = block.conv1(x)
    out = block.bn1(out)
    out = torch.relu(out)
    out = block.conv2(out)
    out = block.bn2(out)
    if block.downsample is not None:
        identity = block.downsample(x)
    out = out + identity
    out = torch.relu(out)
    return out


# %%
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# %%
def show_layer(layer, title=None, cmap='viridis'):
    t = layer.detach().cpu()
    H, W = t.shape[-2], t.shape[-1]
    maps = t.reshape(-1, H, W)
    n = maps.shape[0]
    vmin, vmax = maps.min().item(), maps.max().item()   # ONE scale for all maps
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.4, rows*1.4))
    axes = axes.flatten()
    for i in range(rows*cols):
        axes[i].axis('off')
        if i < n:
            axes[i].imshow(maps[i].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)  # shared vmin/vmax
    if title: fig.suptitle(title)
    plt.tight_layout(); plt.show()


# %%
def show_vector(v, title=None, labels=None):
    v = v.detach().cpu().flatten().numpy()
    plt.figure(figsize=(max(6, len(v) * 0.02), 3))
    plt.bar(range(len(v)), v)
    if labels: plt.xticks(range(len(v)), labels, rotation=45, ha='right')
    if title: plt.title(title)
    plt.tight_layout(); plt.show()


# %%
model.eval()   # use trained running stats — and single-image BN is a non-issue in eval

x = transform_image[None]   # [1, 3, 224, 224]

with torch.no_grad():
    x_conv1   = model.conv1(x)          # [1, 64, 112, 112]
    x_bn1     = model.bn1(x_conv1)      # [1, 64, 112, 112]
    x_relu    = torch.relu(x_bn1)       # [1, 64, 112, 112]
    x_maxpool = model.maxpool(x_relu)   # [1, 64,  56,  56]

    # ---- residual blocks ----
    l1_0 = run_block(model.layer1[0], x_maxpool)  # [1,  64, 56, 56]
    l1_1 = run_block(model.layer1[1], l1_0)       # [1,  64, 56, 56]
    l2_0 = run_block(model.layer2[0], l1_1)       # [1, 128, 28, 28]
    l2_1 = run_block(model.layer2[1], l2_0)       # [1, 128, 28, 28]
    l3_0 = run_block(model.layer3[0], l2_1)       # [1, 256, 14, 14]
    l3_1 = run_block(model.layer3[1], l3_0)       # [1, 256, 14, 14]
    l4_0 = run_block(model.layer4[0], l3_1)       # [1, 512,  7,  7]
    l4_1 = run_block(model.layer4[1], l4_0)       # [1, 512,  7,  7]

    x_avg  = model.avgpool(l4_1)        # [1, 512, 1, 1]
    x_flat = torch.flatten(x_avg, 1)    # [1, 512]
    logits = model.fc(x_flat)           # [1, 8]

# %%
plt.figure()
plt.imshow(original_image)
plt.title(class_names[label])
plt.axis('off')
plt.show()
plt.figure()
imshow(transform_image, title=class_names[label])
plt.show()

show_layer(x_conv1,   title="conv1")
show_layer(x_bn1,     title="bn1")
show_layer(x_relu,    title="after relu")     # sparser/darker — ReLU zeroing negatives
show_layer(x_maxpool, title="maxpool")
show_layer(l1_1,      title="layer1")
show_layer(l2_1,      title="layer2")         # 128 maps, half the resolution
show_layer(l3_1,      title="layer3")         # 256 maps
show_layer(l4_1,      title="layer4")         # 512 maps, 7x7

show_vector(x_avg,  title="avgpool — 512 feature strengths")
show_vector(logits, title="logits — score per class", labels=class_names)

# %%

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

device = torch.device("cuda:1")

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

c = a @ b

torch.cuda.synchronize()

print(c.sum())
print("OK GPU computation")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import os
import math
from PIL import Image

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


# %%
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 24, 3)
        self.conv4 = nn.Conv2d(24, 38, 3)
        self.conv5 = nn.Conv2d(38, 64, 3)

        self.fc1 = nn.LazyLinear(350)
        self.fc2 = nn.LazyLinear(120)
        self.fc3 = nn.LazyLinear(8)

        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, input):
        c1 = F.relu(self.conv1(input))
        c2 = F.relu(self.conv2(c1))
        c2 = self.pool(c2)
        c2 = self.drop1(c2)
        c3 = F.relu(self.conv3(c2))
        c3 = self.pool(c3)
        c4 = F.relu(self.conv4(c3))
        c4 = self.pool(c4)
        c4 = self.drop1(c4)
        c5 = F.relu(self.conv5(c4))
        c5 = self.pool(c5)
        c6 = torch.flatten(c5, 1)
        c7 = self.drop2(c6)
        c8 = self.fc1(c7)
        C8 = self.drop2(c8)
        c9 = self.fc2(c8)
        c9 = self.drop2(c9)
        f = self.fc3(c9)
        return f


# %%
GB_MEAN = np.array([0.87403, 0.74848, 0.72027])
GB_STD = np.array([0.16162, 0.18585, 0.07877])

gb_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(200),
    v2.CenterCrop(180),
    v2.Normalize(GB_MEAN.tolist(), GB_STD.tolist()),
])


# %%
data_dir = "barcelona"
gb_dataset = datasets.ImageFolder(os.path.join(data_dir, "valid"), gb_transform)
classes = gb_dataset.classes


# %%
CNN_GB = CNN().to(device)

# LazyLinear must see one real forward before its weights exist, otherwise
# load_state_dict has no shape to load into. One dummy pass materializes them.
_ = CNN_GB(torch.randn(1, 3, 180, 180, device=device))

CNN_GB.load_state_dict(torch.load("best_model_params.pt",
                                  map_location=device, weights_only=True))


# %%
def imshow(inp, title=None):
    """Display an ImageNet-normalized tensor. (kept as-is from the ResNet notebook)"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def show_layer(layer, title=None, cmap='viridis'):
    t = layer.detach().cpu()
    H, W = t.shape[-2], t.shape[-1]
    maps = t.reshape(-1, H, W)
    n = maps.shape[0]
    vmin, vmax = maps.min().item(), maps.max().item()   # ONE scale for all maps
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis('off')
        if i < n:
            axes[i].imshow(maps[i].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)  # shared vmin/vmax
    if title:
        fig.suptitle(title)
    plt.tight_layout(); plt.show()


def show_vector(v, title=None, labels=None):
    v = v.detach().cpu().flatten().numpy()
    plt.figure(figsize=(max(6, len(v) * 0.02), 3))
    plt.bar(range(len(v)), v)
    if labels:
        plt.xticks(range(len(v)), labels, rotation=45, ha='right')
    if title:
        plt.title(title)
    plt.tight_layout(); plt.show()


# %%
def imshow_gb(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = GB_STD * inp + GB_MEAN
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# %%
sample_number = 100
path, label = gb_dataset.samples[sample_number]
original_image = Image.open(path)
transform_image_gb = gb_dataset[sample_number][0]   # [3, 180, 180]


# %%
CNN_GB.eval()
x = transform_image_gb[None].to(device)

with torch.no_grad():
    c1 = F.relu(CNN_GB.conv1(x))
    c2 = F.relu(CNN_GB.conv2(c1)); c2 = CNN_GB.pool(c2)
    c3 = F.relu(CNN_GB.conv3(c2)); c3 = CNN_GB.pool(c3)
    c4 = F.relu(CNN_GB.conv4(c3)); c4 = CNN_GB.pool(c4)
    c5 = F.relu(CNN_GB.conv5(c4)); c5 = CNN_GB.pool(c5)
    flat = torch.flatten(c5, 1)
    f1 = CNN_GB.fc1(flat)
    f2 = CNN_GB.fc2(f1)
    logits_gb = CNN_GB.fc3(f2)


# %%
plt.figure(); plt.imshow(original_image); plt.title(classes[label]); plt.axis('off'); plt.show()
plt.figure(); imshow_gb(transform_image_gb, title=classes[label]); plt.show()

show_layer(c1, title="conv1")
show_layer(c2, title="conv2 + pool")
show_layer(c3, title="conv3 + pool")
show_layer(c4, title="conv4 + pool")
show_layer(c5, title="conv5 + pool")

show_vector(f1,        title="fc1 - 350 features")
show_vector(logits_gb, title="logits - score per class", labels=classes)

# %%

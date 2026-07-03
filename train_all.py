"""
train_all.py — Entraînement unifié de TOUS les modèles + test CPU
=================================================================
Fusion des 5 scripts (CNN_Martin, show_after_each_layer_gab,
show_after_each_layer_resnet18, tests_reseau4-3, transfer_learning).

Ce que fait ce fichier, en UN SEUL lancement (`python train_all.py`) :
  1. Entraîne (ou réutilise) tous les modèles listés dans MODELS_SPEC.
  2. Sauvegarde chacun dans  modeles/<nom>.pt  sous forme d'un *bundle* :
         {'state_dict', 'meta', 'best_val_acc', 'historique'}
     La 'meta' contient l'architecture ET le pré-traitement du modèle
     (taille d'entrée + normalisation), ce qui permet à la démo de
     reproduire EXACTEMENT le bon pré-traitement pour chaque modèle.
  3. Recharge chaque modèle sauvegardé EN CPU et évalue son accuracy
     sur le val-set (le "test sur chaque modèle").

Réutilisation :
  - barcelona_resnet18.pth   -> réutilisé comme  resnet18_finetune
  - best_model_params.pt     -> réutilisé comme  CNN_gab
  (place ces deux fichiers à côté de ce script.)

Options :
  python train_all.py                 # réutilise l'existant, entraîne le reste, puis teste
  python train_all.py --force-retrain # réentraîne TOUT, même les modèles réutilisables
  python train_all.py --only-test     # ne fait que le test CPU des modèles déjà dans modeles/
  python train_all.py --epochs-scale 0.5   # réduit toutes les durées (debug rapide)

Aucune visualisation accessoire (features, espace latent, courbes, grad-cam) :
tout ça vit dans la démo, pas ici.
"""

import os
import re
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MODELS_DIR = "modeles"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"


def _find_data_dir():
    """Trouve le dossier contenant train/ et val/, peu importe le niveau
    d'imbrication après dézippage (barcelona/barcelona, barcelona, ou courant)."""
    for cand in ("barcelona/barcelona", "barcelona", "."):
        if os.path.isdir(os.path.join(cand, VAL_SPLIT)):
            return cand
    return "barcelona/barcelona"


DATA_DIR = _find_data_dir()               # contient train/ et val/

# Poids déjà entraînés à réutiliser (cherchés à côté du script)
REUSE_WEIGHTS = {
    "resnet18_finetune": "barcelona_resnet18.pth",
    "CNN_gab":           "best_model_params.pt",
}

# Normalisations utilisées par les différents modèles d'origine
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GB_MEAN,       GB_STD       = [0.87403, 0.74848, 0.72027], [0.16162, 0.18585, 0.07877]
HALF_MEAN,     HALF_STD     = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

# Device : GPU serveur pour l'entraînement, CPU imposé pour le test final
TRAIN_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TEST_DEVICE = torch.device("cpu")


# ═════════════════════════════════════════════════════════════
# Architectures (reprises telles quelles des 5 fichiers)
# ═════════════════════════════════════════════════════════════
class Net(nn.Module):
    """CNN_Martin — entrée 180x180, normalisation (0.5, 0.5, 0.5)."""
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(11552, 120)     # 32 * 19 * 19  (entrée 180)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        return self.fc3(x)


class CNN_gab(nn.Module):
    """show_after_each_layer_gab — LazyLinear, entrée 180x180, normalisation 'GB'.
    num_classes ignoré (tête figée à 8 dans le modèle d'origine)."""
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 24, 3)
        self.conv4 = nn.Conv2d(24, 38, 3)
        self.conv5 = nn.Conv2d(38, 64, 3)
        self.fc1 = nn.LazyLinear(350)
        self.fc2 = nn.LazyLinear(120)
        self.fc3 = nn.LazyLinear(num_classes)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = self.pool(F.relu(self.conv2(c1)))
        c2 = self.drop1(c2)
        c3 = self.pool(F.relu(self.conv3(c2)))
        c4 = self.pool(F.relu(self.conv4(c3)))
        c4 = self.drop1(c4)
        c5 = self.pool(F.relu(self.conv5(c4)))
        c6 = torch.flatten(c5, 1)
        c7 = self.drop2(c6)
        c8 = self.fc1(c7)
        c9 = self.drop2(c8)
        c9 = self.fc2(c9)
        c9 = self.drop2(c9)
        return self.fc3(c9)


class MonCNN(nn.Module):
    """tests_reseau4-3 — entrée 224, normalisation ImageNet."""
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1);   self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1);  self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1);  self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1); self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


class MonCNN2(nn.Module):
    """tests_reseau4-3 — entrée 224, normalisation ImageNet."""
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ConfigCNN(nn.Module):
    """tests_reseau4-3 — CNN paramétrable (GAP), entrée 224, normalisation ImageNet."""
    def __init__(self, num_classes, channels=(16, 32, 64), fc_dim=128):
        super().__init__()
        self.channels = tuple(channels)
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        blocs, in_c = [], 3
        for out_c in channels:
            blocs += [nn.Conv2d(in_c, out_c, 3, padding=1),
                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                      nn.MaxPool2d(2, 2)]
            in_c = out_c
        self.features = nn.Sequential(*blocs)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(in_c, fc_dim), nn.ReLU(inplace=True),
            nn.Linear(fc_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


def build_resnet18(num_classes, pretrained=False):
    m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def nb_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════
# Constructeur générique depuis une 'meta'
# ═════════════════════════════════════════════════════════════
def build_from_meta(meta):
    arch = meta["arch"]
    nc = meta.get("num_classes", 8)
    if arch == "ResNet18":
        return build_resnet18(nc, pretrained=False)
    if arch == "MonCNN":
        return MonCNN(nc)
    if arch == "MonCNN2":
        return MonCNN2(nc)
    if arch == "ConfigCNN":
        return ConfigCNN(nc, channels=tuple(meta["channels"]),
                         fc_dim=meta.get("fc_dim", 128))
    if arch == "Net":
        return Net(nc)
    if arch == "CNN_gab":
        return CNN_gab(nc)
    raise ValueError(f"Architecture inconnue : {arch}")


def is_lazy(meta):
    return meta["arch"] == "CNN_gab"


def materialize_if_needed(model, meta, device):
    """Les modules Lazy* n'ont de poids qu'après un premier forward."""
    if is_lazy(meta):
        s = meta["input_size"]
        model.to(device)
        model(torch.zeros(1, 3, s, s, device=device))
    return model


# ═════════════════════════════════════════════════════════════
# Pré-traitement décrit par la meta -> transforms torchvision
# ═════════════════════════════════════════════════════════════
def make_transforms(meta, train=False):
    ops = []
    if meta.get("resize"):
        ops.append(transforms.Resize(meta["resize"]))
    ops.append(transforms.CenterCrop(meta["crop"]))
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(meta["mean"], meta["std"])]
    return transforms.Compose(ops)


# Cache des dataloaders : une seule construction par pré-traitement distinct
_LOADER_CACHE = {}


def get_loaders(meta, batch_size=32, num_workers=4):
    key = (meta.get("resize"), meta["crop"], tuple(meta["mean"]), tuple(meta["std"]), batch_size)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    ds = {
        TRAIN_SPLIT: datasets.ImageFolder(os.path.join(DATA_DIR, TRAIN_SPLIT),
                                          make_transforms(meta, train=True)),
        VAL_SPLIT:   datasets.ImageFolder(os.path.join(DATA_DIR, VAL_SPLIT),
                                          make_transforms(meta, train=False)),
    }
    loaders = {
        s: torch.utils.data.DataLoader(ds[s], batch_size=batch_size,
                                       shuffle=(s == TRAIN_SPLIT), num_workers=num_workers)
        for s in (TRAIN_SPLIT, VAL_SPLIT)
    }
    sizes = {s: len(ds[s]) for s in (TRAIN_SPLIT, VAL_SPLIT)}
    classes = ds[TRAIN_SPLIT].classes
    _LOADER_CACHE[key] = (loaders, sizes, classes)
    return _LOADER_CACHE[key]


# ═════════════════════════════════════════════════════════════
# Boucle d'entraînement (train + val, garde le meilleur poids val)
# ═════════════════════════════════════════════════════════════
def train_model(model, loaders, sizes, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    histo = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_epoch = 0.0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(num_epochs):
        for phase in (TRAIN_SPLIT, VAL_SPLIT):
            model.train() if phase == TRAIN_SPLIT else model.eval()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(TRAIN_DEVICE), labels.to(TRAIN_DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN_SPLIT):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == TRAIN_SPLIT:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            if phase == TRAIN_SPLIT and scheduler is not None:
                scheduler.step()
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects / sizes[phase]
            histo[f"{phase}_loss"].append(epoch_loss)
            histo[f"{phase}_acc"].append(epoch_acc)
            if phase == VAL_SPLIT and epoch_acc > best_acc:
                best_acc, best_epoch = epoch_acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch + 1}/{num_epochs}  "
              f"train_acc={histo['train_acc'][-1]:.3f}  val_acc={histo['val_acc'][-1]:.3f}")

    model.load_state_dict(best_state)
    elapsed = time.time() - since
    print(f"    -> best val acc = {best_acc:.4f} (epoch {best_epoch})  |  {elapsed:.0f}s")
    return model, histo, best_acc, best_epoch, elapsed


# ═════════════════════════════════════════════════════════════
# Registre des modèles à produire
# ═════════════════════════════════════════════════════════════
def _spec(resize, crop, mean, std):
    return {"resize": resize, "crop": crop, "mean": mean, "std": std, "input_size": crop}


IMAGENET_224 = _spec(256, 224, IMAGENET_MEAN, IMAGENET_STD)
GB_180       = _spec(200, 180, GB_MEAN, GB_STD)
HALF_180     = _spec(None, 180, HALF_MEAN, HALF_STD)

CONFIGCNN_CHANNELS = [
    (16, 32, 64),
    (32, 64, 128),
    (16, 32, 64, 128, 256),
    (32, 64, 128, 256),
    (8, 16, 32, 64, 128, 256),
    (8, 16, 32, 64, 128),
    (4, 8, 16, 32, 64, 128),
    (2, 4, 8, 16, 32, 64, 128),
    (8, 16, 32, 64, 128, 256, 512),
]


def build_registry():
    reg = []

    # --- Modèles réutilisés (poids déjà entraînés) ---
    reg.append(dict(name="resnet18_finetune", arch="ResNet18", pre=IMAGENET_224,
                    reuse="barcelona_resnet18.pth",
                    note="ResNet18 finetuné (poids réutilisés)"))
    reg.append(dict(name="CNN_gab", arch="CNN_gab", pre=GB_180,
                    reuse="best_model_params.pt",
                    note="CNN LazyLinear (poids réutilisés)"))

    # --- ResNet18 à entraîner ---
    reg.append(dict(name="resnet18_scratch", arch="ResNet18", pre=IMAGENET_224,
                    pretrained=False, opt="adam", lr=1e-3, epochs=25,
                    note="ResNet18 from scratch"))
    reg.append(dict(name="resnet18_featextract", arch="ResNet18", pre=IMAGENET_224,
                    pretrained=True, freeze_backbone=True, opt="sgd", lr=1e-3,
                    momentum=0.9, epochs=15, note="ResNet18 feature-extractor (backbone gelé)"))

    # --- CNN maison ---
    reg.append(dict(name="MonCNN", arch="MonCNN", pre=IMAGENET_224,
                    opt="adam", lr=1e-3, epochs=10, note="CNN maison 4 blocs"))
    reg.append(dict(name="MonCNN2", arch="MonCNN2", pre=IMAGENET_224,
                    opt="adam", lr=1e-3, epochs=15, note="CNN maison 3 blocs"))

    # --- Net (CNN_Martin) ---
    reg.append(dict(name="Net_Martin", arch="Net", pre=HALF_180,
                    opt="adam", lr=1e-3, weight_decay=1e-4, epochs=10,
                    note="Net (CNN_Martin), 180px norm 0.5"))

    # --- Balayage ConfigCNN largeur/profondeur ---
    for ch in CONFIGCNN_CHANNELS:
        reg.append(dict(name="ConfigCNN_" + "-".join(map(str, ch)),
                        arch="ConfigCNN", channels=ch, fc_dim=128, pre=IMAGENET_224,
                        opt="adam", lr=1e-3, epochs=10,
                        note=f"ConfigCNN {ch}"))
    return reg


def make_optimizer(model, spec):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if spec.get("opt") == "sgd":
        return optim.SGD(params, lr=spec.get("lr", 1e-3),
                         momentum=spec.get("momentum", 0.9))
    return optim.Adam(params, lr=spec.get("lr", 1e-3),
                      weight_decay=spec.get("weight_decay", 0.0))


def meta_for(spec, num_classes):
    meta = {
        "arch": spec["arch"],
        "num_classes": num_classes,
        "resize": spec["pre"]["resize"],
        "crop": spec["pre"]["crop"],
        "input_size": spec["pre"]["crop"],
        "mean": spec["pre"]["mean"],
        "std": spec["pre"]["std"],
        "note": spec.get("note", ""),
    }
    if spec["arch"] == "ConfigCNN":
        meta["channels"] = list(spec["channels"])
        meta["fc_dim"] = spec.get("fc_dim", 128)
    if spec["arch"] == "CNN_gab":
        meta["num_classes"] = 8
    return meta


# ═════════════════════════════════════════════════════════════
# Production d'un modèle : réutilisation OU entraînement
# ═════════════════════════════════════════════════════════════
def produce_model(spec, force_retrain, epochs_scale):
    name = spec["name"]
    out_path = os.path.join(MODELS_DIR, f"{name}.pt")

    # Nombre de classes lu depuis les données (une fois via un loader)
    loaders, sizes, classes = get_loaders(spec["pre"])
    num_classes = len(classes)
    meta = meta_for(spec, num_classes)

    reuse_file = spec.get("reuse")
    can_reuse = reuse_file and os.path.exists(reuse_file) and not force_retrain

    if can_reuse:
        print(f"[{name}] réutilisation des poids {reuse_file}")
        model = build_from_meta(meta)
        model = materialize_if_needed(model, meta, TEST_DEVICE)
        sd = torch.load(reuse_file, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd)
        best_acc, histo = None, {}
    else:
        if reuse_file and not os.path.exists(reuse_file):
            print(f"[{name}] {reuse_file} introuvable -> entraînement à la place")
        print(f"[{name}] entraînement ({spec['arch']})")
        pretrained = spec.get("pretrained", False)
        if spec["arch"] == "ResNet18":
            model = build_resnet18(num_classes, pretrained=pretrained)
            if spec.get("freeze_backbone"):
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.fc.parameters():
                    p.requires_grad = True
        else:
            model = build_from_meta(meta)
            model = materialize_if_needed(model, meta, TRAIN_DEVICE)
        model = model.to(TRAIN_DEVICE)
        n_ep = max(1, int(round(spec["epochs"] * epochs_scale)))
        criterion = nn.CrossEntropyLoss()
        optimizer = make_optimizer(model, spec)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        model, histo, best_acc, _, _ = train_model(
            model, loaders, sizes, criterion, optimizer, scheduler, n_ep)

    # Sauvegarde du bundle
    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": meta,
        "best_val_acc": best_acc,
        "historique": histo,
    }
    torch.save(bundle, out_path)
    print(f"[{name}] -> sauvé {out_path}  ({nb_params(model):,} params)")
    return out_path


# ═════════════════════════════════════════════════════════════
# TEST CPU : recharge chaque modèle et calcule l'accuracy val
# ═════════════════════════════════════════════════════════════
@torch.no_grad()
def test_one_cpu(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    meta = bundle["meta"]
    model = build_from_meta(meta)
    model = materialize_if_needed(model, meta, TEST_DEVICE)
    model.load_state_dict(bundle["state_dict"])
    model.eval().to(TEST_DEVICE)

    loaders, sizes, classes = get_loaders(meta, num_workers=0)
    correct = 0
    for inputs, labels in loaders[VAL_SPLIT]:
        preds = model(inputs).argmax(1)
        correct += (preds == labels).sum().item()
    acc = correct / sizes[VAL_SPLIT]
    return meta, nb_params(model), acc


def test_all_cpu():
    print("\n" + "=" * 66)
    print("  TEST CPU — accuracy sur le val-set (pré-traitement propre à chaque modèle)")
    print("=" * 66)
    paths = sorted(os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR)
                   if f.endswith(".pt"))
    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            meta, params, acc = test_one_cpu(p)
            rows.append((name, meta["arch"], params, acc))
            print(f"  {name:<28} {meta['arch']:<12} "
                  f"{params:>12,} params   acc = {acc * 100:5.2f}%")
        except Exception as e:
            print(f"  {name:<28} ERREUR : {e}")
    if rows:
        best = max(rows, key=lambda r: r[3])
        print("-" * 66)
        print(f"  Meilleur : {best[0]}  ({best[3] * 100:.2f}%)")
    print("=" * 66)
    return rows


# ═════════════════════════════════════════════════════════════
# Point d'entrée unique
# ═════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-retrain", action="store_true",
                    help="réentraîne tout, même les modèles réutilisables")
    ap.add_argument("--only-test", action="store_true",
                    help="ne fait que le test CPU des modèles déjà présents dans modeles/")
    ap.add_argument("--epochs-scale", type=float, default=1.0,
                    help="facteur multiplicatif sur toutes les durées d'entraînement")
    args = ap.parse_args()

    print(f"Device d'entraînement : {TRAIN_DEVICE}   |   test en : {TEST_DEVICE}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not args.only_test:
        registry = build_registry()
        t0 = time.time()
        for spec in registry:
            produce_model(spec, force_retrain=args.force_retrain,
                          epochs_scale=args.epochs_scale)
        print(f"\nTous les modèles produits en {(time.time() - t0) / 60:.1f} min.")

    test_all_cpu()
    print("\nTerminé. Lance la démo avec :  python app_demo_cpu.py")


if __name__ == "__main__":
    main()

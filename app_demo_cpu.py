"""
app_demo_cpu.py — Démo interactive (Gradio), 100% CPU, toutes fonctionnalités
=============================================================================
Reprend l'intégralité des fonctionnalités de app_gradio-v3 (analyse interactive,
performance, ESPACE LATENT t-SNE 3D, dashboard comparatif, analyse d'erreurs
avancée, Grad-CAM moyen) — mais :
  - tourne exclusivement sur CPU (pour ne pas encombrer le GPU du serveur) ;
  - applique à CHAQUE modèle son propre pré-traitement (taille + normalisation),
    lu dans la méta du bundle produit par train_all.py. Indispensable : le Net
    (180px, norm 0.5) et le CNN_gab (180px, norm 'GB') n'attendent pas la même
    entrée que les ResNet (224px, ImageNet).

Lancement :
    pip install gradio plotly scikit-learn pandas opencv-python
    python app_demo_cpu.py                 # URL locale
    # (en notebook : demo.launch(share=True) pour un lien public)

Prérequis dans le même dossier :
    - barcelona/barcelona/val/<classe>/*.jpg
    - modeles/*.pt   (bundles créés par train_all.py)
"""

import os
import re
import math
import random
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# ─────────────────────────────────────────────────────────────
# Config — CPU imposé
# ─────────────────────────────────────────────────────────────
VAL_SPLIT = "val"
MODELS_DIR = "modeles"


def _find_data_dir():
    """Trouve le dossier contenant `val/`, peu importe le niveau d'imbrication
    après dézippage (barcelona/barcelona, barcelona, ou dossier courant)."""
    for cand in ("barcelona/barcelona", "barcelona", "."):
        if os.path.isdir(os.path.join(cand, VAL_SPLIT)):
            return cand
    return "barcelona/barcelona"


DATA_DIR = _find_data_dir()
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
PALETTE = plt.get_cmap("tab10")


# ═════════════════════════════════════════════════════════════
# Architectures (identiques à train_all.py)
# ═════════════════════════════════════════════════════════════
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(11552, 120)
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


def _build_from_meta(meta):
    arch, nc = meta["arch"], meta.get("num_classes", 8)
    if arch == "ResNet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, nc)
        return m
    if arch == "MonCNN":
        return MonCNN(nc)
    if arch == "MonCNN2":
        return MonCNN2(nc)
    if arch == "ConfigCNN":
        return ConfigCNN(nc, channels=tuple(meta["channels"]), fc_dim=meta.get("fc_dim", 128))
    if arch == "Net":
        return Net(nc)
    if arch == "CNN_gab":
        return CNN_gab(nc)
    raise ValueError(f"Architecture inconnue : {arch}")


# ═════════════════════════════════════════════════════════════
# Pré-traitement par défaut selon l'architecture
# (utilisé si la méta d'un vieux bundle ne le précise pas)
# ═════════════════════════════════════════════════════════════
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GB_MEAN, GB_STD = [0.87403, 0.74848, 0.72027], [0.16162, 0.18585, 0.07877]
HALF_MEAN, HALF_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

_ARCH_DEFAULTS = {
    "ResNet18":  dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "MonCNN":    dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "MonCNN2":   dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "ConfigCNN": dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD),
    "Net":       dict(resize=None, crop=180, mean=HALF_MEAN, std=HALF_STD),
    "CNN_gab":   dict(resize=200, crop=180, mean=GB_MEAN, std=GB_STD),
}


def _complete_meta(meta):
    """Complète une méta incomplète (anciens bundles sans préprocessing)."""
    d = _ARCH_DEFAULTS.get(meta.get("arch"),
                           dict(resize=256, crop=224, mean=IMAGENET_MEAN, std=IMAGENET_STD))
    if "resize" not in meta:
        meta["resize"] = d["resize"]
    meta.setdefault("crop", d["crop"])
    meta.setdefault("mean", d["mean"])
    meta.setdefault("std", d["std"])
    meta.setdefault("input_size", meta["crop"])
    meta.setdefault("num_classes", 8)
    return meta


# ═════════════════════════════════════════════════════════════
# Chargement des modèles (bundles produits par train_all.py
# — ou anciens bundles tests_reseau, dont la méta est complétée)
# ═════════════════════════════════════════════════════════════
def load_one_model(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(bundle, dict) and "meta" in bundle):
        raise ValueError("bundle sans méta — relance train_all.py pour (re)générer modeles/")
    meta = _complete_meta(dict(bundle["meta"]))
    model = _build_from_meta(meta)
    if meta["arch"] == "CNN_gab":                      # LazyLinear : forward à blanc
        s = meta["input_size"]
        model(torch.zeros(1, 3, s, s))
    model.load_state_dict(bundle["state_dict"])
    model.eval().to(DEVICE)
    return model, meta


def load_all_models(models_dir):
    result = {}
    if not os.path.isdir(models_dir):
        return result
    for fn in sorted(os.listdir(models_dir)):
        if not fn.endswith(".pt"):
            continue
        try:
            model, meta = load_one_model(os.path.join(models_dir, fn))
            name = f"{os.path.splitext(fn)[0]} · {meta['arch']}"
            result[name] = {"model": model, "meta": meta}
        except Exception as e:
            print(f"[ignoré] {fn} : {e}")
    return result


# ═════════════════════════════════════════════════════════════
# Pré-traitement PAR modèle (lu dans la méta)
# ═════════════════════════════════════════════════════════════
DISP_TF = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
_TF_CACHE = {}


def norm_tf(meta):
    key = (meta.get("resize"), meta["crop"], tuple(meta["mean"]), tuple(meta["std"]))
    if key not in _TF_CACHE:
        ops = []
        if meta.get("resize"):
            ops.append(transforms.Resize(meta["resize"]))
        ops += [transforms.CenterCrop(meta["crop"]),
                transforms.ToTensor(),
                transforms.Normalize(meta["mean"], meta["std"])]
        _TF_CACHE[key] = transforms.Compose(ops)
    return _TF_CACHE[key]


def make_input(pil_img, meta):
    """Tenseur [1, 3, H, W] pré-traité selon la méta du modèle."""
    return norm_tf(meta)(pil_img.convert("RGB")).unsqueeze(0)


# --- Cache des tenseurs pré-traités du val-set (évite de redécoder les JPEG) ---
BATCH = 64


def _pp_key(meta):
    return (meta.get("resize"), meta["crop"], tuple(meta["mean"]), tuple(meta["std"]))


@lru_cache(maxsize=4096)
def _cached_input(idx, key):
    """Tenseur [3, H, W] de l'image `idx` du val-set, pour un pré-traitement donné.
    Mémorisé : le 2e appel ne redécode pas le JPEG."""
    resize, crop, mean, std = key
    meta = {"resize": resize, "crop": crop, "mean": list(mean), "std": list(std)}
    return norm_tf(meta)(Image.open(SAMPLES[idx][0]).convert("RGB"))


def batch_iter(order, key, batch_size=BATCH):
    """Rend des batchs [B, 3, H, W] à partir d'indices du val-set (avec cache)."""
    for s in range(0, len(order), batch_size):
        idxs = order[s:s + batch_size]
        yield idxs, torch.stack([_cached_input(i, key) for i in idxs])


def disp_from_pil(pil_img):
    return np.asarray(DISP_TF(pil_img.convert("RGB"))).astype(np.float32) / 255.0


def denorm_from_meta(tensor, meta):
    """Dé-normalise un tenseur [3,H,W] selon la méta (pour affichage)."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.array(meta["std"]) * img + np.array(meta["mean"])
    return np.clip(img, 0, 1)


# ═════════════════════════════════════════════════════════════
# Données
# ═════════════════════════════════════════════════════════════
def load_dataset(data_dir):
    ds = datasets.ImageFolder(os.path.join(data_dir, VAL_SPLIT))
    return ds.classes, ds.samples


# ═════════════════════════════════════════════════════════════
# Grad-CAM + prédiction
# ═════════════════════════════════════════════════════════════
def last_conv_layer(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def grad_cam(model, tensor, class_idx=None):
    target = last_conv_layer(model)
    acts, grads = {}, {}
    h1 = target.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))
    model.zero_grad()
    # requires_grad sur l'entrée : garantit que le graphe existe même si le
    # backbone est gelé (ResNet feature-extractor).
    t = tensor.to(DEVICE).clone().detach().requires_grad_(True)
    logits = model(t)
    if class_idx is None:
        class_idx = logits.argmax(1).item()
    logits[0, class_idx].backward()
    w = grads["v"].mean(dim=(2, 3))
    cam = F.relu((w[:, :, None, None] * acts["v"]).sum(1)).squeeze().detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    h1.remove(); h2.remove()
    return cam, class_idx


def overlay_cam(disp_np, cam, alpha=0.45):
    h, w = disp_np.shape[:2]
    if HAS_CV2:
        heat = cv2.resize(cam, (w, h))
        heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = heat[..., ::-1].astype(np.float32) / 255.0
    else:
        heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h)))
        heat = plt.get_cmap("jet")(heat / 255.0)[..., :3]
    return np.clip((1 - alpha) * disp_np + alpha * heat, 0, 1)


def _resize_cam(cam, size=224):
    if HAS_CV2:
        return cv2.resize(cam, (size, size))
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size))) / 255.0


@torch.no_grad()
def predict(model, tensor):
    return torch.softmax(model(tensor.to(DEVICE)), 1).squeeze().cpu().numpy()


def head_linear(model):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier[-1]
    if hasattr(model, "fc3"):
        return model.fc3
    if hasattr(model, "fc2") and isinstance(model.fc2, nn.Linear):
        return model.fc2
    return None


def nb_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════
# Chargement global (à l'import)
# ═════════════════════════════════════════════════════════════
CLASS_NAMES, SAMPLES = load_dataset(DATA_DIR)
MODELS = load_all_models(MODELS_DIR)
MODEL_NAMES = list(MODELS.keys())
print(f"Dataset : {len(CLASS_NAMES)} classes, {len(SAMPLES)} images val")
print(f"Modèles : {MODEL_NAMES if MODEL_NAMES else 'AUCUN (lance train_all.py)'}")


def _order(n_eval, seed=0):
    order = list(range(len(SAMPLES)))
    random.Random(seed).shuffle(order)
    return order[:int(n_eval)]


# ═════════════════════════════════════════════════════════════
# 1) Analyse interactive — une figure composite (1 ligne / modèle)
# ═════════════════════════════════════════════════════════════
def analyse(pil_img, true_label):
    if not MODELS:
        return None, "⚠️ Aucun modèle chargé. Lance `python train_all.py` d'abord."

    disp_np = disp_from_pil(pil_img)
    n = len(MODELS)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.3 * n))
    axes = np.array(axes).reshape(n, 3)
    lignes = []

    for row, (name, entry) in enumerate(MODELS.items()):
        try:
            model, meta = entry["model"], entry["meta"]
            tensor = make_input(pil_img, meta)
            probs = predict(model, tensor)
            pred = int(probs.argmax()); conf = float(probs[pred])
            pred_name = CLASS_NAMES[pred]
            wrong = (true_label is not None and pred != true_label)
            correct = (true_label is not None and pred == true_label)
            verdict = "✅" if correct else ("❌" if wrong else "•")
            color = "green" if correct else ("red" if wrong else "black")

            cam, _ = grad_cam(model, tensor, class_idx=pred)
            axes[row][0].imshow(overlay_cam(disp_np, cam))
            axes[row][0].set_title(f"{name}\n{verdict} → {pred_name} ({conf * 100:.0f}%)",
                                   color=color, fontsize=9)
            axes[row][0].axis("off")

            if wrong:
                cam_t, _ = grad_cam(model, tensor, class_idx=true_label)
                axes[row][1].imshow(overlay_cam(disp_np, cam_t))
                axes[row][1].set_title(f"Grad-CAM → {CLASS_NAMES[true_label]} (vraie)", fontsize=9)
            else:
                axes[row][1].imshow(disp_np)
                axes[row][1].set_title("Image", fontsize=9)
            axes[row][1].axis("off")

            order = np.argsort(probs)[::-1]
            cols = ["#2e7d32" if i == pred else "#bbbbbb" for i in order]
            axes[row][2].barh([CLASS_NAMES[i] for i in order][::-1],
                              [probs[i] for i in order][::-1], color=cols[::-1])
            axes[row][2].set_xlim(0, 1); axes[row][2].tick_params(labelsize=7)
            axes[row][2].set_title("Probabilités", fontsize=9)

            lignes.append(f"| {name} | {pred_name} | {conf * 100:.1f}% | {verdict} |")
        except Exception as e:
            for c in range(3):
                axes[row][c].axis("off")
            axes[row][0].set_title(f"{name}\n⚠️ {type(e).__name__}", color="red", fontsize=8)
            axes[row][0].text(0.5, 0.5, str(e)[:60], ha="center", va="center", fontsize=7)
            lignes.append(f"| {name} | ⚠️ {type(e).__name__} | — | — |")

    fig.tight_layout()
    entete = "| Modèle | Prédiction | Confiance | |\n|---|---|---|---|\n"
    vrai = f"**Vraie classe : `{CLASS_NAMES[true_label]}`**\n\n" if true_label is not None \
        else "*Image externe — pas de vérité terrain*\n\n"
    return fig, vrai + entete + "\n".join(lignes)


def maj_galerie(classe):
    cls_idx = CLASS_NAMES.index(classe)
    pool = [i for i, (_, lab) in enumerate(SAMPLES) if lab == cls_idx][:12]
    return gr.update(value=[SAMPLES[i][0] for i in pool]), pool


def _sel_index(evt):
    """Index robuste : selon la version de Gradio, .index est un int ou [row, col]."""
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        idx = idx[0]
    return int(idx)


def sur_selection(evt: gr.SelectData, pool):
    if not pool:
        return None, "Choisis d'abord un type de globule pour peupler la galerie."
    path, label = SAMPLES[pool[_sel_index(evt)]]
    return analyse(Image.open(path), label)


def tirer_cas(ref_name, veut_erreur):
    entry = MODELS[ref_name]
    ref, meta = entry["model"], entry["meta"]
    order = list(range(len(SAMPLES)))
    random.shuffle(order)
    for i in order[:300]:
        path, label = SAMPLES[i]
        img = Image.open(path)
        pred = int(predict(ref, make_input(img, meta)).argmax())
        if (pred != label) == veut_erreur:
            return analyse(img, label)
    return None, "Aucun cas de ce type dans l'échantillon (essaie l'autre option)."


def analyser_upload(img):
    if img is None:
        return None, "Uploade une image."
    return analyse(img, None)


# ═════════════════════════════════════════════════════════════
# Évaluation d'un modèle (utilitaire commun)
# ═════════════════════════════════════════════════════════════
def eval_model(entry, order):
    """Retourne y_true, y_pred, confs, idxs — évaluation par batch (rapide)."""
    model, meta = entry["model"], entry["meta"]
    key = _pp_key(meta)
    y_pred, confs = [], []
    with torch.inference_mode():
        for _, batch in batch_iter(order, key):
            p = torch.softmax(model(batch.to(DEVICE)), 1)
            c, pr = p.max(1)
            y_pred.extend(pr.cpu().tolist()); confs.extend(c.cpu().tolist())
    y_true = np.array([SAMPLES[i][1] for i in order])
    return y_true, np.array(y_pred), np.array(confs), np.array(order)


# ═════════════════════════════════════════════════════════════
# 2) Performance — matrice de confusion + rapport
# ═════════════════════════════════════════════════════════════
def matrice_confusion(mname, n_eval):
    from sklearn.metrics import confusion_matrix, classification_report
    y_true, y_pred, _, _ = eval_model(MODELS[mname], _order(n_eval))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    acc = (y_true == y_pred).mean()
    ax.set_title(f"Matrice de confusion — accuracy {acc * 100:.1f}%")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)
    return fig, f"```\n{rep}\n```"


# ═════════════════════════════════════════════════════════════
# 3) Espace latent — t-SNE 3D interactif (Plotly)
# ═════════════════════════════════════════════════════════════
def projection_tsne(mname, n_pts):
    from sklearn.manifold import TSNE
    import plotly.graph_objects as go
    import matplotlib.colors as mcolors

    entry = MODELS[mname]
    model, meta = entry["model"], entry["meta"]
    lin = head_linear(model)
    if lin is None:
        return None
    feats = {}
    hook = lin.register_forward_pre_hook(lambda m, inp: feats.__setitem__("v", inp[0].detach()))
    order = _order(n_pts)
    key = _pp_key(meta)
    X = []
    with torch.inference_mode():
        for _, batch in batch_iter(order, key):
            model(batch.to(DEVICE))
            X.append(feats["v"].cpu().numpy())        # [B, D]
    hook.remove()
    X = np.concatenate(X, 0)
    y = np.array([SAMPLES[i][1] for i in order])
    if len(X) < 10:
        return None
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    perp = min(30, max(5, len(Xn) // 4))
    emb = TSNE(n_components=3, perplexity=perp, init="pca",
               metric="cosine", random_state=0).fit_transform(Xn)

    fig = go.Figure()
    for c in range(len(CLASS_NAMES)):
        m = y == c
        col = mcolors.to_hex(PALETTE(c % 10))
        fig.add_trace(go.Scatter3d(
            x=emb[m, 0], y=emb[m, 1], z=emb[m, 2],
            mode="markers", name=CLASS_NAMES[c],
            marker=dict(size=4, color=col, opacity=0.85),
            hovertemplate=f"{CLASS_NAMES[c]}<extra></extra>"))
    fig.update_layout(
        title=f"t-SNE 3D — {mname}",
        showlegend=True, legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0), height=650,
        scene=dict(xaxis=dict(showticklabels=False, title=""),
                   yaxis=dict(showticklabels=False, title=""),
                   zaxis=dict(showticklabels=False, title="")))
    return fig


# ═════════════════════════════════════════════════════════════
# 4) Dashboard comparatif (tous les modèles)
# ═════════════════════════════════════════════════════════════
def dashboard(n_eval):
    from sklearn.metrics import accuracy_score, f1_score
    import pandas as pd
    order = _order(n_eval)

    rows = []
    for name, entry in MODELS.items():
        y_true, yp, _, _ = eval_model(entry, order)
        rows.append({
            "Modèle": name.split(" · ")[0],
            "Archi": name.split(" · ")[-1],
            "Accuracy %": round(accuracy_score(y_true, yp) * 100, 1),
            "F1 macro %": round(f1_score(y_true, yp, average="macro", zero_division=0) * 100, 1),
            "Params": nb_params(entry["model"]),
        })
    df = pd.DataFrame(rows).sort_values("F1 macro %", ascending=False).reset_index(drop=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(df)); w = 0.38
    a1.bar(x - w / 2, df["Accuracy %"], w, label="Accuracy", color="#1976d2")
    a1.bar(x + w / 2, df["F1 macro %"], w, label="F1 macro", color="#ef6c00")
    a1.set_xticks(x); a1.set_xticklabels(df["Modèle"], rotation=40, ha="right", fontsize=7)
    a1.set_ylabel("%"); a1.set_ylim(0, 100); a1.legend()
    a1.set_title("Performance par modèle"); a1.grid(alpha=0.3, axis="y")

    a2.scatter(df["Params"], df["F1 macro %"], s=90, edgecolor="k", color="#7b1fa2")
    for _, r in df.iterrows():
        a2.annotate(r["Modèle"], (r["Params"], r["F1 macro %"]), fontsize=6,
                    xytext=(4, 4), textcoords="offset points")
    a2.set_xscale("log"); a2.set_xlabel("Nombre de paramètres (log)")
    a2.set_ylabel("F1 macro (%)"); a2.set_title("Performance vs taille du modèle")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return df, fig


# ═════════════════════════════════════════════════════════════
# 5) Analyse d'erreurs avancée + Grad-CAM moyen
# ═════════════════════════════════════════════════════════════
def _montage(sel_idx, titre_couleur="red"):
    n = len(sel_idx)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "Aucun cas trouvé", ha="center"); ax.axis("off")
        return fig
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    for k, (i, vrai, pred, conf) in enumerate(sel_idx):
        axes[k].imshow(disp_from_pil(Image.open(SAMPLES[i][0]))); axes[k].axis("off")
        axes[k].set_title(f"Vrai : {CLASS_NAMES[vrai]}\n"
                          f"Prédit : {CLASS_NAMES[pred]} ({conf * 100:.0f}%)",
                          color=titre_couleur, fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def analyse_erreurs(mname, n_eval):
    from sklearn.metrics import confusion_matrix
    y_true, y_pred, confs, idxs = eval_model(MODELS[mname], _order(n_eval))

    # 1) Confusion normalisée (%)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES))).astype(float)
    cmn = cm / (cm.sum(1, keepdims=True) + 1e-9)
    fig_cm, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    ax.set_title("Matrice de confusion normalisée (% par vraie classe)")
    fig_cm.colorbar(im, fraction=0.046); fig_cm.tight_layout()

    # 2) Confusions les plus fréquentes
    pairs = [(cm[i, j], i, j) for i in range(len(CLASS_NAMES))
             for j in range(len(CLASS_NAMES)) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    md = "### Confusions les plus fréquentes\n\n| Vraie classe | Prédite | Nb |\n|---|---|---|\n"
    for cnt, i, j in pairs[:8]:
        md += f"| {CLASS_NAMES[i]} | {CLASS_NAMES[j]} | {int(cnt)} |\n"
    if not pairs:
        md += "| — | — | 0 |\n"

    # 3) Hall of shame : erreurs les plus confiantes
    err = np.where(y_true != y_pred)[0]
    err = err[np.argsort(-confs[err])][:8]
    sel = [(int(idxs[k]), int(y_true[k]), int(y_pred[k]), float(confs[k])) for k in err]
    fig_shame = _montage(sel, "red")
    return fig_cm, md, fig_shame


def gradcam_moyen(mname, n_per):
    """Carte d'attention moyenne par classe (Grad-CAM sur la vraie classe)."""
    entry = MODELS[mname]
    model, meta = entry["model"], entry["meta"]
    K = len(CLASS_NAMES)
    cams = {c: [] for c in range(K)}
    counts = {c: 0 for c in range(K)}
    for i in range(len(SAMPLES)):
        c = SAMPLES[i][1]
        if counts[c] >= n_per:
            continue
        t = make_input(Image.open(SAMPLES[i][0]), meta)
        cam, _ = grad_cam(model, t, class_idx=c)
        cams[c].append(_resize_cam(cam, 224))
        counts[c] += 1
        if all(v >= n_per for v in counts.values()):
            break
    cols = min(4, K); rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for c in range(K):
        axes[c].axis("off")
        if cams[c]:
            axes[c].imshow(np.mean(cams[c], 0), cmap="jet")
        axes[c].set_title(f"{CLASS_NAMES[c]}  (n={counts[c]})", fontsize=9)
    for ax in axes[K:]:
        ax.axis("off")
    fig.suptitle("Grad-CAM moyen par classe — zones d'attention typiques", fontsize=12)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════
# 6) Couche par couche (show after each layer) — générique
# ═════════════════════════════════════════════════════════════
def _fig_to_array(fig):
    """Rend une figure matplotlib en tableau RGB (pour gr.Gallery)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    arr = buf[..., :3].copy()
    plt.close(fig)
    return arr


def _grid_image(maps, title, cmap="viridis", max_maps=64):
    """Grille des feature maps d'une couche (une case par canal, échelle commune)."""
    t = maps.detach().cpu()
    H, W = t.shape[-2], t.shape[-1]
    m = t.reshape(-1, H, W)
    n = min(m.shape[0], max_maps)
    sub = m[:n]
    vmin, vmax = sub.min().item(), sub.max().item()
    cols = math.ceil(math.sqrt(n)); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(sub[i].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    extra = "" if m.shape[0] <= max_maps else f" (premiers {max_maps}/{m.shape[0]})"
    fig.suptitle(f"{title} — {m.shape[0]}×{H}×{W}{extra}", fontsize=10)
    fig.tight_layout()
    return _fig_to_array(fig)


def _bar_image(v, labels, title):
    """Graphe en barres d'un vecteur (façon show_vector).
    labels=None -> pas d'étiquettes en x (utile pour les vecteurs longs, ex. avgpool 512)."""
    v = np.asarray(v).flatten()
    fig, ax = plt.subplots(figsize=(max(6, len(v) * 0.03), 3))
    ax.bar(range(len(v)), v, color="#1976d2")
    if labels is not None and len(labels) == len(v):
        ax.set_xticks(range(len(v))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _fig_to_array(fig)


def maj_galerie_couches(classe):
    return maj_galerie(classe)


def couches_par_couche(evt: gr.SelectData, pool, mname, max_maps):
    if not MODELS or mname is None:
        return [], "Aucun modèle chargé."
    if not pool:
        return [], "Choisis d'abord un type de globule pour peupler la galerie."
    return _couches_core(pool[_sel_index(evt)], mname, max_maps)


def _couches_core(idx, mname, max_maps):
    path, label = SAMPLES[idx]
    entry = MODELS[mname]
    model, meta = entry["model"], entry["meta"]
    tensor = make_input(Image.open(path), meta).to(DEVICE)

    # Hooks : chaque Conv2d (dans l'ordre d'exécution) + le pooling global (avgpool)
    acts = []
    pooled = {}
    handles = []
    for nm, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(
                lambda m, i, o, name=nm: acts.append((name, o.detach()[0]))))
        elif isinstance(mod, nn.AdaptiveAvgPool2d):
            handles.append(mod.register_forward_hook(
                lambda m, i, o: pooled.__setitem__("v", o.detach()[0])))
    with torch.no_grad():
        logits = model(tensor)
    for h in handles:
        h.remove()

    logits_np = logits.squeeze(0).cpu().numpy()          # avant softmax
    probs = torch.softmax(logits, 1).squeeze().cpu().numpy()
    pred = int(probs.argmax())

    galerie = [(disp_from_pil(Image.open(path)), f"entrée · vrai = {CLASS_NAMES[label]}")]
    for nm, a in acts:
        galerie.append((_grid_image(a, nm, max_maps=int(max_maps)), nm))
    if "v" in pooled:                                     # ResNet avgpool, ConfigCNN gap...
        avg_vec = pooled["v"].flatten().cpu().numpy()     # moyenne par canal (512 pour ResNet)
        galerie.append((_bar_image(avg_vec, None,
                                   f"avgpool — {len(avg_vec)} moyennes par canal"), "avgpool"))
    # logits (avant softmax) et probabilités (après) — en barres, avec les classes
    galerie.append((_bar_image(logits_np, CLASS_NAMES, "logits (avant softmax)"), "logits"))
    galerie.append((_bar_image(probs, CLASS_NAMES, "probabilités (après softmax)"), "probas"))

    ok = "✅" if pred == label else "❌"
    pool_txt = " · avgpool inclus" if "v" in pooled else ""
    md = (f"### {mname}\n\n{ok} Prédit **{CLASS_NAMES[pred]}** ({probs[pred] * 100:.1f}%) · "
          f"vrai **{CLASS_NAMES[label]}** · {len(acts)} couches conv{pool_txt} · logits + probas.")
    return galerie, md


# ═════════════════════════════════════════════════════════════
# Interface Gradio
# ═════════════════════════════════════════════════════════════
with gr.Blocks(title="Démo Globules Blancs (CPU)", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🔬 Reconnaissance de globules blancs — démo CPU\n"
                f"Comparaison de **{len(MODEL_NAMES)}** modèle(s) · calcul sur **CPU** · "
                f"chaque modèle avec son propre pré-traitement.")

    with gr.Tab("🎯 Analyse interactive"):
        pool_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1 · Choisis une image")
                cls = gr.Dropdown(CLASS_NAMES, label="Type de globule",
                                  value=CLASS_NAMES[0] if CLASS_NAMES else None)
                galerie = gr.Gallery(label="Clique sur une image pour l'analyser",
                                     columns=4, height=260, allow_preview=False)
                gr.Markdown("**ou** tire un cas au hasard :")
                ref = gr.Dropdown(MODEL_NAMES, label="Modèle de référence",
                                  value=MODEL_NAMES[0] if MODEL_NAMES else None)
                veut_err = gr.Radio(["Prédiction correcte", "Erreur"],
                                    value="Erreur", label="Type de cas")
                btn_cas = gr.Button("🎲 Tirer un exemple")
                gr.Markdown("**ou** uploade ta propre image :")
                up = gr.Image(type="pil", label="Image externe")
            with gr.Column(scale=2):
                gr.Markdown("### 2 · Verdict de chaque modèle")
                resume = gr.Markdown()
                sortie = gr.Plot(label="Grad-CAM + probabilités")

        cls.change(maj_galerie, cls, [galerie, pool_state])
        galerie.select(sur_selection, pool_state, [sortie, resume])
        btn_cas.click(lambda r, v: tirer_cas(r, v == "Erreur"),
                      [ref, veut_err], [sortie, resume])
        up.change(analyser_upload, up, [sortie, resume])

    with gr.Tab("🧠 Couche par couche"):
        gr.Markdown("### Activations après chaque couche convolutive\n"
                    "Choisis un modèle et une image : la galerie montre l'entrée, "
                    "la grille de feature maps après **chaque** conv, puis les probabilités.")
        pool_state_l = gr.State([])
        with gr.Row():
            ml = gr.Dropdown(MODEL_NAMES, label="Modèle",
                             value=MODEL_NAMES[0] if MODEL_NAMES else None)
            cls_l = gr.Dropdown(CLASS_NAMES, label="Type de globule",
                                value=CLASS_NAMES[0] if CLASS_NAMES else None)
            max_maps_l = gr.Slider(9, 128, value=64, step=1,
                                   label="Feature maps max par couche")
        gal_pick = gr.Gallery(label="Clique sur une image", columns=6,
                              height=160, allow_preview=False)
        md_layers = gr.Markdown()
        gal_layers = gr.Gallery(label="Couche par couche", columns=3,
                                height=620, object_fit="contain")
        cls_l.change(maj_galerie_couches, cls_l, [gal_pick, pool_state_l])
        gal_pick.select(couches_par_couche, [pool_state_l, ml, max_maps_l],
                        [gal_layers, md_layers])

    with gr.Tab("📊 Performance"):
        with gr.Row():
            mconf = gr.Dropdown(MODEL_NAMES, label="Modèle",
                                value=MODEL_NAMES[0] if MODEL_NAMES else None)
            n_eval = gr.Slider(50, min(1500, len(SAMPLES)), value=min(300, len(SAMPLES)),
                               step=50, label="Images évaluées")
        btn_perf = gr.Button("Évaluer")
        plot_cm = gr.Plot()
        rap = gr.Markdown()
        btn_perf.click(matrice_confusion, [mconf, n_eval], [plot_cm, rap])

    with gr.Tab("🗺️ Espace latent (t-SNE 3D)"):
        gr.Markdown("Projection 3D interactive : rotation clic-glissé, zoom molette, survol des points.")
        with gr.Row():
            mtsne = gr.Dropdown(MODEL_NAMES, label="Modèle",
                                value=MODEL_NAMES[0] if MODEL_NAMES else None)
            n_pts = gr.Slider(100, min(1000, len(SAMPLES)), value=min(400, len(SAMPLES)),
                              step=100, label="Nombre d'images (CPU : plus = plus lent)")
        btn_tsne = gr.Button("Projeter (t-SNE 3D)")
        plot_tsne = gr.Plot()
        btn_tsne.click(projection_tsne, [mtsne, n_pts], plot_tsne)

    with gr.Tab("🏆 Dashboard comparatif"):
        gr.Markdown("### Tous les modèles sur les mêmes critères")
        n_dash = gr.Slider(50, min(1500, len(SAMPLES)), value=min(300, len(SAMPLES)),
                           step=50, label="Images évaluées")
        btn_dash = gr.Button("Comparer tous les modèles")
        tab_dash = gr.Dataframe(label="Récapitulatif (trié par F1 macro)")
        plot_dash = gr.Plot()
        btn_dash.click(dashboard, n_dash, [tab_dash, plot_dash])

    with gr.Tab("🔎 Analyse d'erreurs"):
        with gr.Row():
            merr = gr.Dropdown(MODEL_NAMES, label="Modèle",
                               value=MODEL_NAMES[0] if MODEL_NAMES else None)
            n_err = gr.Slider(50, min(1500, len(SAMPLES)), value=min(300, len(SAMPLES)),
                              step=50, label="Images évaluées")
        btn_err = gr.Button("Analyser les erreurs")
        with gr.Row():
            plot_cmn = gr.Plot(label="Confusion normalisée")
            md_pairs = gr.Markdown()
        gr.Markdown("#### Hall of shame — erreurs les plus confiantes (les plus instructives)")
        plot_shame = gr.Plot()
        btn_err.click(analyse_erreurs, [merr, n_err], [plot_cmn, md_pairs, plot_shame])

        gr.Markdown("---\n#### Grad-CAM moyen par classe *(un peu long sur CPU)*")
        n_per = gr.Slider(5, 30, value=10, step=5, label="Images par classe")
        btn_mean = gr.Button("Calculer le Grad-CAM moyen")
        plot_mean = gr.Plot()
        btn_mean.click(gradcam_moyen, [merr, n_per], plot_mean)

    # ─────────────────────────────────────────────────────────
    # Au chargement : un exemple aléatoire sur chaque onglet
    # (tailles modestes pour un démarrage rapide en CPU)
    # ─────────────────────────────────────────────────────────
    def _rand_model():
        return random.choice(MODEL_NAMES) if MODEL_NAMES else None

    def _n(cap):
        return min(cap, len(SAMPLES)) if SAMPLES else cap

    def init_analyse():
        if not MODELS or not SAMPLES:
            return None, "Aucun modèle / donnée."
        i = random.randrange(len(SAMPLES))
        fig, md = analyse(Image.open(SAMPLES[i][0]), SAMPLES[i][1])
        return fig, md

    def init_couches():
        if not MODELS or not SAMPLES:
            return gr.update(), gr.update(), [], [], "Aucun modèle / donnée."
        mname = _rand_model()
        gal_update, pool = maj_galerie(CLASS_NAMES[0])
        i = pool[random.randrange(len(pool))] if pool else random.randrange(len(SAMPLES))
        gallery, md = _couches_core(i, mname, 64)
        return gr.update(value=mname), gal_update, pool, gallery, md

    def init_perf():
        if not MODELS:
            return gr.update(), None, ""
        mname = _rand_model()
        fig, md = matrice_confusion(mname, _n(150))
        return gr.update(value=mname), fig, md

    def init_tsne():
        if not MODELS:
            return gr.update(), None
        mname = _rand_model()
        return gr.update(value=mname), projection_tsne(mname, _n(300))

    def init_dash():
        if not MODELS:
            return None, None
        return dashboard(_n(150))

    def init_err():
        if not MODELS:
            return gr.update(), None, "", None
        mname = _rand_model()
        a, b, c = analyse_erreurs(mname, _n(150))
        return gr.update(value=mname), a, b, c

    if CLASS_NAMES:
        demo.load(lambda: maj_galerie(CLASS_NAMES[0]), None, [galerie, pool_state])
    demo.load(init_analyse, None, [sortie, resume])
    demo.load(init_couches, None, [ml, gal_pick, pool_state_l, gal_layers, md_layers])
    demo.load(init_perf, None, [mconf, plot_cm, rap])
    demo.load(init_tsne, None, [mtsne, plot_tsne])
    demo.load(init_dash, None, [tab_dash, plot_dash])
    demo.load(init_err, None, [merr, plot_cmn, md_pairs, plot_shame])


if __name__ == "__main__":
    demo.launch()          # ajoute share=True pour un lien public gradio.live

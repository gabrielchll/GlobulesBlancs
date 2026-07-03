# Reconnaissance de globules blancs — CNN & démo interactive

Classification de cellules sanguines (8 classes : basophil, eosinophil, erythroblast,
ig, lymphocyte, monocyte, neutrophil, platelet) sur le dataset **Barcelona**.

Le projet entraîne et compare plusieurs architectures (ResNet18 finetune / from scratch /
feature-extractor, un CNN maison paramétrable `ConfigCNN` en balayage largeur/profondeur,
`MonCNN`, `MonCNN2`, `Net`, `CNN_gab`), puis fournit une démo Gradio qui les compare
en direct, image par image, avec Grad-CAM, matrices de confusion, espace latent t-SNE 3D
et visualisation couche par couche.

Deux fichiers :

- **`train_all.py`** : entraîne (ou réutilise) tous les modèles, les sauvegarde dans
  `modeles/`, puis teste chacun en CPU.
- **`app_demo_cpu.py`** : la démo interactive.

---

## 1. Prérequis

### Python


```bash
pip install -r requirements.txt
```

### Données
Le dataset doit être rangé ainsi :

```
barcelona/barcelona/
├── train/<classe>/*.jpg
└── val/<classe>/*.jpg
```

Cela se fait automatiquement en téléchargeant les données sous forme d'un dossier zip, et en le dézipant avec ce code : 

```
import zipfile

with zipfile.ZipFile("barcelona.zip", "r") as zf:
  zf.extractall("barcelona")
```


## 2. Entraîner tous les modèles

```bash
python train_all.py
```

Ce script, en un seul lancement :

1. réutilise les poids existants et entraîne les autres modèles ;
2. sauvegarde chaque modèle dans `modeles/<nom>.pt` (poids + méta de pré-traitement + accuracy) ;
3. recharge chaque modèle en CPU et affiche son accuracy sur le val-set.

L'entraînement tourne sur le GPU si disponible, sinon CPU. Le test final est toujours CPU.

Une fois `modeles/` rempli, plus besoin de réentraîner : la démo ne fait que charger
ces fichiers.

---

## 3. Lancer la démo

```bash
python app_demo_cpu.py
```

En ouvrant l'URL locale affichée (`http://127.0.0.1:7860`), la démo charge tous les
`modeles/*.pt` et tourne exclusivement sur CPU.

Chaque modèle est pré-traité avec sa propre normalisation / taille d'entrée
(lue dans sa méta) — indispensable car les architectures n'attendent pas la même entrée
(ResNet 224px ImageNet, `Net` 180px norm 0.5, `CNN_gab` 180px norm « GB »).

### Onglets
- **Analyse interactive** : Grad-CAM + probabilités de chaque modèle sur une image.
- **Couche par couche** : feature maps après chaque conv, avgpool, logits, probas.
- **Performance** : matrice de confusion + rapport par classe (par modèle).
- **Espace latent** : projection 3D interactive (Plotly).
- **Dashboard comparatif** : accuracy / F1 / nb de paramètres de tous les modèles.
- **Analyse d'erreurs** : confusion normalisée, hall of shame, Grad-CAM moyen.

Au chargement, chaque onglet affiche automatiquement un exemple aléatoire.


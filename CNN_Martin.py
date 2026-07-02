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
import torchvision
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import os
import time


# %%
class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 32, 5)
        self.fc1 = torch.nn.Linear(11552, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)
        self.drop = torch.nn.Dropout(p=0.2, inplace=False)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)


# %%
def imshow(img):
    plt.imshow(np.transpose((img / 2 + 0.5).numpy(), (1, 2, 0)))
    plt.show()


# %%
device = torch.device("cuda:1")

# %%
transform = v2.Compose([v2.ToImage(), v2.CenterCrop(180), v2.ToDtype(torch.float32, scale=True), v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.ImageFolder(root='./barcelona/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True, num_workers=0)
testset = torchvision.datasets.ImageFolder(root='./barcelona/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0) # batch_size réduit à 16 pour l'affichage
validset = torchvision.datasets.ImageFolder(root='./barcelona/valid', transform=transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=16, shuffle= False, num_workers=0)
    
classes = testset.classes
nom_des_classes = ('platelet', 'neutrophil', 'monocyte', 'lymphocyte',
               'ig', 'erythroblast', 'eosinophil', 'basophil')
    # 3. Chargement instantané du modèle entraîné
net = Net(num_classes=len(nom_des_classes))
net = net.to(device)
    # On ne charge le fichier QUE s'il existe déjà sur le disque
PATH = './barcelona_net.pth'

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay= 1e-4)


# %%
def training(net, criterion, optimizer, num_epochs):
    history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc': [], 'valid_acc': []}
    # 1. Chargement des poids si le fichier existe déjà
    if os.path.exists(PATH):
        print("Chargement des poids existants...")
        net.load_state_dict(torch.load(PATH))
    else:
        print("Aucun fichier de poids trouvé. Entraînement à partir de zéro.")

    print("Début de l'entraînement...")
    
    for epoch in range(num_epochs): 
        print(f'\n--- Époque {epoch + 1}/{num_epochs} ---')
        
        # Chaque époque a une phase d'entraînement et une phase de validation
        for phase in ['train', 'valid']:
            if phase == 'train':
                net.train()  # Mode entraînement (active dropout/batchnorm)
                dataloader = trainloader  # On utilise le trainloader global
            else:
                net.eval()   # Mode évaluation (désactive dropout/batchnorm)
                dataloader = validloader  # On utilise le validloader global

            running_loss = 0.0
            running_corrects = 0
            total_images = 0

            # Boucle sur les données de la phase en cours
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Réinitialiser les gradients uniquement pendant l'entraînement
                optimizer.zero_grad()

                # On ne calcule les gradients QUE si on est en phase de train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)

                    # Si on est en train, on fait la rétropropagation et la mise à jour
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistiques cumulées du batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (predicted == labels).sum().item()
                total_images += labels.size(0)
                

            # Calcul des scores finaux pour la phase actuelle
            epoch_loss = running_loss / total_images
            epoch_acc = (running_corrects / total_images) * 100
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['valid_loss'].append(epoch_loss)
                history['valid_acc'].append(epoch_acc)

            # 3. On affiche le résumé dans la console
            print(f'[{phase.upper()}] Loss : {epoch_loss:.4f} | Précision : {epoch_acc:.1f}%')

    print('\nFinished Training')
    
    # Sauvegarde finale
    torch.save(net.state_dict(), PATH)
    print(f"Modèle sauvegardé sous : {PATH}")
    
    return net, history
    
    

# %%
debut=time.perf_counter()
# On lance l'entraînement sur 15 ou 20 époques pour voir une vraie évolution
net, mon_historique = training(net, criterion, optimizer, num_epochs=10)
fin=time.perf_counter()
print("Durée", fin-debut)


# %%
def test(model, criterion, optimizer):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_images = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (predicted == labels).sum().item()
            total_images += labels.size(0)
    test_acc = (running_corrects / total_images) * 100
    print(f"\n====== RÉSULTATS SUR LE TEST SET ======")
    print(f"Précision Test : {test_acc:.2f}%")
    return test_acc


# %%
net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
acc_finale = test(net, criterion, testloader)
print(acc_finale)

# %%
epochs_range = range(1, len(mon_historique['train_acc']) + 1)

plt.figure(figsize=(12, 5))

# --- GRAPHIQUE 1 : L'ÉVOLUTION DE LA PRÉCISION (ACCURACY) ---
plt.subplot(1, 2, 1)
plt.plot(epochs_range, mon_historique['valid_acc'], label='Précision Validation (Valid)', color='orange', marker='s')
plt.title('Évolution de la Précision au fil des Époques')
plt.xlabel('Époques')
plt.ylabel('Précision (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%

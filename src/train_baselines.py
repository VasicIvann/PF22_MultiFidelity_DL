# --- Fichier : src/train_baselines.py ---
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler

# --- Configuration ---
BASE_DIR = "/content/processed_multifidelity"
RESULTS_DIR = "/content/drive/MyDrive/UTBM_PF22/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Définition des coûts unitaires
COST_HF = 10
COST_BF = 1

# --- 1. Custom Transform pour le Test BF à la volée ---
class AddDegradationTransform:
    """Applique le flou et le bruit à la volée pour évaluer sur le domaine BF"""
    def __call__(self, img_tensor):
        # 1. Baisse de résolution et upscale
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        img_degraded = transforms.functional.resize(img_tensor, (64, 64), antialias=True)
        img_degraded = transforms.functional.resize(img_degraded, (h, w), antialias=True)
        # 2. Ajout de Bruit Gaussien
        noise = torch.randn_like(img_degraded) * 0.15
        img_noisy = img_degraded + noise
        return torch.clamp(img_noisy, 0, 1)

def run_baseline(mode, epochs=10, batch_size=64, lr=0.001):
    print(f"\n{'='*50}\n🚀 DÉMARRAGE BASELINE : {mode}\n{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Entraînement en cours sur : {device}")
    
    # --- 2. Préparation des Datasets ---
    # Transformations basiques pour ResNet
    transform_standard = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformations pour le Test BF (On dégrade d'abord, on normalise ensuite)
    transform_bf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        AddDegradationTransform(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Chargement des données d'entraînement selon le mode
    dataset_hf = datasets.ImageFolder(os.path.join(BASE_DIR, 'train/HF'), transform=transform_standard)
    dataset_bf = datasets.ImageFolder(os.path.join(BASE_DIR, 'train/BF'), transform=transform_standard)
    
    if mode == "HF":
        train_dataset = dataset_hf
        cost_per_epoch = len(train_dataset) * COST_HF
    elif mode == "BF":
        train_dataset = dataset_bf
        cost_per_epoch = len(train_dataset) * COST_BF
    elif mode == "MIXTE":
        train_dataset = ConcatDataset([dataset_hf, dataset_bf])
        cost_per_epoch = (len(dataset_hf) * COST_HF) + (len(dataset_bf) * COST_BF)
    else:
        raise ValueError("Le mode doit être 'HF', 'BF' ou 'MIXTE'")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Chargement des données de test (Le même dossier, mais lu avec 2 transforms différents)
    test_hf_loader = DataLoader(datasets.ImageFolder(os.path.join(BASE_DIR, 'test'), transform=transform_standard), batch_size=batch_size, shuffle=False)
    test_bf_loader = DataLoader(datasets.ImageFolder(os.path.join(BASE_DIR, 'test'), transform=transform_bf_test), batch_size=batch_size, shuffle=False)

    print(f"📦 Images d'entraînement : {len(train_dataset)}")
    print(f"💰 Coût par époque : {cost_per_epoch} CA | Coût total estimé : {cost_per_epoch * epochs} CA")

    # --- 3. Initialisation du Modèle (ResNet-18) ---
    model = models.resnet18(weights=None) # weights=None car on entraîne "from scratch"
    model.fc = nn.Linear(model.fc.in_features, 10) # 10 classes d'animaux
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler('cuda') # Pour le FP16 (accélération)

    # --- 4. Boucle d'entraînement ---
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Époque {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    training_time = time.time() - start_time
    print(f"⏱️ Entraînement terminé en {training_time/60:.2f} minutes.")

    # --- 5. Phase d'Évaluation sur les 3 Domaines ---
    model.eval()
    def evaluate(loader, name):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"📊 Précision {name} : {acc:.2f}%")
        return acc

    print("\n--- RÉSULTATS D'ÉVALUATION ---")
    acc_hf = evaluate(test_hf_loader, "Test HF (Propre)")
    acc_bf = evaluate(test_bf_loader, "Test BF (Bruité)")
    acc_mixte = (acc_hf + acc_bf) / 2
    print(f"📊 Précision Mixte (Moyenne) : {acc_mixte:.2f}%")

    # --- 6. Sauvegarde des résultats ---
    results = {
        "mode": mode,
        "epochs": epochs,
        "total_cost_CA": cost_per_epoch * epochs,
        "training_time_sec": training_time,
        "accuracy_HF": acc_hf,
        "accuracy_BF": acc_bf,
        "accuracy_Mixte": acc_mixte
    }
    
    # Enregistrer le JSON
    with open(f"{RESULTS_DIR}/results_baseline_{mode}.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Enregistrer les poids du modèle
    torch.save(model.state_dict(), f"{RESULTS_DIR}/model_baseline_{mode}.pth")
    print(f"💾 Résultats et Modèle sauvegardés dans {RESULTS_DIR}")

# Permet d'importer ce script sans le lancer automatiquement
if __name__ == "__main__":
    pass
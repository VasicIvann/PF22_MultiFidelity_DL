# --- Fichier : src/train_baselines.py ---
import os
import sys
import time
import json
import random
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler

# --- Modules partagés (source unique dégradation + coût) ---
# Ils se trouvent dans le même dossier que ce script (src/), ajouté au sys.path.
# Sur Colab, ce dossier est /content/drive/MyDrive/UTBM_PF22/src.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from degradation import hf_transform, clean_tensor_transform, DegradedDataset
from cost import data_cost, unit_cost

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# --- Configuration ---
BASE_DIR = "/content/processed_multifidelity"
RESULTS_DIR = "/content/drive/MyDrive/UTBM_PF22/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Coûts unitaires d'acquisition (modèle de coût résolution², src/cost.py)
COST_HF = unit_cost(None)   # image HF pleine résolution (= 10 CA)
COST_BF = unit_cost(64)     # image BF canonique 64px (≈ 1 CA)

# Seeds par défaut pour le multi-seed (moyenne ± écart-type)
DEFAULT_SEEDS = (42, 1, 2)

# Configuration W&B par défaut
WANDB_PROJECT_DEFAULT = "PF22-MultiFidelity"


def _set_seed(seed):
    """Fixe toutes les sources d'aléa (reproductibilité par seed)."""
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_eval_once(mode, epochs, batch_size, lr, seed, device,
                     dataset_name, track_wandb, wandb_project, wandb_run_name):
    """Un entraînement + évaluation complet pour UN seed donné.

    Returns:
        (results_single: dict, model: nn.Module, costs: dict)
    """
    _set_seed(seed)

    # --- Datasets / transforms (via le module partagé) ---
    transform_hf = hf_transform()
    transform_clean = clean_tensor_transform()

    dataset_hf = datasets.ImageFolder(os.path.join(BASE_DIR, 'train/HF'), transform=transform_hf)
    dataset_bf = DegradedDataset(
        datasets.ImageFolder(os.path.join(BASE_DIR, 'train/BF'), transform=transform_clean),
        seeded=True,
    )

    if mode == "HF":
        train_dataset = dataset_hf
        data_cost_CA = data_cost(n_hf=len(dataset_hf))
    elif mode == "BF":
        train_dataset = dataset_bf
        data_cost_CA = data_cost(n_bf=len(dataset_bf))
    elif mode == "MIXTE":
        train_dataset = ConcatDataset([dataset_hf, dataset_bf])
        data_cost_CA = data_cost(n_hf=len(dataset_hf), n_bf=len(dataset_bf))
    else:
        raise ValueError("Le mode doit être 'HF', 'BF' ou 'MIXTE'")

    # Coût CALCUL (images vues, non pondéré) et coût TOTAL (pondéré = ancien coût).
    compute_images_seen = len(train_dataset) * epochs
    total_cost_CA = data_cost_CA * epochs

    # Décomposition par domaine (pour l'analyse de sensibilité au ratio HF:BF).
    n_hf_pool = len(dataset_hf) if mode in ("HF", "MIXTE") else 0
    n_bf_pool = len(dataset_bf) if mode in ("BF", "MIXTE") else 0
    hf_images_seen = n_hf_pool * epochs
    bf_images_seen = n_bf_pool * epochs

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_hf_loader = DataLoader(
        datasets.ImageFolder(os.path.join(BASE_DIR, 'test'), transform=transform_hf),
        batch_size=batch_size, shuffle=False)
    test_bf_loader = DataLoader(
        DegradedDataset(
            datasets.ImageFolder(os.path.join(BASE_DIR, 'test'), transform=transform_clean),
            seeded=True,
        ),
        batch_size=batch_size, shuffle=False)

    costs = {
        "data_cost_CA": data_cost_CA,
        "compute_images_seen": compute_images_seen,
        "total_cost_CA": total_cost_CA,
        "n_hf_pool": n_hf_pool,
        "n_bf_pool": n_bf_pool,
        "hf_images_seen": hf_images_seen,
        "bf_images_seen": bf_images_seen,
    }

    # --- W&B (un run par seed) ---
    if track_wandb:
        base_name = wandb_run_name or f"Baseline_{mode}"
        run_name = f"{base_name}_seed{seed}"
        tags = ["baseline", f"mode_{mode}", f"seed_{seed}"]
        if dataset_name:
            tags.append(dataset_name)
        wandb.init(
            project=wandb_project, name=run_name, tags=tags, reinit=True,
            config={
                "strategy": "baseline", "mode": mode,
                "dataset": dataset_name or "unknown", "architecture": "resnet18",
                "weights_init": "from_scratch", "num_classes": 10,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
                "seed": seed, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                "amp_fp16": True, "input_size": 224,
                "cost_HF_per_image": COST_HF, "cost_BF_per_image": COST_BF,
                "data_cost_CA": data_cost_CA, "compute_images_seen": compute_images_seen,
                "total_cost_CA": total_cost_CA, "train_size": len(train_dataset),
            },
        )

    # --- Modèle ResNet-18 from scratch ---
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler('cuda')

    # --- Boucle d'entraînement ---
    start_time = time.time()
    loss_history = []
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
        loss_history.append(epoch_loss)
        print(f"  [seed {seed}] Époque {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        if track_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_loss,
                "cumulative_images_seen": len(train_dataset) * (epoch + 1),
            })

    training_time = time.time() - start_time

    # --- Évaluation ---
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
        print(f"  [seed {seed}] 📊 {name} : {acc:.2f}%")
        return acc

    acc_hf = evaluate(test_hf_loader, "Test HF")
    acc_bf = evaluate(test_bf_loader, "Test BF")
    # Test Mixte = jeu mixte équilibré (chaque image en HF ET en BF, 2N préd.) :
    # moyenner les deux accuracies sur les mêmes N images équivaut exactement à
    # cette accuracy mixte (déterministe).
    acc_mixte = (acc_hf + acc_bf) / 2

    if track_wandb:
        wandb.log({
            "test/accuracy_HF": acc_hf, "test/accuracy_BF": acc_bf,
            "test/accuracy_Mixte": acc_mixte,
            "training_time_sec": training_time, "training_time_min": training_time / 60.0,
        })
        wandb.summary["final/accuracy_HF"] = acc_hf
        wandb.summary["final/accuracy_BF"] = acc_bf
        wandb.summary["final/accuracy_Mixte"] = acc_mixte
        wandb.summary["final/data_cost_CA"] = data_cost_CA
        wandb.summary["final/total_cost_CA"] = total_cost_CA
        wandb.summary["final/compute_images_seen"] = compute_images_seen
        wandb.finish()

    results_single = {
        "seed": seed,
        "accuracy_HF": acc_hf,
        "accuracy_BF": acc_bf,
        "accuracy_Mixte": acc_mixte,
        "training_time_sec": training_time,
        "loss_history": loss_history,
    }
    return results_single, model, costs


def _agg(values):
    """Retourne (moyenne, écart-type échantillon, liste des valeurs)."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std, list(values)


def run_baseline(mode, epochs=10, batch_size=64, lr=0.001,
                 dataset_name=None, use_wandb=True,
                 wandb_project=WANDB_PROJECT_DEFAULT, wandb_run_name=None,
                 seeds=DEFAULT_SEEDS):
    """Entraîne/évalue une baseline sur plusieurs seeds et agrège (moyenne ± std).

    Args:
        mode (str): "HF", "BF" ou "MIXTE".
        epochs, batch_size, lr: hyperparamètres.
        dataset_name (str|None): nom du dataset (tags/identification W&B).
        use_wandb (bool): tracking W&B (un run par seed).
        wandb_project, wandb_run_name: configuration W&B.
        seeds (tuple): seeds à exécuter (défaut: 3 seeds).

    Le JSON sauvegardé contient les accuracies MOYENNES (clé accuracy_HF/BF/Mixte,
    pour compatibilité avec les notebooks de bilan), leur écart-type (_std), les
    valeurs par seed (_seeds), ainsi que les 3 coûts. Le checkpoint du PREMIER seed
    est sauvegardé comme modèle canonique.
    """
    print(f"\n{'='*50}\n🚀 BASELINE : {mode} | seeds={list(seeds)}\n{'='*50}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Entraînement sur : {device}")

    track_wandb = bool(use_wandb and _WANDB_AVAILABLE)
    if use_wandb and not _WANDB_AVAILABLE:
        print("⚠️ wandb non installé : tracking désactivé.")

    per_seed = []
    costs = None
    canonical_model = None
    for i, seed in enumerate(seeds):
        res, model, costs = _train_eval_once(
            mode, epochs, batch_size, lr, seed, device,
            dataset_name, track_wandb, wandb_project, wandb_run_name)
        per_seed.append(res)
        if i == 0:
            canonical_model = model  # checkpoint canonique = premier seed

    # --- Agrégation moyenne ± écart-type ---
    hf_m, hf_s, hf_all = _agg([r["accuracy_HF"] for r in per_seed])
    bf_m, bf_s, bf_all = _agg([r["accuracy_BF"] for r in per_seed])
    mx_m, mx_s, mx_all = _agg([r["accuracy_Mixte"] for r in per_seed])
    t_m, t_s, _ = _agg([r["training_time_sec"] for r in per_seed])

    print(f"\n--- RÉSULTATS AGRÉGÉS ({len(seeds)} seeds) ---")
    print(f"📊 Test HF    : {hf_m:.2f} ± {hf_s:.2f} %")
    print(f"📊 Test BF    : {bf_m:.2f} ± {bf_s:.2f} %")
    print(f"📊 Test Mixte : {mx_m:.2f} ± {mx_s:.2f} %")
    print(f"💰 Coût données : {costs['data_cost_CA']:.0f} CA | 🧮 Calcul : {costs['compute_images_seen']} | 💵 Total : {costs['total_cost_CA']:.0f} CA")

    results = {
        "mode": mode,
        "dataset": dataset_name or "unknown",
        "epochs": epochs,
        "seeds": list(seeds),
        "data_cost_CA": costs["data_cost_CA"],
        "compute_images_seen": costs["compute_images_seen"],
        "total_cost_CA": costs["total_cost_CA"],
        "n_hf_pool": costs["n_hf_pool"],
        "n_bf_pool": costs["n_bf_pool"],
        "hf_images_seen": costs["hf_images_seen"],
        "bf_images_seen": costs["bf_images_seen"],
        "accuracy_HF": hf_m, "accuracy_HF_std": hf_s, "accuracy_HF_seeds": hf_all,
        "accuracy_BF": bf_m, "accuracy_BF_std": bf_s, "accuracy_BF_seeds": bf_all,
        "accuracy_Mixte": mx_m, "accuracy_Mixte_std": mx_s, "accuracy_Mixte_seeds": mx_all,
        "training_time_sec": t_m, "training_time_sec_std": t_s,
        "per_seed": per_seed,
    }

    json_path = f"{RESULTS_DIR}/results_baseline_{mode}.json"
    pth_path = f"{RESULTS_DIR}/model_baseline_{mode}.pth"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    torch.save(canonical_model.state_dict(), pth_path)
    print(f"💾 Résultats agrégés + modèle (seed {seeds[0]}) sauvegardés dans {RESULTS_DIR}")


# Permet d'importer ce script sans le lancer automatiquement
if __name__ == "__main__":
    pass
